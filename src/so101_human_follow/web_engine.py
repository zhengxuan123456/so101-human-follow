from __future__ import annotations

import os
import copy
import math
import logging
import queue
import re
import statistics
import threading
import time
from collections import deque
from collections.abc import Iterator
from contextlib import nullcontext
from dataclasses import dataclass, field, replace
from itertools import pairwise
from pathlib import Path
from typing import Any

from .agent_actions import (
    AgentKeyframe,
    AgentMotionPlan,
    AgentMotionTiming,
    AGENT_JOINTS,
    MAX_EXPLICIT_PLAN_SECONDS,
    VisualProximity,
    agent_frame_schedule,
    build_agent_motion_plan,
    calculate_agent_motion_timing,
    estimate_visual_proximity,
    step_toward_agent_target,
)
from .camera_device import camera_index_by_name
from .config import ALL_JOINTS, CONTROLLED_JOINTS, FIXED_JOINTS, AppConfig
from .motion_profile import cosine_progress
from .pose_detector import (
    MediaPipePoseDetector,
    PoseDetection,
    annotate_frame,
    orient_camera_frame,
)
from .pose_math import PoseFeatures
from .robot_adapter import SO101CalibrationSession, SO101RobotAdapter, preview_initial_positions
from .safe_range import (
    SafeRangeProfile,
    load_safe_range_profile,
    save_safe_range_profile,
)
from .safety import FEATURE_BY_JOINT, MotionController
from .teach_recording import TeachRecorder
from .teach_trajectory import action_metadata


CONTROL_FEATURES = tuple(dict.fromkeys(FEATURE_BY_JOINT.values()))
FIXED_POSITION_TIMEOUT_S = 10.0
RANGE_CAPTURE_TIMEOUT_S = 60.0
RANGE_POLL_INTERVAL_S = 0.20
BASE_CALIBRATION_TIMEOUT_S = 90.0
BASE_CALIBRATION_POLL_INTERVAL_S = 0.15
BASE_CALIBRATION_MIN_RAW_SPAN = 20
BASE_CALIBRATION_MOTORS = tuple(joint.removesuffix(".pos") for joint in ALL_JOINTS)
NEUTRAL_CAPTURE_TIMEOUT_S = 30.0
CONTROL_CONFIRMATION_CODE = "123"
AGENT_VISUAL_START_TIMEOUT_S = 15.0
AGENT_KEYFRAME_TIMEOUT_S = 15.0
AGENT_POSITION_TOLERANCE = 2.0  # LeRobot normalized positions, not degrees.
AGENT_BODY_FEEDBACK_TOLERANCE = 3.0
AGENT_PLAYBACK_TIMEOUT_S = 3.0
TRAJECTORY_TRACKING_TIMEOUT_S = 0.35
TEACH_MIN_CONTROL_HZ = 25.0
TEACH_TARGET_SMOOTHING_S = 0.04
TEACH_TARGET_MAX_AGE_S = 0.15


def _agent_feedback_tolerance(joint: str) -> float:
    # Loaded body joints can settle ~2.05 units away from their goal.  This is
    # an encoder arrival deadband, not an expansion of any travel/step limit.
    return (
        AGENT_POSITION_TOLERANCE if joint == "gripper.pos"
        else AGENT_BODY_FEEDBACK_TOLERANCE
    )


LOGGER = logging.getLogger("so101_human_follow.control")


def _valid_base_calibration_sample(joint: str, value: int) -> bool:
    """Return whether a raw position is safe to include in EEPROM limits."""
    if value < 0 or value >= (1 << 15):
        return False
    # ID5 carries the arm-mounted camera cable and must remain within one turn.
    if joint == "wrist_roll" and value > 4095:
        return False
    return True


class ControlError(RuntimeError):
    pass


@dataclass
class ControlCommand:
    name: str
    payload: dict[str, Any]
    client_id: str | None = None
    submitted_epoch: int = 0
    event: threading.Event = field(default_factory=threading.Event)
    result: dict[str, Any] | None = None
    error: str | None = None
    started: bool = False
    cancelled: bool = False
    state_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


class ControlEngine:
    """Sole serial owner with an independent optional camera/inference worker."""

    def __init__(
        self,
        config: AppConfig,
        *,
        robot_allowed: bool,
        lease_timeout_s: float = 2.0,
        profile_path: str | Path | None = None,
    ):
        self.config = config
        self.robot_allowed = robot_allowed
        self.lease_timeout_s = lease_timeout_s
        self.profile_path = None if profile_path is None else Path(profile_path)

        self._lock = threading.RLock()
        self._send_lock = threading.RLock()
        self._frame_condition = threading.Condition(self._lock)
        self._commands: queue.Queue[ControlCommand] = queue.Queue(maxsize=8)
        self._shutdown = threading.Event()
        self._motion_stop = threading.Event()
        self._motion_stop.set()
        self._thread: threading.Thread | None = None
        self._vision_thread: threading.Thread | None = None
        self._vision_stop: threading.Event | None = None
        self._camera_owner: str | None = None
        self._camera_name: str | None = None
        self._camera_session = 0
        self._epoch = 0

        self._preview_state = "STOPPED"
        self._preview_error: str | None = None
        self._robot_state = "DISABLED" if not robot_allowed else "DISCONNECTED"
        self._armed = False
        self._message = "Waiting to start"
        self._error: str | None = None

        self._operator_client: str | None = None
        self._operator_heartbeat = 0.0

        self._robot: SO101RobotAdapter | None = None
        self._connection_positions: dict[str, float] | None = None
        self._teach = TeachRecorder(config.robot)
        self._teach_client: str | None = None
        self._teach_stop_reason: str | None = None
        self._teach_last_send: float | None = None
        self._teach_target: tuple[float, dict[str, float]] | None = None
        self._teach_motion_started_at = 0.0
        self._teach_leader_read_s = 0.0
        self._teach_target_stale = False
        self._teach_error: str | None = None
        self._release_pending = False
        self._controller = MotionController(
            preview_initial_positions(), config.mapping, config.safety
        )
        self._initial = dict(self._controller.initial)
        self._last_action = self._controller.hold()
        self._last_send_time: float | None = None
        self._last_motion_telemetry_time = 0.0
        self._motion_telemetry_run: tuple[str, float] | None = None
        self._motion_telemetry_log_at = 0.0
        self._motion_telemetry_ticks: deque[tuple[float, bool]] = deque(maxlen=64)

        # Conversational Agent motion is a separate, semantic layer around the
        # existing controller.  It never owns the serial bus: keyframes are
        # advanced by this same control thread and sent through the same adapter.
        self._agent_plan: AgentMotionPlan | None = None
        self._agent_timing: AgentMotionTiming | None = None
        self._agent_speech_duration_s: float | None = None
        self._agent_media_duration_s: float | None = None
        self._agent_playback_rate = 1.0
        self._agent_playback: dict[str, Any] | None = None
        self._agent_audio_paused = False
        self._agent_frame_origin: dict[str, float] | None = None
        self._agent_frame_duration_s = 0.0
        self._agent_frame_hold_s = 0.0
        self._agent_frame_joint_schedule: dict[str, tuple[float, float]] = {}
        self._agent_frame_elapsed_s = 0.0
        self._agent_frame_tick_at: float | None = None
        self._agent_measured_speeds: dict[str, float] = {}
        self._agent_cycle_s: float | None = None
        self._agent_feedback_sample: tuple[float, dict[str, float], dict[str, float]] | None = None
        self._trajectory_error_since: dict[str, float] = {}
        self._agent_hold_positions: dict[str, float] = {}
        self._agent_semantic_actions: list[dict[str, Any]] = []
        self._agent_source_indices: list[int] = []
        self._agent_completed_cycles: dict[int, int] = {}
        self._agent_last_snapshot: dict[str, Any] = {}
        self._agent_plan_client: str | None = None
        self._agent_plan_epoch: int | None = None
        self._agent_plan_index = 0
        self._agent_plan_started_at: float | None = None
        self._agent_keyframe_reached_at: float | None = None
        self._agent_keyframe_started_at: float | None = None
        self._agent_visual_wait_started_at: float | None = None
        self._agent_recovery_positions: dict[str, float] | None = None
        self._agent_recovery_joints: tuple[str, ...] = ()
        self._agent_hold_pending = False
        self._agent_return_verified = False
        self._agent_motion_state = "IDLE"
        self._agent_motion_message = "No Agent motion is active"
        self._agent_motion_error: str | None = None
        self._pet_restore_positions: dict[str, float] | None = None
        self._visual_proximity = VisualProximity("UNKNOWN", "not started")
        self._visual_proximity_time: float | None = None
        self._last_safe_visual_time: float | None = None

        self._calibration_fingerprint: str | None = None
        self._safe_profile: SafeRangeProfile | None = None
        self._safe_profile_error: str | None = None
        self._pending_fixed_positions: dict[str, float] | None = None
        self._range_state = "IDLE"
        self._range_started_at: float | None = None
        self._range_last_poll = 0.0
        self._range_current: dict[str, float] | None = None
        self._range_mins: dict[str, float] = {}
        self._range_maxes: dict[str, float] = {}
        self._range_cancel_requested: str | None = None
        self._fixed_position_started_at: float | None = None

        self._base_calibration: SO101CalibrationSession | None = None
        self._base_calibration_state = "IDLE"
        self._base_calibration_started_at: float | None = None
        self._base_calibration_last_poll = 0.0
        self._base_calibration_current: dict[str, int] = {}
        self._base_calibration_mins: dict[str, int] = {}
        self._base_calibration_maxes: dict[str, int] = {}
        self._base_calibration_error: str | None = None
        self._base_calibration_cancel_requested: str | None = None

        self._detection = PoseDetection(None, (), 0, "not started")
        self._latest_features: PoseFeatures | None = None
        self._latest_valid_features: frozenset[str] = frozenset()
        self._feature_last_seen: dict[str, float | None] = {
            feature: None for feature in CONTROL_FEATURES
        }
        self._last_pose_time: float | None = None
        self._last_center_x: float | None = None
        self._valid_streak = 0
        self._neutral: PoseFeatures | None = None
        self._neutral_samples: dict[str, list[float]] = {
            feature: [] for feature in CONTROL_FEATURES
        }
        self._collecting_neutral = False
        self._neutral_started_at: float | None = None
        self._neutral_last_features: PoseFeatures | None = None
        self._neutral_collection_error: str | None = None

        self._latest_jpeg: bytes | None = None
        self._frame_revision = 0
        self._last_frame_time: float | None = None
        LOGGER.info(
            "engine initialized | robot_id=%s port=%s robot_allowed=%s log=%s",
            self.config.robot.id,
            self.config.robot.port,
            self.robot_allowed,
            Path(__file__).resolve().parents[2] / "logs" / "control.log",
        )

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._shutdown.clear()
            self._message = "正在启动本地控制服务。"
            self._error = None
            self._thread = threading.Thread(
                target=self._run,
                name="so101-web-control",
                daemon=False,
            )
            self._thread.start()
            LOGGER.info("control engine start requested")

    def shutdown(self, timeout_s: float = 10.0) -> None:
        LOGGER.info("control engine shutdown requested")
        self.disarm("server shutdown")
        self._shutdown.set()
        self.disconnect_camera()
        try:
            self._commands.put_nowait(ControlCommand("shutdown", {}))
        except queue.Full:
            pass
        thread = self._thread
        if thread is not None:
            thread.join(timeout_s)
            if thread.is_alive():
                LOGGER.error("control thread did not stop within %.1f seconds", timeout_s)
                print("WARNING: control thread did not stop; use the physical power switch if needed.")
        vision_thread = self._vision_thread
        if vision_thread is not None:
            vision_thread.join(min(timeout_s, 2.0))

    def heartbeat(self, client_id: str) -> bool:
        if not client_id:
            return False
        now = time.monotonic()
        disarm_reason = None
        with self._lock:
            expired = (
                self._operator_client is not None
                and now - self._operator_heartbeat > self.lease_timeout_s
            )
            if self._operator_client is None or expired:
                if expired:
                    # Invalidate commands from the previous lease even when the arm
                    # was already DISARMED.
                    self._motion_stop.set()
                    self._epoch += 1
                if expired and (self._armed or self._agent_plan is not None):
                    disarm_reason = "operator lease expired"
                # Native browser confirmation prompts pause JavaScript timers, so the
                # status heartbeat can legitimately expire while the operator is
                # confirming the next calibration step.  Keep torque-free/manual
                # calibration sessions alive; the next local request reacquires the
                # lease.  Powered following is still disarmed below when applicable.
                self._operator_client = client_id
                self._operator_heartbeat = now
                owner = True
            elif self._operator_client == client_id:
                self._operator_heartbeat = now
                owner = True
            else:
                owner = False
        if disarm_reason:
            LOGGER.warning("operator lease expired; disarming")
            self.disarm(disarm_reason)
        return owner

    def has_lease(self, client_id: str) -> bool:
        now = time.monotonic()
        with self._lock:
            return bool(
                client_id
                and self._operator_client == client_id
                and now - self._operator_heartbeat <= self.lease_timeout_s
            )

    def disarm(self, reason: str = "operator request") -> None:
        self._motion_stop.set()
        # If a serial write is already in flight, it cannot be interrupted. Holding this
        # gate guarantees that no stale write can start after disarm() returns.
        with self._send_lock:
            with self._lock:
                was_armed = self._armed
                was_agent_active = self._agent_plan is not None
                self._armed = False
                self._epoch += 1
                if was_agent_active:
                    self._cancel_agent_plan_locked(reason)
                if self._teach.sync_active and self._teach_stop_reason is None:
                    self._teach_stop_reason = reason
                if self._range_state in {
                    "POSITIONING_FIXED",
                    "STARTING",
                    "RECORDING",
                    "WAITING_RETURN",
                } and reason != (
                    "safe-range capture started"
                ):
                    self._range_cancel_requested = (
                        f"{reason}; range values were not saved"
                    )
                if self._base_calibration_state in {
                    "WAITING_MIDDLE",
                    "MIDDLE_CAPTURED",
                    "RECORDING",
                    "RANGE_INCOMPLETE",
                }:
                    self._base_calibration_cancel_requested = reason
                if self._robot is not None and self._robot_state not in {
                    "FAULT",
                    "DISCONNECTING",
                    "RANGE_RECORDING",
                }:
                    self._robot_state = "CONNECTED_DISARMED"
                if was_armed or reason != "operator request":
                    self._message = f"DISARMED: {reason}"
        LOGGER.info(
            "DISARM | reason=%s | was_armed=%s | agent_active=%s",
            reason,
            was_armed,
            was_agent_active,
        )

    def stop_following(self, reason: str = "operator pressed stop-following") -> None:
        """DISARM without making a false torque-hold promise during release."""

        with self._send_lock:
            with self._lock:
                release_in_progress = (
                    self._release_pending or self._robot_state == "DISCONNECTING"
                )
            self.disarm(reason)
            if release_in_progress:
                raise ControlError(
                    "Following was DISARMED, but torque release is already queued or in "
                    "progress and cannot be cancelled by STOP. Support the arm and use "
                    "the physical power switch if the situation is unsafe."
                )

    def submit(
        self,
        name: str,
        payload: dict[str, Any] | None = None,
        *,
        client_id: str | None = None,
        timeout_s: float = 30.0,
    ) -> dict[str, Any]:
        thread = self._thread
        if thread is None or not thread.is_alive():
            raise ControlError("Control thread is not running.")
        gate = self._send_lock if name == "disconnect" else nullcontext()
        with gate:
            with self._lock:
                if name == "disconnect":
                    expected = CONTROL_CONFIRMATION_CODE
                    if str((payload or {}).get("confirmation", "")) != expected:
                        raise ControlError(
                            f"Confirmation must exactly match: {expected}"
                        )
                    if not self._lease_valid_locked(client_id, time.monotonic()):
                        raise ControlError(
                            "The browser control lease expired before torque release."
                        )
                    if self._release_pending:
                        raise ControlError("Torque release is already queued or in progress.")
                    self._release_pending = True
                submitted_epoch = self._epoch
            if name == "disconnect":
                # Reserving a torque release must never leave pose following active while
                # the command waits in the worker queue.
                self.disarm("torque release queued")
            command = ControlCommand(
                name,
                payload or {},
                client_id=client_id,
                submitted_epoch=submitted_epoch,
            )
            try:
                self._commands.put_nowait(command)
            except queue.Full as exc:
                if name == "disconnect":
                    with self._lock:
                        self._release_pending = False
                raise ControlError("Control command queue is busy; try again.") from exc
        if not command.event.wait(timeout_s):
            with command.state_lock:
                command.cancelled = True
                started = command.started
            if started:
                raise ControlError(
                    f"Command '{name}' is still finishing. Check the live status before "
                    "issuing another command."
                )
            raise ControlError(f"Command '{name}' was cancelled before it started.")
        if command.error:
            raise ControlError(command.error)
        return command.result or {"ok": True}

    def status(self, client_id: str | None = None) -> dict[str, Any]:
        now = time.monotonic()
        with self._lock:
            pose_age = None if self._last_pose_time is None else max(0.0, now - self._last_pose_time)
            frame_age = None if self._last_frame_time is None else max(0.0, now - self._last_frame_time)
            send_age = None if self._last_send_time is None else max(0.0, now - self._last_send_time)
            lease_age = (
                None
                if self._operator_client is None
                else max(0.0, now - self._operator_heartbeat)
            )
            operator = bool(
                client_id
                and self._operator_client == client_id
                and lease_age is not None
                and lease_age <= self.lease_timeout_s
            )
            pose_fresh = pose_age is not None and pose_age <= self.config.safety.tracking_timeout_s
            feature_ages = {
                feature: (
                    None
                    if self._feature_last_seen[feature] is None
                    else max(0.0, now - self._feature_last_seen[feature])
                )
                for feature in CONTROL_FEATURES
            }
            valid_channels = sorted(
                set(self._latest_valid_features) & set(CONTROL_FEATURES)
            )
            feature_values = {
                feature: (
                    None
                    if self._latest_features is None
                    else float(getattr(self._latest_features, feature))
                )
                for feature in CONTROL_FEATURES
            }
            profile_ready = self._safe_profile is not None
            range_recording = self._range_state == "RECORDING"
            fixed_positioning = self._range_state == "POSITIONING_FIXED"
            range_elapsed = (
                0.0
                if self._range_started_at is None
                else max(0.0, now - self._range_started_at)
            )
            range_remaining = (
                max(0.0, RANGE_CAPTURE_TIMEOUT_S - range_elapsed)
                if range_recording
                else None
            )
            fixed_elapsed = (
                0.0
                if self._fixed_position_started_at is None
                else max(0.0, now - self._fixed_position_started_at)
            )
            fixed_remaining = (
                max(0.0, FIXED_POSITION_TIMEOUT_S - fixed_elapsed)
                if fixed_positioning
                else None
            )
            base_state = self._base_calibration_state
            base_active = base_state in {
                "STARTING",
                "WAITING_MIDDLE",
                "MIDDLE_CAPTURED",
                "RECORDING",
                "RANGE_INCOMPLETE",
                "SAVING",
                "CANCELLING",
            }
            base_elapsed = (
                0.0
                if self._base_calibration_started_at is None
                else max(0.0, now - self._base_calibration_started_at)
            )
            base_remaining = (
                max(0.0, BASE_CALIBRATION_TIMEOUT_S - base_elapsed)
                if base_state == "RECORDING"
                else None
            )
            neutral_elapsed = (
                0.0
                if self._neutral_started_at is None
                else max(0.0, now - self._neutral_started_at)
            )
            neutral_remaining = (
                max(0.0, NEUTRAL_CAPTURE_TIMEOUT_S - neutral_elapsed)
                if self._collecting_neutral
                else None
            )
            proximity_age = (
                None
                if self._visual_proximity_time is None
                else max(0.0, now - self._visual_proximity_time)
            )
            agent_active = self._agent_plan is not None
            agent_elapsed = (
                None
                if self._agent_plan_started_at is None
                else max(0.0, now - self._agent_plan_started_at)
            )
            can_start_agent_motion = bool(
                operator
                and self._robot is not None
                and self._robot.connected
                and self._robot_state == "CONNECTED_DISARMED"
                and not self._armed
                and not agent_active
                and not self._teach.active
                and self._safe_profile is not None
                and not self._collecting_neutral
                and self._range_state == "IDLE"
                and not base_active
                and not self._release_pending
                and self._error is None
            )
            can_arm = bool(
                operator
                and self._robot is not None
                and self._robot.connected
                and self._robot_state == "CONNECTED_DISARMED"
                and not agent_active
                and not self._teach.active
                and not self._release_pending
                and profile_ready
                and self._preview_state == "RUNNING"
                and self._neutral is not None
                and not self._collecting_neutral
                and self._latest_features is not None
                and pose_fresh
                and self._valid_streak >= self.config.safety.valid_frames_before_arm
                and self._error is None
            )

            joint_rows = []
            controlled = set(self.config.mapping.controlled_joints)
            for index, name in enumerate(ALL_JOINTS, start=1):
                role = "moving" if name in controlled else "fixed"
                mapping = {
                    "elbow_flex.pos": "human_elbow_angle",
                    "wrist_flex.pos": "human_upper_arm_lift",
                    "wrist_roll.pos": "hand_roll",
                    "gripper.pos": "hand_open",
                }.get(name, "fixed_at_startup")
                joint_rows.append(
                    {
                        "id": index,
                        "name": name,
                        "role": role,
                        "mapping": mapping,
                        "startup": self._initial.get(name),
                        "last_command": self._last_action.get(name),
                        "current": (
                            None if self._range_current is None else self._range_current.get(name)
                        ),
                        "safe_min": (
                            self._safe_profile.controlled_limits[name][0]
                            if self._safe_profile is not None and name in CONTROLLED_JOINTS
                            else self._range_mins.get(name)
                        ),
                        "safe_max": (
                            self._safe_profile.controlled_limits[name][1]
                            if self._safe_profile is not None and name in CONTROLLED_JOINTS
                            else self._range_maxes.get(name)
                        ),
                        "fixed_target": (
                            self._safe_profile.fixed_positions.get(name)
                            if self._safe_profile is not None
                            else (
                                None
                                if self._pending_fixed_positions is None
                                else self._pending_fixed_positions.get(name)
                            )
                        ),
                    }
                )

            return {
                "preview_state": self._preview_state,
                "preview_error": self._preview_error,
                "camera_name": self._camera_name,
                "camera_session": self._camera_session,
                "robot_allowed": self.robot_allowed,
                "robot_port": self.config.robot.port,
                "robot_state": self._robot_state,
                "robot_connected": bool(self._robot is not None and self._robot.connected),
                "teach_active": self._teach.active,
                "teach_following": self._teach.sync_active,
                "armed": self._armed,
                "message": self._message,
                "error": self._error,
                "pose_reason": self._detection.reason,
                "people_count": self._detection.people_count,
                "pose_valid": self._latest_features is not None and pose_fresh,
                "valid_channels": valid_channels,
                "feature_values": feature_values,
                "channel_age_s": feature_ages,
                "pose_age_s": pose_age,
                "frame_age_s": frame_age,
                "last_send_age_s": send_age,
                "valid_streak": self._valid_streak,
                "neutral_ready": self._neutral is not None,
                "neutral_collecting": self._collecting_neutral,
                "neutral_progress": min(self._neutral_progress_locked().values(), default=0),
                "neutral_channel_progress": self._neutral_progress_locked(),
                "neutral_required": self.config.pose.neutral_sample_count,
                "neutral_timeout_s": NEUTRAL_CAPTURE_TIMEOUT_S,
                "neutral_elapsed_s": neutral_elapsed if self._collecting_neutral else None,
                "neutral_remaining_s": neutral_remaining,
                "neutral_error": self._neutral_collection_error,
                "operator": operator,
                "lease_age_s": lease_age,
                "lease_timeout_s": self.lease_timeout_s,
                "can_arm": can_arm,
                "can_start_agent_motion": can_start_agent_motion,
                "visual_proximity": {
                    "state": self._visual_proximity.state,
                    "reason": self._visual_proximity.reason,
                    "shoulder_size": self._visual_proximity.shoulder_size,
                    "palm_size": self._visual_proximity.palm_size,
                    "age_s": proximity_age,
                    "metric_distance_available": False,
                },
                "agent_motion": {
                    "state": self._agent_motion_state,
                    "motion_id": self._agent_plan_epoch,
                    "active": agent_active,
                    "name": None if self._agent_plan is None else self._agent_plan.name,
                    "step_index": self._agent_plan_index,
                    "step_count": (
                        0 if self._agent_plan is None else len(self._agent_plan.keyframes)
                    ),
                    "elapsed_s": agent_elapsed,
                    "message": self._agent_motion_message,
                    "error": self._agent_motion_error,
                    "pet_pose_saved": self._pet_restore_positions is not None,
                    "recovery_available": self._agent_recovery_positions is not None,
                    "return_verified": self._agent_return_verified,
                    "timing": None if self._agent_timing is None else {
                        "speech_duration_s": self._agent_speech_duration_s,
                        "estimated_remaining_s": self._agent_timing.estimated_duration_s,
                        "speeds": dict(self._agent_timing.speeds),
                        "max_speeds": dict(self._agent_timing.max_speeds),
                        "effective_speeds": dict(self._agent_timing.effective_speeds),
                        "playback_state": None if self._agent_playback is None else self._agent_playback["state"],
                        "reason": self._agent_timing.reason,
                        "units": "normalized_positions_per_second",
                    },
                },
                "safe_range": {
                    "state": self._range_state,
                    "profile_ready": profile_ready,
                    "profile_error": self._safe_profile_error,
                    "test_mode": self.config.safety.skip_safe_range_calibration,
                    "skipped": self.config.safety.skip_safe_range_calibration,
                    "fixed_ready": self._pending_fixed_positions is not None,
                    "timeout_s": (
                        FIXED_POSITION_TIMEOUT_S
                        if fixed_positioning
                        else RANGE_CAPTURE_TIMEOUT_S
                    ),
                    "remaining_s": fixed_remaining if fixed_positioning else range_remaining,
                    "fixed_timeout_s": FIXED_POSITION_TIMEOUT_S,
                    "fixed_remaining_s": fixed_remaining,
                    "range_timeout_s": RANGE_CAPTURE_TIMEOUT_S,
                    "range_remaining_s": range_remaining,
                    "current": dict(self._range_current or {}),
                    "mins": dict(self._range_mins),
                    "maxes": dict(self._range_maxes),
                    "value_unit": "normalized_position",
                },
                "base_calibration": {
                    "state": base_state,
                    "active": base_active,
                    "error": self._base_calibration_error,
                    "timeout_s": BASE_CALIBRATION_TIMEOUT_S,
                    "remaining_s": base_remaining,
                    "current_raw": dict(self._base_calibration_current),
                    "mins_raw": dict(self._base_calibration_mins),
                    "maxes_raw": dict(self._base_calibration_maxes),
                },
                "can_connect": bool(
                    operator
                    and self.robot_allowed
                    and self._robot is None
                    and self._robot_state in {"DISCONNECTED", "FAULT"}
                    and not base_active
                    and not self._teach.active
                ),
                "can_start_base_calibration": bool(
                    operator
                    and self.robot_allowed
                    and self._robot is None
                    and not base_active
                    and not self._teach.active
                ),
                "can_capture_base_middle": bool(
                    operator and base_state == "WAITING_MIDDLE"
                ),
                "can_start_base_range": bool(
                    operator and base_state in {"MIDDLE_CAPTURED", "RANGE_INCOMPLETE"}
                ),
                "can_finish_base_calibration": bool(
                    operator and base_state == "RECORDING"
                ),
                "can_cancel_base_calibration": bool(operator and base_active),
                "can_collect_neutral": bool(
                    operator
                    and self._preview_state == "RUNNING"
                    and self._latest_features is not None
                    and not self._armed
                    and not agent_active
                    and not self._teach.active
                    and not self._release_pending
                    and not self._collecting_neutral
                    and (not self.robot_allowed or profile_ready)
                    and self._range_state == "IDLE"
                    and (
                        not self.robot_allowed
                        or self._robot_state == "CONNECTED_DISARMED"
                    )
                ),
                "can_disconnect": bool(
                    operator
                    and self._robot is not None
                    and not self._armed
                    and not self._release_pending
                    and self._robot_state in {"CONNECTED_DISARMED", "FAULT"}
                ),
                "can_capture_fixed": bool(
                    operator
                    and not self.config.safety.skip_safe_range_calibration
                    and self._robot is not None
                    and self._robot.connected
                    and self._robot_state == "CONNECTED_DISARMED"
                    and not self._armed
                    and not self._collecting_neutral
                    and self._range_state == "IDLE"
                ),
                "can_start_safe_range": bool(
                    False
                ),
                "can_finish_safe_range": bool(
                    operator
                    and not self.config.safety.skip_safe_range_calibration
                    and self._range_state == "WAITING_RETURN"
                ),
                "can_cancel_safe_range": bool(
                    operator
                    and not self.config.safety.skip_safe_range_calibration
                    and self._range_state
                    in {"POSITIONING_FIXED", "STARTING", "RECORDING", "WAITING_RETURN"}
                ),
                "release_pending": self._release_pending,
                "port": self.config.robot.port,
                "robot_id": self.config.robot.id,
                "camera": {
                    "index": self.config.camera.index,
                    "width": self.config.camera.width,
                    "height": self.config.camera.height,
                    "fps": self.config.camera.fps,
                    "rotation_degrees": self.config.camera.rotation_degrees,
                    "flip_horizontal": self.config.camera.flip_horizontal,
                    "flip_vertical": self.config.camera.flip_vertical,
                },
                "pose_side": self.config.pose.side,
                "control_hz": self.config.robot.control_hz,
                "joints": joint_rows,
                "frame_revision": self._frame_revision,
            }

    def camera_frames(self) -> Iterator[bytes]:
        with self._lock:
            session = self._camera_session
        revision = -1
        while True:
            with self._frame_condition:
                while (
                    session == self._camera_session
                    and self._preview_state in {"STARTING", "RUNNING"}
                    and not self._shutdown.is_set()
                ):
                    jpeg = self._latest_jpeg
                    if jpeg is not None and self._frame_revision > revision:
                        revision = self._frame_revision
                        break
                    self._frame_condition.wait()
                else:
                    return
            yield jpeg

    def _camera_status_locked(self) -> dict[str, Any]:
        return {
            "ok": True,
            "camera_name": self._camera_name,
            "camera_session": self._camera_session,
            "preview_state": self._preview_state,
            "preview_error": self._preview_error,
        }

    def connect_camera(self, name: str, *, client_id: str) -> dict[str, Any]:
        name = name.strip()
        with self._frame_condition:
            if self._shutdown.is_set() or self._thread is None or not self._thread.is_alive():
                raise ControlError("本地控制服务尚未就绪。")
            if not self._lease_valid_locked(client_id, time.monotonic()):
                raise ControlError("请先取得网页控制权，再连接相机。")
            if self._vision_thread is not None:
                raise ControlError("请先断开当前相机，并等待设备释放。")
            stop = threading.Event()
            self._vision_stop = stop
            self._camera_owner = client_id
            self._camera_session += 1
            self._camera_name = name or f"默认相机（编号 {self.config.camera.index}）"
            self._preview_state = "STARTING"
            self._preview_error = None
            self._clear_camera_tracking_locked("camera connecting")
            self._vision_thread = threading.Thread(
                target=self._vision_loop, args=(name, stop), name="so101-optional-vision", daemon=True,
            )
            self._vision_thread.start()
            return self._camera_status_locked()

    def disconnect_camera(self, *, expired_owner: str | None = None) -> dict[str, Any]:
        with self._send_lock, self._frame_condition:
            if expired_owner is not None and (
                expired_owner != self._camera_owner or self._lease_valid_locked(expired_owner, time.monotonic())
            ):
                return self._camera_status_locked()
            if self._vision_stop is not None:
                self._vision_stop.set()
            self._camera_owner = None
            if self._armed:
                self.disarm("相机已断开，人体跟随已停止。")
            self._clear_camera_tracking_locked("camera disconnected")
            self._preview_state = "STOPPING" if self._vision_thread is not None else "STOPPED"
            self._preview_error = None
            if self._preview_state == "STOPPED":
                self._camera_name = None
            return self._camera_status_locked()

    def _clear_camera_tracking_locked(self, reason: str) -> None:
        self._latest_jpeg = None
        self._last_frame_time = None
        self._detection = PoseDetection(None, (), 0, reason)
        self._latest_features = None
        self._latest_valid_features = frozenset()
        self._feature_last_seen = dict.fromkeys(CONTROL_FEATURES)
        self._last_pose_time = None
        self._last_center_x = None
        self._valid_streak = 0
        self._visual_proximity = VisualProximity("UNKNOWN", reason)
        self._visual_proximity_time = None
        self._last_safe_visual_time = None
        self._neutral = None
        self._clear_neutral_samples_locked()
        self._collecting_neutral = False
        self._neutral_started_at = None
        self._neutral_last_features = None
        self._neutral_collection_error = None
        self._frame_condition.notify_all()

    def _open_camera(self, name: str):
        import cv2

        backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
        index = camera_index_by_name(name) if name else self.config.camera.index
        camera = cv2.VideoCapture(index, backend)
        try:
            if not camera.isOpened():
                raise RuntimeError(f"无法打开相机「{name or index}」。请检查设备连接和占用情况。")
            if name and camera_index_by_name(name) != index:
                raise RuntimeError("连接期间相机列表发生变化，请重新连接。")
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.camera.width)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.camera.height)
            camera.set(cv2.CAP_PROP_FPS, self.config.camera.fps)
        except Exception:
            camera.release()
            raise
        return camera

    def _process_commands(self, limit: int = 4) -> None:
        for _ in range(limit):
            try:
                command = self._commands.get_nowait()
            except queue.Empty:
                return
            try:
                with command.state_lock:
                    if command.cancelled:
                        raise ControlError("Command was cancelled before execution.")
                    command.started = True
                LOGGER.info("command started | name=%s", command.name)
                if command.name == "shutdown":
                    command.result = {"ok": True}
                else:
                    self._authorize_command(command)
                    command.result = self._execute_command(command)
                LOGGER.info("command completed | name=%s", command.name)
            except Exception as exc:
                command.error = str(exc)
                LOGGER.exception("command failed | name=%s | error=%s", command.name, exc)
            finally:
                if command.name == "disconnect":
                    with self._lock:
                        self._release_pending = False
                command.event.set()
                self._commands.task_done()

    def _authorize_command(self, command: ControlCommand) -> None:
        now = time.monotonic()
        with self._lock:
            owns_lease = self._lease_valid_locked(command.client_id, now)
            current_epoch = self._epoch
        if not owns_lease:
            raise ControlError("The browser control lease expired before the command ran.")
        if command.name in {
            "arm",
            "agent_plan_start",
            "start_safe_range",
            "start_base_calibration",
            "start_base_range",
            "teach_calibration_start",
            "teach_recording_start",
        } and command.submitted_epoch != current_epoch:
            action = {
                "arm": "ARM",
                "agent_plan_start": "Agent motion",
                "start_safe_range": "Safe-range start",
                "start_base_calibration": "Base-calibration start",
                "start_base_range": "Base-range start",
                "teach_calibration_start": "Leader calibration",
                "teach_recording_start": "Teach recording",
            }[command.name]
            raise ControlError(f"{action} was cancelled by a newer STOP/DISARM action.")

    def _lease_valid_locked(self, client_id: str | None, now: float) -> bool:
        return bool(
            client_id
            and self._operator_client == client_id
            and now - self._operator_heartbeat <= self.lease_timeout_s
        )

    def _execute_command(self, command: ControlCommand) -> dict[str, Any]:
        name = command.name
        payload = command.payload
        if name.startswith("teach_"):
            return self._execute_teach_command(command)
        if self._teach.active and name in {
            "connect", "start_base_calibration", "capture_fixed", "start_safe_range",
            "collect_neutral", "arm", "agent_plan_start",
        }:
            raise ControlError("请先结束主臂校准或示教录制。")
        if name == "connect":
            return self._connect_robot(
                str(payload.get("confirmation", "")), client_id=command.client_id,
                port=payload.get("port"),
            )
        if name == "disconnect":
            return self._disconnect_robot(
                str(payload.get("confirmation", "")), client_id=command.client_id
            )
        if name == "collect_neutral":
            return self._start_neutral_collection()
        if name == "start_base_calibration":
            return self._start_base_calibration(
                str(payload.get("confirmation", "")),
                client_id=command.client_id,
                submitted_epoch=command.submitted_epoch,
                port=payload.get("port"),
            )
        if name == "capture_base_middle":
            return self._capture_base_middle(str(payload.get("confirmation", "")))
        if name == "start_base_range":
            return self._start_base_range(
                str(payload.get("confirmation", "")),
                client_id=command.client_id,
                submitted_epoch=command.submitted_epoch,
            )
        if name == "finish_base_calibration":
            return self._finish_base_calibration(str(payload.get("confirmation", "")))
        if name == "cancel_base_calibration":
            return self._cancel_base_calibration("operator cancelled base calibration")
        if name == "capture_fixed":
            return self._capture_fixed_positions(str(payload.get("confirmation", "")))
        if name == "start_safe_range":
            return self._start_safe_range(
                str(payload.get("confirmation", "")),
                client_id=command.client_id,
                submitted_epoch=command.submitted_epoch,
            )
        if name == "finish_safe_range":
            return self._finish_safe_range(str(payload.get("confirmation", "")))
        if name == "cancel_safe_range":
            return self._cancel_safe_range("operator cancelled safe-range capture")
        if name == "arm":
            return self._arm(
                str(payload.get("confirmation", "")),
                client_id=command.client_id,
                submitted_epoch=command.submitted_epoch,
            )
        if name == "agent_plan_start":
            return self._start_agent_plan(
                payload.get("actions"),
                client_id=command.client_id,
                submitted_epoch=command.submitted_epoch,
                speech_duration_s=payload.get("speech_duration_s"),
                media_duration_s=payload.get("media_duration_s"),
                playback_rate=payload.get("playback_rate", 1.0),
                semantic_actions=payload.get("semantic_actions"),
                source_indices=payload.get("source_indices"),
            )
        if name == "agent_interrupt_hold":
            self._hold_agent_if_pending()
            return {"ok": True, "message": "Interrupted motion held at measured pose."}
        if name == "agent_plan_cancel":
            return self._cancel_agent_plan("operator cancelled Agent motion")
        raise ControlError(f"Unknown control command: {name}")

    @staticmethod
    def _serial_port(value: Any) -> str:
        if isinstance(value, str):
            port = value.strip()
            if re.fullmatch(r"COM[1-9][0-9]{0,3}", port, re.IGNORECASE):
                return port.upper()
            if os.name != "nt" and re.fullmatch(r"/dev/[A-Za-z0-9_./:-]+", port):
                if Path(port).resolve().is_relative_to("/dev"):
                    return port
        raise ControlError("请输入有效串口，例如 COM4。")

    def _select_follower_port(self, value: Any) -> None:
        if value is not None:
            port = self._serial_port(value)
            self.config = replace(self.config, robot=replace(self.config.robot, port=port))

    def teach_status(self) -> dict[str, Any]:
        with self._send_lock, self._lock:
            adapter = self._robot
            status = self._teach.status()
            ranges = adapter.calibration_ranges() if adapter is not None else {}
            if not ranges and self._base_calibration_state == "COMPLETE":
                ranges = {
                    f"{motor}.pos": (low, self._base_calibration_maxes[motor])
                    for motor, low in self._base_calibration_mins.items()
                }
            return {
                **status,
                "robot_allowed": self.robot_allowed,
                "follower_port": self.config.robot.port,
                "follower_connected": bool(adapter is not None and adapter.connected),
                "follower_calibrating": self._base_calibration is not None,
                "follower_limits": self.agent_joint_limits(),
                "follower_calibration_ranges": ranges,
                "error": self._teach_error or status.get("error"),
            }

    def _execute_teach_command(self, command: ControlCommand) -> dict[str, Any]:
        if not self.robot_allowed:
            raise ControlError("预览模式可以编辑和回放动作。请用真机模式启动服务后再进行示教录制。")
        name, payload = command.name, command.payload
        with self._send_lock:
            if name in {"teach_calibration_middle", "teach_calibration_range", "teach_calibration_save"} and self._teach_client != command.client_id:
                raise ControlError("主臂校准会话已失效，请重新开始。")
            if name == "teach_recording_stop":
                return {"ok": True, "recording": self._stop_teach_recording()}
            if name == "teach_calibration_cancel":
                if self._teach.recording:
                    raise ControlError("正在示教录制，请点击结束录制。")
                self._teach.cancel()
            elif name == "teach_calibration_middle":
                self._teach.capture_middle()
            elif name == "teach_calibration_range":
                self._teach.start_range()
            elif name == "teach_calibration_save":
                self._teach.save_calibration()
            elif name in {"teach_calibration_start", "teach_recording_start"}:
                port = self._serial_port(payload.get("port"))
                sync = payload.get("sync", False) if name == "teach_recording_start" else False
                if type(sync) is not bool:
                    raise ControlError("录制模式无效。")
                with self._lock:
                    if self._armed or self._agent_plan is not None or self._agent_hold_pending:
                        raise ControlError("请先停止当前的机械臂动作。")
                    if self._base_calibration is not None or self._range_state != "IDLE" or self._release_pending:
                        raise ControlError("请先结束从臂校准或断开操作。")
                    if self._robot is not None and port.casefold() == self.config.robot.port.casefold():
                        raise ControlError("主臂和已连接的从臂使用了相同串口，请核对两台设备的端口。")
                    if self._teach.active:
                        raise ControlError("请先结束当前主臂操作。")
                    robot = self._robot
                    if sync and (robot is None or not robot.connected or self._robot_state != "CONNECTED_DISARMED"):
                        raise ControlError("请先连接从臂，再开启同步录制。")
                    if sync and "follower_port" in payload and self._serial_port(payload["follower_port"]) != self.config.robot.port:
                        raise ControlError("从臂串口与已连接设备不同，请先核对连接。")
                    if sync and (self._safe_profile is None or self._error is not None):
                        raise ControlError("同步录制需要从臂安全范围就绪，并且没有连接故障。")
                    if sync and payload.get("mapping_confirmed") is not True:
                        raise ControlError("请先确认六轴对应关系、旋转方向和夹爪开合方向。")
                if name == "teach_calibration_start":
                    self._teach.start_calibration(port)
                    self._teach_client = command.client_id
                else:
                    metadata = (action_metadata(payload.get("name"), payload.get("description", ""))
                                if "name" in payload or "description" in payload else None)
                    follower_start = None
                    if sync:
                        limits = self.agent_joint_limits()
                        observed = robot.read_positions()
                        for joint, (low, high) in limits.items():
                            if not low <= observed[joint] <= high:
                                raise ControlError(f"从臂 {joint} 当前值 {observed[joint]:.2f} 超出同步范围 {low:.2f}～{high:.2f}。")
                        follower_start = robot.hold_current()
                        with self._lock:
                            self._last_action = dict(follower_start)
                            self._range_current = dict(follower_start)
                        self._motion_stop.clear()
                    try:
                        self._teach.start_recording(
                            port, sync=sync, directions=payload.get("directions"), metadata=metadata,
                            follower_start=follower_start,
                            follower_calibration_fingerprint=self._calibration_fingerprint if sync else None,
                        )
                    except Exception:
                        if sync:
                            self._motion_stop.set()
                        raise
                    self._teach_client = command.client_id
                    self._teach_stop_reason = None
                    self._teach_last_send = None
                    self._teach_target = None
                    self._teach_motion_started_at = time.monotonic()
                    self._teach_target_stale = False
                    LOGGER.info(
                        "teach recording started | leader=%s follower=%s sync=%s follower_target_hz=%s",
                        port, self.config.robot.port, sync,
                        max(TEACH_MIN_CONTROL_HZ, self.config.robot.control_hz) if sync else None,
                    )
            else:
                raise ControlError(f"Unknown teach command: {name}")
            if not self._teach.active:
                self._teach_client = None
            self._teach_error = None
        return {"ok": True}

    def _finish_teach_motion(self, sync: bool) -> None:
        self._teach_client = None
        self._teach_stop_reason = None
        self._teach_last_send = None
        self._teach_target = None
        if not sync:
            return
        self._motion_stop.set()
        robot = self._robot
        if robot is None or not robot.connected or self._release_pending:
            return
        try:
            measured = robot.hold_current()
        except Exception as exc:
            LOGGER.exception("teach follower hold failed")
            self._teach.invalidate_follower_replay()
            with self._lock:
                self._error = self._teach_error = f"示教已停止，从臂保持失败：{exc}"
                self._robot_state = "FAULT"
            return
        with self._lock:
            self._range_current = dict(measured)
            self._last_action = dict(measured)
            self._controller.reset_to(measured)
        self._teach.record_follower_sample(measured, time.monotonic())

    def _stop_teach_recording(self) -> dict[str, Any]:
        self._finish_teach_motion(self._teach.sync_active)
        result = self._teach.stop_recording()
        LOGGER.info("teach recording stopped | duration_s=%s samples=%d", result["duration_s"], len(result["samples"]))
        return result

    def _poll_teach(self, now: float) -> None:
        if not self._teach.active:
            return
        with self._send_lock:
            sync = self._teach.sync_active
            try:
                with self._lock:
                    owns_lease = self._lease_valid_locked(self._teach_client, now)
                if self._teach.recording:
                    if not owns_lease or self._teach_stop_reason or (sync and self._motion_stop.is_set()):
                        LOGGER.info("teach recording interrupted | cause=%s",
                                    self._teach_stop_reason or ("lease expired" if not owns_lease else "STOP"))
                        sync = False  # The stop path handles the follower hold itself.
                        self._stop_teach_recording()
                        return
                elif not owns_lease:
                    self._teach.cancel("网页控制会话已断开")
                    self._teach_client = None
                    return
                poll_started_at = time.monotonic()
                positions = self._teach.poll(now)
                if sync and not self._teach.sync_active:
                    self._finish_teach_motion(sync)
                elif sync:
                    tick_at = time.monotonic()
                    if positions is not None:
                        self._teach_target = (tick_at, positions)
                        self._teach_leader_read_s = tick_at - poll_started_at
                        if self._teach_target_stale:
                            LOGGER.info("teach follower target refreshed | elapsed_s=%.2f",
                                        tick_at - self._teach_motion_started_at)
                            self._teach_target_stale = False
                    self._send_teach_if_due(tick_at)
                elif not self._teach.active:
                    self._teach_client = None
            except Exception as exc:
                LOGGER.exception(
                    "teach operation failed | sync=%s recording=%s leader_port=%s follower_port=%s last_send_age_ms=%s",
                    sync, self._teach.recording, self._teach.port, self.config.robot.port,
                    None if self._teach_last_send is None else
                    round((time.monotonic() - self._teach_last_send) * 1000, 1),
                )
                self._teach_error = str(exc)
                if sync:
                    self._teach.invalidate_follower_replay()
                self._finish_teach_motion(sync)
                try:
                    if self._teach.recording:
                        self._teach.stop_recording()
                    else:
                        self._teach.cancel()
                except Exception:
                    LOGGER.exception("teach cleanup failed")

    def _send_teach_if_due(self, now: float) -> None:
        target = self._teach_target
        if target is None or self._motion_stop.is_set():
            return
        if now - target[0] > TEACH_TARGET_MAX_AGE_S:
            if not self._teach_target_stale:
                LOGGER.warning("teach follower sends paused | cause=stale leader sample age_ms=%.1f elapsed_s=%.2f",
                               (now - target[0]) * 1000, now - self._teach_motion_started_at)
                self._teach_target_stale = True
            return
        period = 1 / max(TEACH_MIN_CONTROL_HZ, self.config.robot.control_hz)
        previous_send = self._teach_last_send
        elapsed = period if previous_send is None else now - previous_send
        if elapsed < period:
            return
        # A slow serial cycle must not accumulate an arbitrarily large step.
        self._send_teach_positions(target[1], now, min(elapsed, 2 * period), target[0])

    def _send_teach_positions(
        self, positions: dict[str, float], now: float, elapsed: float, target_at: float,
    ) -> None:
        robot = self._robot
        if robot is None or not robot.connected or self._release_pending:
            raise ControlError("同步录制已停止，从臂连接不可用。")
        limits = self.agent_joint_limits()
        with self._lock:
            previous = dict(self._last_action)
        blend = -math.expm1(-elapsed / TEACH_TARGET_SMOOTHING_S)
        targets = {
            joint: min(high, max(low, previous[joint] + blend * (positions[joint] - previous[joint])))
            for joint, (low, high) in limits.items()
        }
        step_limit = self.config.safety.max_step_per_cycle
        # Maintain the original joint speed ceiling when increasing write frequency.
        max_step = min(step_limit, step_limit * self.config.robot.control_hz * elapsed)
        goals, _ = step_toward_agent_target(
            previous, targets, fixed_positions={},
            max_step=max_step, speed_scale=1.0,
        )
        if self._motion_stop.is_set():
            return
        io_started_at = time.monotonic()
        sent, observed = robot.send_agent_targets(goals, ALL_JOINTS, {})
        io_finished_at = time.monotonic()
        self._teach_last_send = now
        self._teach.record_follower_sample(observed, io_started_at)
        with self._lock:
            self._last_action = dict(sent)
            self._range_current = dict(observed)
            self._last_send_time = now
        self._log_motion_telemetry(
            "teach", started_at=self._teach_motion_started_at, tick_at=io_started_at,
            io_elapsed_s=io_finished_at - io_started_at,
            control_hz=max(TEACH_MIN_CONTROL_HZ, self.config.robot.control_hz),
            target=positions, planned=goals, previous=previous, sent=sent,
            measured=observed, active_joints=ALL_JOINTS, wrote=True,
            details={"leader_read_ms": round(self._teach_leader_read_s * 1000, 1),
                     "leader_sample_age_ms": round((io_started_at - target_at) * 1000, 1)},
        )

    def agent_joint_limits(self) -> dict[str, tuple[float, float]]:
        """Agent/editor envelope; the live-follow four-axis profile stays intact."""
        with self._lock:
            limits = {joint: tuple(map(float, self.config.safety.absolute_limits[joint])) for joint in ALL_JOINTS}
            if self._safe_profile is not None:
                for joint, (low, high) in self._safe_profile.controlled_limits.items():
                    limits[joint] = (max(limits[joint][0], low), min(limits[joint][1], high))
            if any(not math.isfinite(low + high) or low >= high for low, high in limits.values()):
                raise ControlError("Invalid six-axis Agent safety limits.")
            return limits

    def agent_joint_status(self, client_id: str) -> dict[str, Any]:
        with self._lock:
            connected = bool(self._robot is not None and self._robot.connected)
            return {
                "ok": True, "connected": connected,
                "positions": dict(self._range_current or self._last_action),
                "limits": self.agent_joint_limits(), "units": "normalized_positions",
                "motion": {
                    "id": self._agent_plan_epoch,
                    "state": self._agent_motion_state,
                    "step": self._agent_plan_index + 1 if self._agent_plan is not None else 0,
                    "error": self._agent_motion_error,
                },
                "can_move": bool(connected and self._lease_valid_locked(client_id, time.monotonic())
                                 and self._robot_state == "CONNECTED_DISARMED" and self._agent_plan is None
                                 and not self._armed and not self._release_pending and not self._teach.active),
            }

    def interruption_snapshot(self) -> dict[str, Any]:
        """Copy semantic progress; raw resume frames remain local to the dispatcher."""
        with self._lock:
            plan = self._agent_plan
            if plan is None:
                return copy.deepcopy(self._agent_last_snapshot)
            actions = copy.deepcopy(self._agent_semantic_actions)
            if not actions:
                return {"motion_id": self._agent_plan_epoch, "state": self._agent_motion_state,
                        "pending_actions": [], "completed_actions": [],
                        "progress_note": "A local manual/editor motion was interrupted; do not replay it."}
            def source(frame):
                index = getattr(frame, "action_index", 0)
                return self._agent_source_indices[index] if index < len(self._agent_source_indices) else 0
            remaining = plan.keyframes[self._agent_plan_index:]
            pending_indices = {source(frame) for frame in remaining}
            completed = [action for index, action in enumerate(actions) if index not in pending_indices]
            pending = [action for index, action in enumerate(actions) if index in pending_indices]
            current_index = source(remaining[0]) if remaining else None
            done_cycles = self._agent_completed_cycles.get(current_index, 0)
            if remaining and self._agent_keyframe_reached_at is not None:
                done_cycles = max(done_cycles, getattr(remaining[0], "completed_cycles", 0))
            current = copy.deepcopy(actions[current_index]) if current_index is not None else None
            if current is not None and "count" in current.get("arguments", {}):
                current["arguments"]["count"] = max(0, int(current["arguments"]["count"]) - done_cycles)
                if current["arguments"]["count"] == 0:
                    completed.append(copy.deepcopy(actions[current_index]))
                    current = {"name": "return_to_action_start", "arguments": {"speed": "normal"}}
                for offset, index in enumerate(sorted(pending_indices)):
                    if index == current_index:
                        pending[offset] = copy.deepcopy(current)
            snapshot = {
                "motion_id": self._agent_plan_epoch, "state": self._agent_motion_state,
                "completed_actions": completed, "pending_actions": pending,
                "current_action": current,
                "progress_note": "Continue unfinished actions only. The current action is partially executed; completed cycles must not repeat.",
            }
            if current is not None:
                if any(frame.trajectory is not None and source(frame) == current_index for frame in plan.keyframes):
                    snapshot["pending_actions"] = [
                        copy.deepcopy(actions[index]) for index in sorted(pending_indices)
                        if index != current_index
                    ]
                    snapshot["current_action"] = None
                    snapshot["progress_note"] = (
                        "The current recorded action was interrupted and is not resumable. "
                        "Do not restart it unless the operator explicitly requests it."
                    )
                    return snapshot
                resume_frames = []
                for frame in remaining:
                    if source(frame) != current_index:
                        continue
                    hold = frame.hold_s
                    duration = frame.duration_s
                    if frame.joint_timing is not None:
                        joint_timing = (
                            self._remaining_agent_joint_timing_locked(frame, time.monotonic())
                            if frame is remaining[0] else
                            {joint: dict(timing) for joint, timing in frame.joint_timing.items()}
                        )
                    else:
                        joint_timing = None
                    if frame is remaining[0] and joint_timing is None:
                        if self._agent_keyframe_reached_at is not None:
                            hold = max(0.0, hold - (time.monotonic() - self._agent_keyframe_reached_at))
                        if duration is not None:
                            time_left = duration - self._agent_frame_elapsed_s
                            duration = max(0.05, time_left) if time_left > 0 else None
                    resume_frame = {
                        "targets": {joint: frame.targets[joint] for joint in frame.active_joints},
                        "hold_s": hold,
                        "duration_s": duration,
                        "completed_cycles": max(0, frame.completed_cycles - done_cycles),
                    }
                    if joint_timing is not None:
                        resume_frame["joint_timing"] = joint_timing
                    resume_frames.append(resume_frame)
                snapshot["_resume"] = {
                    "action": copy.deepcopy(current),
                    "frames": resume_frames,
                }
            return snapshot

    def _remaining_agent_joint_timing_locked(
        self, frame: AgentKeyframe, now: float,
    ) -> dict[str, dict[str, float]]:
        if not self._agent_frame_joint_schedule:
            return {joint: dict(timing) for joint, timing in frame.joint_timing.items()}
        elapsed = self._agent_frame_elapsed_s
        if self._agent_keyframe_reached_at is not None:
            elapsed += max(0.0, now - self._agent_keyframe_reached_at)
        # A resumed frame starts at encoder feedback and must pass the same
        # explicit-time limits. The speed cap may lengthen its ramp again.
        return {
            joint: {
                "duration_s": min(10.0, max(0.1, duration - elapsed)),
                "hold_s": max(0.0, hold - max(0.0, elapsed - duration)),
            }
            for joint, (duration, hold) in self._agent_frame_joint_schedule.items()
        }

    def _start_agent_plan(
        self,
        actions: Any,
        *,
        client_id: str | None,
        submitted_epoch: int | None,
        speech_duration_s: float | None = None,
        media_duration_s: float | None = None,
        playback_rate: float = 1.0,
        semantic_actions: list[dict[str, Any]] | None = None,
        source_indices: list[int] | None = None,
    ) -> dict[str, Any]:
        """Validate and schedule semantic keyframes without blocking the web call."""

        now = time.monotonic()
        with self._send_lock, self._lock:
            if not self._lease_valid_locked(client_id, now):
                raise ControlError("The browser control lease expired before Agent motion.")
            if submitted_epoch is None or submitted_epoch != self._epoch:
                raise ControlError("Agent motion was cancelled by a newer STOP/DISARM action.")
            if self._robot is None or not self._robot.connected:
                raise ControlError("Connect the robot before starting an Agent motion.")
            if self._robot_state != "CONNECTED_DISARMED" or self._armed:
                raise ControlError(
                    "Agent motion requires a connected, DISARMED robot; stop human following first."
                )
            if self._agent_plan is not None:
                raise ControlError("上一个机械臂动作仍在执行或回位，请完成后再发送，或先点击停止。")
            if self._safe_profile is None:
                raise ControlError("A saved/configured servo safety envelope is required.")
            if self._range_state != "IDLE" or self._base_calibration_state in {
                "STARTING",
                "WAITING_MIDDLE",
                "MIDDLE_CAPTURED",
                "RECORDING",
                "RANGE_INCOMPLETE",
                "SAVING",
                "CANCELLING",
            }:
                raise ControlError("Finish or cancel servo calibration before Agent motion.")
            if self._collecting_neutral or self._release_pending:
                raise ControlError("Finish the current robot operation before Agent motion.")
            if self._error is not None:
                raise ControlError(f"Resolve the runtime fault first: {self._error}")
            # Every request starts from an encoder snapshot, not a possibly
            # unfinished goal left by the previous request.
            current = (
                self._robot.hold_current() if hasattr(self._robot, "hold_current")
                else self._robot.read_positions()
            )
            if set(current) != set(ALL_JOINTS) or not all(
                math.isfinite(float(value)) for value in current.values()
            ):
                raise ControlError("无法读取完整有效的机械臂位置，本次动作没有启动。")
            current = {joint: float(current[joint]) for joint in ALL_JOINTS}
            # Visual recalibration may change the working pose in _initial.
            # The connection pose is a separate encoder snapshot for explicit return.
            home = dict(self._initial)
            names = {
                item.get("name") for item in (actions if isinstance(actions, (list, tuple)) else ())
                if isinstance(item, dict) and isinstance(item.get("name"), str)
            }
            if "return_to_neutral" in names:
                if self._connection_positions is None:
                    raise ControlError("没有本次从臂连接时的位置记录，请重新连接从臂。")
                home = dict(self._connection_positions)
            recovery_requested = "return_to_action_start" in names
            if recovery_requested:
                if len(actions) != 1:
                    raise ControlError("请单独发送“回到刚才的位置”，不要与其他动作混合。")
                if self._agent_recovery_positions is None:
                    raise ControlError("还没有记录上次动作的起始位置。")
                home = dict(self._agent_recovery_positions)
            limits = self.agent_joint_limits()
            fixed = {}
            pet_restore = (
                None
                if self._pet_restore_positions is None
                else dict(self._pet_restore_positions)
            )
            gripper_open_is_high = (
                self.config.mapping.signs.get("gripper.pos", 1.0) >= 0.0
            )

            try:
                recorded_actions = [
                    item for item in (actions if isinstance(actions, (list, tuple)) else ())
                    if isinstance(item, dict) and item.get("name") == "play_follower_trajectory"
                ]
                if recorded_actions:
                    for item in recorded_actions:
                        args = item.get("arguments")
                        if (not self._calibration_fingerprint or not isinstance(args, dict)
                                or args.get("calibration_fingerprint") != self._calibration_fingerprint):
                            raise ValueError("录制时的从臂校准与当前不一致，请重新同步录制。")
                    for motor_id, (joint, value) in enumerate(current.items(), 1):
                        low, high = limits[joint]
                        if not low <= value <= high:
                            raise ValueError(
                                f"ID{motor_id} 当前位置 {value:.2f} 超出回放范围 [{low:.2f}, {high:.2f}]。"
                            )
                plan = build_agent_motion_plan(
                    actions,
                    current_positions=current,
                    home_positions=home,
                    limits=limits,
                    fixed_positions=fixed,
                    gripper_open_is_high=gripper_open_is_high,
                    pet_restore_positions=pet_restore,
                    recovery_joints=self._agent_recovery_joints if recovery_requested else None,
                )
                plan = replace(plan, keyframes=tuple(
                    replace(frame, completed_cycles=int(actions[frame.action_index].get("_completed_cycles", frame.completed_cycles)))
                    for frame in plan.keyframes
                ))
                timing = calculate_agent_motion_timing(
                    plan, current,
                    max_step_per_cycle=self.config.safety.max_step_per_cycle,
                    max_relative_target=self.config.robot.max_relative_target,
                    base_control_hz=self.config.robot.control_hz,
                    speech_duration_s=speech_duration_s,
                    proximity_speed_scale=1.0,
                    measured_speeds=self._agent_measured_speeds,
                    cycle_time_s=self._agent_cycle_s,
                )
                if any(frame.trajectory is not None for frame in plan.keyframes):
                    plan = replace(plan, keyframes=tuple(
                        replace(frame, trajectory=frame.trajectory.limited_to(timing.effective_speeds))
                        if frame.trajectory is not None else frame
                        for frame in plan.keyframes
                    ))
                    timing = calculate_agent_motion_timing(
                        plan, current,
                        max_step_per_cycle=self.config.safety.max_step_per_cycle,
                        max_relative_target=self.config.robot.max_relative_target,
                        base_control_hz=self.config.robot.control_hz,
                        measured_speeds=self._agent_measured_speeds,
                        cycle_time_s=self._agent_cycle_s,
                    )
                    if timing.estimated_duration_s > MAX_EXPLICIT_PLAN_SECONDS:
                        raise ValueError("限速后的录制或组合动作过长，请缩短动作后再试。")
                for frame in plan.keyframes:
                    for joint in frame.active_joints:
                        low, high = limits[joint]
                        if not low <= frame.targets[joint] <= high:
                            raise ValueError(f"{joint} 的动作或回位目标超出安全范围。")
            except (TypeError, ValueError) as exc:
                raise ControlError(f"Invalid Agent motion plan: {exc}") from exc

            self._epoch += 1
            self._agent_plan = plan
            self._agent_timing = timing
            self._agent_speech_duration_s = speech_duration_s
            self._agent_media_duration_s = media_duration_s if media_duration_s is not None else speech_duration_s
            self._agent_playback_rate = playback_rate
            self._agent_hold_positions = dict(current)
            self._agent_semantic_actions = copy.deepcopy(semantic_actions or [])
            self._agent_source_indices = list(source_indices or range(len(actions)))
            self._agent_completed_cycles = {}
            self._agent_last_snapshot = {}
            self._agent_playback = None
            self._agent_audio_paused = False
            self._agent_frame_origin = None
            self._agent_frame_duration_s = 0.0
            self._agent_frame_hold_s = 0.0
            self._agent_frame_joint_schedule = {}
            self._agent_frame_elapsed_s = 0.0
            self._agent_frame_tick_at = None
            self._trajectory_error_since = {}
            self._agent_feedback_sample = None
            if not recovery_requested:
                self._agent_recovery_positions = dict(current)
                self._agent_recovery_joints = tuple(
                    joint for joint in ALL_JOINTS
                    if any(joint in frame.active_joints for frame in plan.keyframes)
                )
            self._agent_return_verified = False
            self._agent_hold_pending = False
            self._last_action = dict(current)
            self._range_current = dict(current)
            self._controller.reset_to(current)
            self._agent_plan_client = client_id
            self._agent_plan_epoch = self._epoch
            self._agent_plan_index = 0
            # Encoder reads and plan validation above may take a serial cycle.
            self._agent_plan_started_at = time.monotonic()
            self._agent_keyframe_reached_at = None
            self._agent_keyframe_started_at = None
            self._agent_visual_wait_started_at = None
            self._agent_motion_state = "RUNNING"
            self._agent_motion_error = None
            self._agent_motion_message = f"对话动作已开始：{plan.name}（不依赖视觉）"
            self._motion_stop.clear()
            if plan.pet_restore_to_save is not None:
                self._pet_restore_positions = dict(plan.pet_restore_to_save)
            self._message = self._agent_motion_message

        LOGGER.info(
            "agent motion scheduled | name=%s keyframes=%d visual_guard=disabled state=%s start_measured=%s",
            plan.name,
            len(plan.keyframes),
            self._agent_motion_state,
            current,
        )
        LOGGER.info(
            "agent timing | speech_s=%s estimated_s=%.2f hz=%.1f min_speeds=%s max_speeds=%s speeds=%s reason=%s",
            speech_duration_s, timing.estimated_duration_s, timing.control_hz,
            timing.min_speeds, timing.max_speeds, timing.speeds, timing.reason,
        )
        return {
            "ok": True,
            "message": self._agent_motion_message,
            "agent_motion": {
                "state": self._agent_motion_state,
                "motion_id": self._agent_plan_epoch,
                "name": plan.name,
                "step_count": len(plan.keyframes),
            },
        }

    def update_agent_playback(
        self, *, client_id: str, motion_id: int, position_s: float,
        duration_s: float, state: str, sequence: int, playback_rate: float = 1.0,
    ) -> bool:
        """Update the current motion's audio clock; never dispatch serial work here."""
        if not isinstance(state, str) or state not in {"playing", "paused", "ended"}:
            raise ValueError("Invalid playback state.")
        if isinstance(motion_id, bool) or not isinstance(motion_id, int):
            raise ValueError("Invalid motion identifier.")
        if isinstance(playback_rate, bool) or not isinstance(playback_rate, (int, float)) or not math.isfinite(playback_rate) or not 0.5 <= playback_rate <= 2.0:
            raise ValueError("Playback rate must be between 0.5 and 2.0.")
        if isinstance(sequence, bool) or not isinstance(sequence, int) or not 0 <= sequence < 2**31:
            raise ValueError("Invalid playback sequence.")
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(value) for value in (position_s, duration_s)):
            raise ValueError("Playback times must be finite numbers.")
        if not 0.1 <= duration_s <= 120 or not 0 <= position_s <= duration_s + 0.25:
            raise ValueError("Playback times are outside the allowed range.")
        if state == "ended" and position_s < duration_s - 0.25:
            raise ValueError("Audio has not reached its end position.")
        now = time.monotonic()
        with self._lock:
            if not self._lease_valid_locked(client_id, now):
                raise ControlError("The browser control lease expired.")
            if (self._agent_plan is None or self._agent_plan_client != client_id
                    or self._agent_plan_epoch != motion_id or self._motion_stop.is_set()):
                return False
            previous = self._agent_playback
            if previous is not None and (
                sequence <= previous["sequence"] or previous["state"] == "ended"
                or position_s + 0.05 < previous["position_s"]
            ):
                return False
            if self._agent_media_duration_s is None or abs(duration_s - self._agent_media_duration_s) > 0.5:
                return False
            self._agent_playback = {
                "position_s": min(position_s, duration_s), "duration_s": duration_s,
                "state": state, "sequence": sequence, "received_at": now,
                "playback_rate": playback_rate,
            }
            if previous is None or previous["state"] != state:
                LOGGER.info(
                    "agent playback clock | motion=%s state=%s position_s=%.3f duration_s=%.3f",
                    motion_id, state, position_s, duration_s,
                )
            return True

    def _agent_speech_remaining_locked(self, now: float) -> float | None:
        clock = self._agent_playback
        if clock is not None:
            rate = clock.get("playback_rate", 1.0)
            elapsed = max(0.0, now - clock["received_at"]) * rate if clock["state"] == "playing" else 0.0
            return 0.0 if clock["state"] == "ended" else max(
                0.0, (clock["duration_s"] - clock["position_s"] - elapsed) / rate,
            )
        return None if self._agent_speech_duration_s is None else max(
            0.0, self._agent_speech_duration_s - (now - self._agent_plan_started_at),
        )

    def _pause_agent_for_audio(self, now: float, plan: AgentMotionPlan) -> float:
        """Freeze phase/hold clocks and seed a paused goal from encoder feedback."""
        with self._send_lock, self._lock:
            if (self._agent_plan is not plan or self._motion_stop.is_set()
                    or not self._agent_playback or self._agent_playback["state"] != "paused"):
                return now
            delta = max(0.0, now - (self._agent_frame_tick_at or now))
            for attribute in ("_agent_keyframe_started_at", "_agent_keyframe_reached_at"):
                value = getattr(self, attribute)
                if value is not None:
                    setattr(self, attribute, value + delta)
            self._agent_frame_tick_at = now
            if not self._agent_audio_paused:
                held = self._robot.hold_current()
                if set(held) != set(ALL_JOINTS) or not all(math.isfinite(float(value)) for value in held.values()):
                    raise ControlError("Invalid encoder positions while pausing audio.")
                self._last_action = dict(held)
                self._range_current = dict(held)
                self._agent_hold_positions = dict(held)
                self._controller.reset_to(held)
                if plan.keyframes[self._agent_plan_index].trajectory is None:
                    self._agent_frame_origin = None
                self._trajectory_error_since.clear()
                self._agent_feedback_sample = None
                self._agent_audio_paused = True
                LOGGER.info("agent motion paused for audio | motion=%s", self._agent_plan_epoch)
            self._agent_motion_state = "PAUSED"
        return now

    def _log_motion_telemetry(
        self, source: str, *, started_at: float, tick_at: float, io_elapsed_s: float,
        control_hz: float, target: dict[str, float], planned: dict[str, float],
        previous: dict[str, float], sent: dict[str, float], measured: dict[str, float],
        active_joints: tuple[str, ...], wrote: bool, details: dict[str, Any] | None = None,
    ) -> None:
        """Summarize existing control ticks without adding reads or writes."""
        run = (source, started_at)
        ticks = self._motion_telemetry_ticks
        if self._motion_telemetry_run != run:
            self._motion_telemetry_run = run
            self._motion_telemetry_log_at = tick_at - 0.5
            ticks.clear()
        ticks.append((tick_at, wrote))
        if tick_at - self._motion_telemetry_log_at < 0.5:
            return
        self._motion_telemetry_log_at = tick_at

        intervals = sorted((after[0] - before[0]) * 1000 for before, after in pairwise(ticks))
        write_times = [at for at, written in ticks if written]
        sample_span = ticks[-1][0] - ticks[0][0]
        sample_hz = len(intervals) / sample_span if sample_span > 0 else None
        write_hz = (
            (len(write_times) - 1) / (write_times[-1] - write_times[0])
            if len(write_times) >= 2 and write_times[-1] > write_times[0] else None
        )
        # The adapter reads encoders before sending, so compare to the prior goal.
        tracking_error = {joint: round(previous[joint] - measured[joint], 2) for joint in active_joints}
        LOGGER.info(
            "%s motion telemetry | elapsed_s=%.2f target_hz=%.1f sample_hz=%s write_hz=%s "
            "cycle_p50_ms=%s cycle_p95_ms=%s io_ms=%.1f wrote=%s "
            "target=%s planned=%s prior_goal=%s sent=%s measured=%s tracking_error=%s details=%s",
            source, tick_at - started_at, control_hz,
            None if sample_hz is None else round(sample_hz, 1),
            None if write_hz is None else round(write_hz, 1),
            round(statistics.median(intervals), 1) if intervals else None,
            round(intervals[math.ceil(0.95 * len(intervals)) - 1], 1) if intervals else None,
            io_elapsed_s * 1000, wrote, target, planned, previous, sent, measured, tracking_error,
            details or {},
        )

    def _sample_agent_feedback_locked(
        self, observed: dict[str, float], sent: dict[str, float],
        active_joints: tuple[str, ...], now: float, timing: AgentMotionTiming,
    ) -> None:
        previous = self._agent_feedback_sample
        if previous is not None:
            then, positions, goals = previous
            delta = now - then
            if 0.005 <= delta <= 0.5:
                self._agent_cycle_s = delta if self._agent_cycle_s is None else self._agent_cycle_s * 0.8 + delta * 0.2
                for joint in active_joints:
                    cap = timing.max_speeds[joint] / timing.control_hz
                    demand = goals[joint] - positions[joint]
                    movement = observed[joint] - positions[joint]
                    # Paced/settled samples do not measure the motor's capacity.
                    if abs(demand) >= cap * 0.60 and movement * demand > 0 and abs(movement) >= 0.1:
                        speed = min(timing.max_speeds[joint], abs(movement) / delta)
                        old = self._agent_measured_speeds.get(joint, speed)
                        self._agent_measured_speeds[joint] = old * 0.8 + speed * 0.2
        self._agent_feedback_sample = (now, dict(observed), dict(sent))

    def _cancel_agent_plan_locked(self, reason: str) -> bool:
        if self._agent_plan is None:
            return False
        name = self._agent_plan.name
        self._agent_plan = None
        self._agent_plan_client = None
        self._agent_plan_epoch = None
        self._agent_playback = None
        self._agent_frame_origin = None
        self._agent_frame_joint_schedule = {}
        self._agent_frame_tick_at = None
        self._agent_audio_paused = False
        self._agent_feedback_sample = None
        self._agent_plan_index = 0
        self._agent_plan_started_at = None
        self._agent_keyframe_reached_at = None
        self._agent_keyframe_started_at = None
        self._agent_visual_wait_started_at = None
        self._agent_hold_pending = True
        self._agent_return_verified = False
        self._agent_motion_state = "CANCELLED"
        self._agent_motion_error = str(reason)
        self._agent_motion_message = f"Agent motion cancelled: {reason}"
        LOGGER.warning("agent motion cancelled | name=%s reason=%s", name, reason)
        return True

    def _cancel_agent_plan(self, reason: str) -> dict[str, Any]:
        with self._send_lock:
            with self._lock:
                cancelled = self._cancel_agent_plan_locked(reason)
                if cancelled:
                    self._epoch += 1
                    self._message = self._agent_motion_message
        return {
            "ok": True,
            "message": (
                self._agent_motion_message
                if cancelled
                else "No Agent motion is active."
            ),
        }

    def _complete_agent_plan_locked(self) -> None:
        plan = self._agent_plan
        if plan is None:
            return
        if plan.clear_pet_restore_on_complete:
            self._pet_restore_positions = None
        self._agent_hold_pending = False
        touched = {joint for frame in plan.keyframes for joint in frame.active_joints}
        self._agent_return_verified = bool(
            touched and self._agent_recovery_positions is not None and self._range_current is not None
            and all(
                abs(self._range_current[joint] - self._agent_recovery_positions[joint])
                <= _agent_feedback_tolerance(joint) for joint in touched
            )
        )
        name = plan.name
        actual_elapsed = (
            None if self._agent_plan_started_at is None
            else time.monotonic() - self._agent_plan_started_at
        )
        audio_remaining = self._agent_speech_remaining_locked(time.monotonic())
        self._agent_last_snapshot = {"motion_id": self._agent_plan_epoch, "state": "COMPLETE",
                                     "completed_actions": copy.deepcopy(self._agent_semantic_actions),
                                     "pending_actions": [], "progress_note": "All previous actions finished; do not repeat them."}
        self._agent_plan = None
        self._agent_plan_client = None
        self._agent_plan_epoch = None
        self._agent_playback = None
        self._agent_frame_origin = None
        self._agent_frame_joint_schedule = {}
        self._agent_frame_tick_at = None
        self._agent_audio_paused = False
        self._agent_feedback_sample = None
        self._agent_plan_index = 0
        self._agent_plan_started_at = None
        self._agent_keyframe_reached_at = None
        self._agent_keyframe_started_at = None
        self._agent_visual_wait_started_at = None
        self._agent_motion_state = "COMPLETE"
        self._agent_motion_error = None
        self._agent_motion_message = f"Agent motion complete: {name}"
        self._motion_stop.set()
        self._message = self._agent_motion_message
        LOGGER.info(
            "agent motion completed | name=%s actual_elapsed_s=%s speech_duration_s=%s return_verified=%s measured=%s audio_remaining_s=%s effective_speeds=%s",
            name, actual_elapsed, self._agent_speech_duration_s,
            self._agent_return_verified, self._range_current, audio_remaining,
            self._agent_measured_speeds,
        )

    def _hold_agent_if_pending(self) -> None:
        """The serial owner seeds an interrupted motion at its measured pose."""
        with self._send_lock:
            with self._lock:
                if not self._agent_hold_pending:
                    return
                self._agent_hold_pending = False
                robot = self._robot
                if robot is None or not robot.connected or self._release_pending:
                    return
                if getattr(robot, "manual_range_active", False) or getattr(robot, "fixed_positioning_active", False):
                    return
            try:
                measured = robot.hold_current() if hasattr(robot, "hold_current") else robot.read_positions()
                with self._lock:
                    self._range_current = dict(measured)
                    self._last_action = dict(measured)
                    self._controller.reset_to(measured)
                LOGGER.info("interrupted Agent goal held at measured pose | positions=%s", measured)
            except Exception as exc:
                LOGGER.exception("could not hold interrupted Agent motion")
                with self._lock:
                    self._agent_motion_state = "FAULT"
                    self._agent_motion_error = f"停止后的当前位置读取失败：{exc}"
                    self._error = self._agent_motion_error

    def _start_base_calibration(
        self,
        confirmation: str,
        *,
        client_id: str | None = None,
        submitted_epoch: int | None = None,
        port: Any = None,
    ) -> dict[str, Any]:
        expected = CONTROL_CONFIRMATION_CODE
        if not self.robot_allowed:
            raise ControlError("This server was started in preview-only mode.")
        if confirmation != expected:
            raise ControlError(f"Confirmation must exactly match: {expected}")
        with self._send_lock:
            with self._lock:
                if not self._lease_valid_locked(client_id, time.monotonic()):
                    raise ControlError("The browser control lease expired before calibration.")
                if submitted_epoch is None or submitted_epoch != self._epoch:
                    raise ControlError(
                        "Base calibration was cancelled by a newer STOP/DISARM action."
                    )
                if self._robot is not None or self._range_state != "IDLE":
                    raise ControlError("Disconnect the robot before base calibration.")
                if self._base_calibration_state in {
                    "STARTING",
                    "WAITING_MIDDLE",
                    "MIDDLE_CAPTURED",
                    "RECORDING",
                    "RANGE_INCOMPLETE",
                    "SAVING",
                }:
                    raise ControlError("Base calibration is already active.")
                self._select_follower_port(port)
                self._base_calibration_state = "STARTING"
                self._base_calibration_cancel_requested = None
                self._base_calibration_error = None
                self._robot_state = "CALIBRATING"
                self._message = f"Opening {self.config.robot.port} for base calibration"

            session = SO101CalibrationSession(self.config.robot)
            try:
                session.start()
                current = session.read_raw_positions()
            except Exception as exc:
                try:
                    session.cancel()
                except Exception:
                    pass
                with self._lock:
                    self._base_calibration_state = "FAULT"
                    self._robot_state = "DISCONNECTED"
                    self._base_calibration_error = str(exc)
                    self._error = f"Base calibration could not start: {exc}"
                    self._message = self._error
                raise ControlError(self._error) from exc

            now = time.monotonic()
            with self._lock:
                self._base_calibration = session
                self._base_calibration_state = "WAITING_MIDDLE"
                self._base_calibration_started_at = None
                self._base_calibration_last_poll = now
                self._base_calibration_current = dict(current)
                self._base_calibration_mins.clear()
                self._base_calibration_maxes.clear()
                self._error = None
                self._message = (
                    "All six servos are torque-free. Support the arm, place every joint near its middle, "
                    "then confirm the middle position."
                )
            LOGGER.info(
                "base calibration torque released | current_raw=%s",
                current,
            )
        return {"ok": True, "message": self._message}

    def _capture_base_middle(self, confirmation: str) -> dict[str, Any]:
        expected = CONTROL_CONFIRMATION_CODE
        if confirmation != expected:
            raise ControlError(f"Confirmation must exactly match: {expected}")
        with self._lock:
            session = self._base_calibration
            if session is None or self._base_calibration_state != "WAITING_MIDDLE":
                raise ControlError("Base calibration is not waiting for a middle position.")
        try:
            offsets, _ = session.capture_middle()
            current = session.read_raw_positions()
        except Exception as exc:
            self._fail_base_calibration(f"Could not capture middle position: {exc}")
            raise ControlError(self._message) from exc
        with self._lock:
            self._base_calibration_state = "MIDDLE_CAPTURED"
            self._base_calibration_started_at = None
            self._base_calibration_current = dict(current)
            self._base_calibration_mins.clear()
            self._base_calibration_maxes.clear()
            self._base_calibration_error = None
            self._message = (
                "Six-axis middle captured. Start the 90-second full-range recording next."
            )
        LOGGER.info(
            "base calibration middle captured | offsets=%s current_raw=%s",
            offsets,
            current,
        )
        return {"ok": True, "message": self._message}

    def _start_base_range(
        self,
        confirmation: str,
        *,
        client_id: str | None = None,
        submitted_epoch: int | None = None,
    ) -> dict[str, Any]:
        expected = CONTROL_CONFIRMATION_CODE
        if confirmation != expected:
            raise ControlError(f"Confirmation must exactly match: {expected}")
        now = time.monotonic()
        with self._lock:
            if not self._lease_valid_locked(client_id, now):
                raise ControlError("The browser control lease expired before range recording.")
            if submitted_epoch is None or submitted_epoch != self._epoch:
                raise ControlError(
                    "Base-range start was cancelled by a newer STOP/DISARM action."
                )
            session = self._base_calibration
            if session is None or self._base_calibration_state not in {
                "MIDDLE_CAPTURED",
                "RANGE_INCOMPLETE",
            }:
                raise ControlError("Confirm the six-axis middle before range recording.")
        try:
            current = session.read_raw_positions()
        except Exception as exc:
            self._fail_base_calibration(f"Could not start base range recording: {exc}")
            raise ControlError(self._message) from exc
        with self._lock:
            self._base_calibration_state = "RECORDING"
            self._base_calibration_started_at = now
            self._base_calibration_last_poll = now
            self._base_calibration_current = dict(current)
            self._base_calibration_mins = {
                joint: int(current[joint])
                for joint in BASE_CALIBRATION_MOTORS
                if _valid_base_calibration_sample(joint, int(current[joint]))
            }
            self._base_calibration_maxes = dict(self._base_calibration_mins)
            self._base_calibration_error = None
            self._message = (
                "Recording all six raw ranges for 90 seconds; move ID1-ID6 one by one."
            )
        LOGGER.info("base range recording started | duration_s=90 current_raw=%s", current)
        return {"ok": True, "message": self._message}

    def _finish_base_calibration(self, confirmation: str) -> dict[str, Any]:
        expected = CONTROL_CONFIRMATION_CODE
        if confirmation != expected:
            raise ControlError(f"Confirmation must exactly match: {expected}")
        return self._save_base_calibration(automatic=False)

    def _save_base_calibration(self, *, automatic: bool) -> dict[str, Any]:
        with self._lock:
            session = self._base_calibration
            if session is None or self._base_calibration_state != "RECORDING":
                raise ControlError("Base calibration range recording is not active.")
            mins = dict(self._base_calibration_mins)
            maxes = dict(self._base_calibration_maxes)
            too_small = [
                joint
                for joint in BASE_CALIBRATION_MOTORS
                if joint not in mins
                or joint not in maxes
                or maxes[joint] - mins[joint] < BASE_CALIBRATION_MIN_RAW_SPAN
            ]
            if too_small:
                names = ", ".join(too_small)
                message = (
                    f"These joints were not moved through enough range: {names}. "
                    "Run the 90-second range capture again."
                )
                if automatic:
                    self._base_calibration_state = "RANGE_INCOMPLETE"
                    self._base_calibration_started_at = None
                    self._base_calibration_error = message
                    self._message = message
                    LOGGER.warning(
                        "base range validation failed | joints=%s mins=%s maxes=%s",
                        names,
                        mins,
                        maxes,
                    )
                    return {"ok": False, "message": message, "missing": too_small}
                raise ControlError(message)
            self._base_calibration_state = "SAVING"
            self._message = "Saving six-axis base calibration"
        LOGGER.info(
            "base calibration save started | automatic=%s mins=%s maxes=%s",
            automatic,
            mins,
            maxes,
        )
        try:
            session.finish(mins, maxes)
        except Exception as exc:
            self._fail_base_calibration(f"Could not save base calibration: {exc}")
            raise ControlError(self._message) from exc
        with self._lock:
            self._base_calibration = None
            self._base_calibration_state = "COMPLETE"
            self._base_calibration_started_at = None
            self._base_calibration_cancel_requested = None
            self._base_calibration_error = None
            self._safe_profile = None
            self._pending_fixed_positions = None
            self._robot_state = "DISCONNECTED"
            self._error = None
            self._message = "Base calibration saved. You can now connect the robot."
        LOGGER.info("base calibration saved | automatic=%s", automatic)
        return {"ok": True, "message": self._message}

    def _cancel_base_calibration(self, reason: str) -> dict[str, Any]:
        LOGGER.warning("base calibration cancellation requested | reason=%s", reason)
        with self._lock:
            session = self._base_calibration
            if session is None:
                self._base_calibration_state = "IDLE"
                return {"ok": True, "message": "Base calibration is not active"}
            self._base_calibration_state = "CANCELLING"
        try:
            session.cancel()
        except Exception as exc:
            with self._lock:
                self._base_calibration_state = "FAULT"
                self._base_calibration_error = str(exc)
                self._robot_state = "FAULT"
                self._error = f"Calibration cleanup failed; turn off robot power: {exc}"
                self._message = self._error
            raise ControlError(self._error) from exc
        with self._lock:
            self._base_calibration = None
            self._base_calibration_state = "IDLE"
            self._base_calibration_started_at = None
            self._base_calibration_cancel_requested = None
            self._robot_state = "DISCONNECTED"
            self._message = f"Base calibration cancelled: {reason}"
        LOGGER.info("base calibration cancelled | reason=%s", reason)
        return {"ok": True, "message": self._message}

    def _fail_base_calibration(self, message: str) -> None:
        LOGGER.error("base calibration failed | error=%s", message)
        with self._lock:
            session = self._base_calibration
        cleanup_error = None
        if session is not None:
            try:
                session.cancel()
            except Exception as exc:
                cleanup_error = str(exc)
        with self._lock:
            self._base_calibration = None
            self._base_calibration_state = "FAULT"
            self._base_calibration_started_at = None
            self._base_calibration_error = (
                message if cleanup_error is None else f"{message}; cleanup: {cleanup_error}"
            )
            self._robot_state = "FAULT"
            self._error = self._base_calibration_error
            self._message = self._error

    def _poll_base_calibration(self, now: float) -> None:
        with self._lock:
            state = self._base_calibration_state
            if state not in {
                "WAITING_MIDDLE",
                "MIDDLE_CAPTURED",
                "RECORDING",
                "RANGE_INCOMPLETE",
            }:
                return
            cancel_reason = self._base_calibration_cancel_requested
            if cancel_reason is not None:
                self._base_calibration_cancel_requested = None
            started_at = self._base_calibration_started_at
            if (
                cancel_reason is None
                and now - self._base_calibration_last_poll < BASE_CALIBRATION_POLL_INTERVAL_S
            ):
                return
            self._base_calibration_last_poll = now
            session = self._base_calibration
        if cancel_reason is not None:
            self._cancel_base_calibration(cancel_reason)
            return
        if session is None:
            return
        try:
            current = session.read_raw_positions()
        except Exception as exc:
            self._fail_base_calibration(f"Could not read calibration positions: {exc}")
            return
        with self._lock:
            self._base_calibration_current = dict(current)
            if state == "RECORDING":
                for joint in BASE_CALIBRATION_MOTORS:
                    value = int(current[joint])
                    if not _valid_base_calibration_sample(joint, value):
                        continue
                    if joint not in self._base_calibration_mins:
                        self._base_calibration_mins[joint] = value
                        self._base_calibration_maxes[joint] = value
                    else:
                        self._base_calibration_mins[joint] = min(
                            self._base_calibration_mins[joint], value
                        )
                        self._base_calibration_maxes[joint] = max(
                            self._base_calibration_maxes[joint], value
                        )
            recording_complete = bool(
                state == "RECORDING"
                and started_at is not None
                and now - started_at >= BASE_CALIBRATION_TIMEOUT_S
            )
        if recording_complete:
            LOGGER.info("base range 90-second timer completed; validating and saving")
            try:
                self._save_base_calibration(automatic=True)
            except Exception as exc:
                LOGGER.exception("automatic base calibration save failed | error=%s", exc)

    def _connect_robot(
        self, confirmation: str, *, client_id: str | None = None, port: Any = None
    ) -> dict[str, Any]:
        expected = CONTROL_CONFIRMATION_CODE
        if not self.robot_allowed:
            raise ControlError("This server was started in preview-only mode.")
        if confirmation != expected:
            raise ControlError(f"Confirmation must exactly match: {expected}")
        with self._lock:
            if not self._lease_valid_locked(client_id, time.monotonic()):
                raise ControlError("The browser control lease expired before connection.")
            if self._base_calibration is not None or self._base_calibration_state in {
                "STARTING",
                "WAITING_MIDDLE",
                "MIDDLE_CAPTURED",
                "RECORDING",
                "RANGE_INCOMPLETE",
                "SAVING",
                "CANCELLING",
            }:
                raise ControlError("Finish or cancel base calibration before connecting the robot.")
            if self._robot is not None:
                raise ControlError("Robot is already connected or connecting.")
            self._select_follower_port(port)
            self._robot_state = "CONNECTING"
            self._message = f"Connecting {self.config.robot.id} on {self.config.robot.port}"
            self._error = None

        adapter = SO101RobotAdapter(self.config.robot)
        try:
            initial = adapter.connect()
            fingerprint = adapter.calibration_fingerprint()
            controller = MotionController(initial, self.config.mapping, self.config.safety)
            profile = None
            profile_error = None
            pending_fixed = None
            if self.config.safety.skip_safe_range_calibration:
                controlled_limits = {
                    joint: tuple(self.config.safety.absolute_limits[joint])
                    for joint in self.config.mapping.controlled_joints
                }
                fixed_positions = {
                    joint: float(initial[joint])
                    for joint in self.config.mapping.fixed_joints.names
                }
                profile = SafeRangeProfile.create(
                    robot_id=self.config.robot.id,
                    calibration_fingerprint=fingerprint,
                    controlled_limits=controlled_limits,
                    fixed_positions=fixed_positions,
                    startup_positions=initial,
                )
                controller = MotionController(
                    initial,
                    self.config.mapping,
                    self.config.safety,
                    dynamic_limits=profile.controlled_limits,
                    fixed_override=profile.fixed_positions,
                )
                pending_fixed = dict(fixed_positions)
                LOGGER.warning(
                    "safe-range capture skipped in test mode | fixed=%s controlled_limits=%s",
                    fixed_positions,
                    controlled_limits,
                )
            elif self.profile_path is not None:
                try:
                    candidate = load_safe_range_profile(self.profile_path)
                    if candidate is not None:
                        if candidate.robot_id != self.config.robot.id:
                            raise ValueError("Saved safe range belongs to a different robot id.")
                        if candidate.calibration_fingerprint != fingerprint:
                            raise ValueError(
                                "LeRobot calibration changed; collect the safe range again."
                            )
                        controller = MotionController(
                            initial,
                            self.config.mapping,
                            self.config.safety,
                            dynamic_limits=candidate.controlled_limits,
                            fixed_override=candidate.fixed_positions,
                        )
                        profile = candidate
                        pending_fixed = dict(candidate.fixed_positions)
                except Exception as exc:
                    profile_error = str(exc)
        except Exception as exc:
            adapter.disconnect()
            with self._lock:
                self._robot_state = "DISCONNECTED"
                self._error = f"Robot connection failed: {exc}"
                self._message = self._error
            raise

        with self._lock:
            self._robot = adapter
            self._controller = controller
            self._initial = dict(initial)
            self._connection_positions = dict(initial)
            self._last_action = controller.hold()
            self._calibration_fingerprint = fingerprint
            self._safe_profile = profile
            self._safe_profile_error = profile_error
            self._pending_fixed_positions = pending_fixed
            self._range_state = "IDLE"
            self._range_started_at = None
            self._range_cancel_requested = None
            self._range_current = dict(initial)
            self._agent_recovery_positions = None
            self._agent_recovery_joints = ()
            self._agent_return_verified = False
            self._agent_hold_pending = False
            self._range_mins.clear()
            self._range_maxes.clear()
            self._neutral = None
            self._clear_neutral_samples_locked()
            self._collecting_neutral = False
            self._pet_restore_positions = None
            self._agent_motion_state = "IDLE"
            self._agent_motion_message = "No Agent motion is active"
            self._agent_motion_error = None
            self._agent_measured_speeds.clear()
            self._agent_cycle_s = None
            self._agent_feedback_sample = None
            self._robot_state = "CONNECTED_DISARMED"
            self._message = (
                "Robot connected in TEST MODE; Agent gestures are available now; "
                "collect human neutral pose only before visual following"
                if self.config.safety.skip_safe_range_calibration
                else "Robot connected; safe range ready; Agent gestures are available now; "
                "collect human neutral pose only before visual following"
                if profile is not None
                else "Robot connected; confirm ID1-ID2 and collect the ID3-ID6 safe range"
            )
        return {"ok": True, "message": self._message}

    def _clear_neutral_samples_locked(self) -> None:
        for samples in self._neutral_samples.values():
            samples.clear()

    def _neutral_progress_locked(self) -> dict[str, int]:
        return {feature: len(self._neutral_samples[feature]) for feature in CONTROL_FEATURES}

    def _finish_neutral_locked(self, current: PoseFeatures) -> None:
        def channel_median(name: str) -> float:
            values = self._neutral_samples[name]
            if name != "hand_roll":
                return float(statistics.median(values))
            # Circular mean avoids a false midpoint when samples straddle -1/+1.
            sine = statistics.fmean(math.sin(value * math.pi) for value in values)
            cosine = statistics.fmean(math.cos(value * math.pi) for value in values)
            return math.atan2(sine, cosine) / math.pi

        self._neutral = PoseFeatures(
            pan=current.pan,
            lift=channel_median("lift"),
            elbow=channel_median("elbow"),
            center_x=current.center_x,
            hand_roll=channel_median("hand_roll"),
            hand_open=channel_median("hand_open"),
        )
        self._clear_neutral_samples_locked()
        self._collecting_neutral = False
        self._neutral_started_at = None
        self._neutral_last_features = current
        self._neutral_collection_error = None
        self._message = "Neutral channels ready; robot remains DISARMED"
        LOGGER.info("30-second human neutral-pose capture completed")

    def _capture_fixed_positions(self, confirmation: str) -> dict[str, Any]:
        if self.config.safety.skip_safe_range_calibration:
            raise ControlError(
                "Safe-range capture is disabled in test mode; continue to neutral-pose collection."
            )
        expected = CONTROL_CONFIRMATION_CODE
        if confirmation != expected:
            raise ControlError(f"Confirmation must exactly match: {expected}")
        self.disarm("fixed-position confirmation started")
        with self._lock:
            adapter = self._robot
            if adapter is None or not adapter.connected:
                raise ControlError("Robot is not connected.")
            if self._range_state != "IDLE":
                raise ControlError("Finish or cancel safe-range capture first.")
            if self._robot_state != "CONNECTED_DISARMED":
                raise ControlError(f"Robot state does not allow fixed capture: {self._robot_state}")
            self._range_state = "STARTING"
            self._range_cancel_requested = None
            self._safe_profile = None
            self._safe_profile_error = None
            self._neutral = None
            self._neutral_collection_error = None
        try:
            positions = adapter.begin_fixed_positioning(self.config.mapping.fixed_joints.names)
        except Exception:
            with self._lock:
                self._range_state = "IDLE"
            raise
        now = time.monotonic()
        with self._lock:
            self._range_state = "POSITIONING_FIXED"
            self._robot_state = "FIXED_POSITIONING"
            self._fixed_position_started_at = now
            self._range_last_poll = now
            self._range_current = dict(positions)
            self._message = (
                "ID1-ID2 torque released for 10 seconds; move them to the desired fixed pose."
            )
        LOGGER.info(
            "fixed-position stage started | duration_s=10 current=%s",
            {joint: positions[joint] for joint in self.config.mapping.fixed_joints.names},
        )
        return {"ok": True, "message": self._message}

    def _begin_controlled_range(self, adapter: SO101RobotAdapter) -> dict[str, float]:
        with self._lock:
            self._range_state = "STARTING"
            self._range_cancel_requested = None
            self._safe_profile = None
            self._safe_profile_error = None
            self._neutral = None
            self._neutral_collection_error = None
        try:
            positions = adapter.begin_manual_range(self.config.mapping.controlled_joints)
        except Exception:
            with self._lock:
                self._range_state = "IDLE"
                self._robot_state = "CONNECTED_DISARMED"
            raise
        now = time.monotonic()
        with self._lock:
            self._range_state = "RECORDING"
            self._robot_state = "RANGE_RECORDING"
            self._range_started_at = now
            self._range_last_poll = now
            self._range_current = dict(positions)
            self._range_mins = {
                joint: float(positions[joint])
                for joint in self.config.mapping.controlled_joints
            }
            self._range_maxes = dict(self._range_mins)
            self._message = (
                "ID1-ID2 locked. ID3-ID6 torque released for a 60-second safe-range capture."
            )
        LOGGER.info(
            "ID3-ID6 safe-range recording started | duration_s=60 current=%s",
            {joint: positions[joint] for joint in self.config.mapping.controlled_joints},
        )
        return positions

    def _start_safe_range(
        self,
        confirmation: str,
        *,
        client_id: str | None = None,
        submitted_epoch: int | None = None,
    ) -> dict[str, Any]:
        if self.config.safety.skip_safe_range_calibration:
            raise ControlError(
                "Safe-range capture is disabled in test mode; continue to neutral-pose collection."
            )
        expected = CONTROL_CONFIRMATION_CODE
        if confirmation != expected:
            raise ControlError(f"Confirmation must exactly match: {expected}")
        with self._send_lock:
            now = time.monotonic()
            with self._lock:
                if not self._lease_valid_locked(client_id, now):
                    raise ControlError("The browser control lease expired before range capture.")
                if submitted_epoch is None or submitted_epoch != self._epoch:
                    raise ControlError(
                        "Safe-range start was cancelled by a newer STOP/DISARM action."
                    )
            self.disarm("safe-range capture started")
            with self._lock:
                adapter = self._robot
                if adapter is None or not adapter.connected:
                    raise ControlError("Robot is not connected.")
                if self._robot_state != "CONNECTED_DISARMED" or self._range_state != "IDLE":
                    raise ControlError("Robot must be connected and DISARMED before range capture.")
                if self._pending_fixed_positions is None:
                    raise ControlError("Confirm the ID1-ID2 fixed positions first.")
                if self._collecting_neutral:
                    raise ControlError("Neutral-pose collection must finish before range capture.")
            self._begin_controlled_range(adapter)
        return {"ok": True, "message": self._message}

    def _poll_safe_range(self, now: float) -> None:
        with self._lock:
            state = self._range_state
            if state not in {"POSITIONING_FIXED", "RECORDING", "WAITING_RETURN"}:
                return
            cancel_reason = self._range_cancel_requested
            if cancel_reason is not None:
                self._range_cancel_requested = None
            started_at = (
                self._fixed_position_started_at
                if state == "POSITIONING_FIXED"
                else self._range_started_at
            )
            if cancel_reason is None and now - self._range_last_poll < RANGE_POLL_INTERVAL_S:
                return
            self._range_last_poll = now
            adapter = self._robot
        if cancel_reason is not None:
            self._cancel_safe_range(cancel_reason)
            return
        if adapter is None:
            return
        try:
            positions = adapter.read_positions()
        except Exception as exc:
            try:
                self._cancel_safe_range("range position read failed; values were not saved")
            except Exception as restore_exc:
                with self._lock:
                    self._robot_state = "FAULT"
                    self._error = (
                        "Range read and torque restore both failed. Support the arm and use "
                        f"the physical power switch: {exc}; restore: {restore_exc}"
                    )
                    self._message = self._error
            return
        with self._lock:
            self._range_current = dict(positions)
            if state == "RECORDING":
                for joint in self.config.mapping.controlled_joints:
                    value = float(positions[joint])
                    self._range_mins[joint] = min(self._range_mins[joint], value)
                    self._range_maxes[joint] = max(self._range_maxes[joint], value)
            fixed_complete = bool(
                state == "POSITIONING_FIXED"
                and started_at is not None
                and now - started_at >= FIXED_POSITION_TIMEOUT_S
            )
            range_complete = bool(
                state == "RECORDING"
                and started_at is not None
                and now - started_at >= RANGE_CAPTURE_TIMEOUT_S
            )

        if fixed_complete:
            try:
                locked_positions = adapter.finish_fixed_positioning(
                    self.config.mapping.fixed_joints.names
                )
            except Exception as exc:
                with self._lock:
                    self._range_state = "FAULT"
                    self._robot_state = "FAULT"
                    self._error = (
                        "Could not safely lock ID1-ID2 after positioning. Support the arm "
                        "and use the physical power switch: " + str(exc)
                    )
                    self._message = self._error
                LOGGER.exception("ID1-ID2 fixed-position lock failed | error=%s", exc)
                return
            fixed = {
                joint: float(locked_positions[joint])
                for joint in self.config.mapping.fixed_joints.names
            }
            with self._lock:
                self._pending_fixed_positions = fixed
                self._fixed_position_started_at = None
                self._range_current = dict(locked_positions)
                self._message = "ID1-ID2 locked; starting ID3-ID6 range capture"
            LOGGER.info("ID1-ID2 fixed positions locked | positions=%s", fixed)
            try:
                self._begin_controlled_range(adapter)
            except Exception as exc:
                with self._lock:
                    self._range_state = "FAULT"
                    self._robot_state = "FAULT"
                    self._error = f"Could not start ID3-ID6 range capture: {exc}"
                    self._message = self._error
                LOGGER.exception("automatic ID3-ID6 range start failed | error=%s", exc)
            return

        if range_complete:
            with self._lock:
                if self._range_state == "RECORDING":
                    self._range_state = "WAITING_RETURN"
                    self._range_started_at = None
                    self._robot_state = "RANGE_WAITING_RETURN"
                    self._message = (
                        "60-second range captured. Return ID3-ID6 to a comfortable middle "
                        "position, then save and lock."
                    )
            LOGGER.info(
                "ID3-ID6 60-second range complete | mins=%s maxes=%s",
                self._range_mins,
                self._range_maxes,
            )

    def _finish_safe_range(self, confirmation: str) -> dict[str, Any]:
        if self.config.safety.skip_safe_range_calibration:
            raise ControlError(
                "Safe-range capture is disabled in test mode; continue to neutral-pose collection."
            )
        expected = CONTROL_CONFIRMATION_CODE
        if confirmation != expected:
            raise ControlError(f"Confirmation must exactly match: {expected}")
        with self._lock:
            if self._range_state != "WAITING_RETURN":
                raise ControlError(
                    "Wait for the 60-second capture, then return ID3-ID6 to middle before saving."
                )
            adapter = self._robot
            fixed = dict(self._pending_fixed_positions or {})
            mins = dict(self._range_mins)
            maxes = dict(self._range_maxes)
            fingerprint = self._calibration_fingerprint
            self._range_state = "FINISHING"
            self._message = "Saving safe range and locking ID3-ID6 at the current middle pose"
        LOGGER.info(
            "safe-range save started | fixed=%s mins=%s maxes=%s",
            fixed,
            mins,
            maxes,
        )
        if adapter is None or fingerprint is None:
            raise ControlError("Robot calibration information is unavailable.")

        # Read and validate the proposed return pose while ID3-ID6 are still
        # torque-free.  A correctable pose/profile error must not discard the
        # completed 60-second capture or force the operator to repeat ID1-ID2.
        try:
            positions = adapter.read_positions()
        except Exception as exc:
            with self._lock:
                self._range_state = "WAITING_RETURN"
                self._robot_state = "RANGE_WAITING_RETURN"
                self._safe_profile_error = str(exc)
                self._message = (
                    "Safe range is still captured, but the current return pose could not be read. "
                    "Keep supporting the arm and try Save again: " + str(exc)
                )
            raise ControlError(self._message) from exc

        limits: dict[str, tuple[float, float]] = {}
        for joint in self.config.mapping.controlled_joints:
            absolute_low, absolute_high = self.config.safety.absolute_limits[joint]
            limits[joint] = (
                max(float(mins[joint]), absolute_low),
                min(float(maxes[joint]), absolute_high),
            )

        try:
            profile = SafeRangeProfile.create(
                robot_id=self.config.robot.id,
                calibration_fingerprint=fingerprint,
                controlled_limits=limits,
                fixed_positions=fixed,
                startup_positions=positions,
            )
            controller = MotionController(
                positions,
                self.config.mapping,
                self.config.safety,
                dynamic_limits=profile.controlled_limits,
                fixed_override=profile.fixed_positions,
            )
        except Exception as exc:
            with self._lock:
                self._range_current = dict(positions)
                self._range_state = "WAITING_RETURN"
                self._range_started_at = None
                self._robot_state = "RANGE_WAITING_RETURN"
                self._safe_profile_error = str(exc)
                self._message = (
                    f"Safe range is still captured: {exc} "
                    "Adjust ID3-ID6 and click Save again; ID1-ID2 and the 60-second data were kept."
                )
            raise ControlError(self._message) from exc

        try:
            # Use the exact pose that passed validation as the new Goal_Position
            # before restoring torque, so validation and locking cannot disagree.
            positions = adapter.finish_manual_range(
                self.config.mapping.controlled_joints,
                positions=positions,
            )
        except Exception as exc:
            with self._lock:
                self._range_state = "FAULT"
                self._robot_state = "FAULT"
                self._error = (
                    "Could not establish a current-position goal before restoring torque. "
                    "Support the arm and use the physical power switch: " + str(exc)
                )
                self._message = self._error
            raise ControlError(self._error) from exc

        try:
            if self.profile_path is not None:
                save_safe_range_profile(self.profile_path, profile)
        except Exception as exc:
            # The pose itself was valid, but persistence failed after torque was
            # restored.  Release ID3-ID6 again so the operator can safely remain
            # in the return-and-save step and retry without losing the capture.
            try:
                released_positions = adapter.begin_manual_range(
                    self.config.mapping.controlled_joints
                )
            except Exception as release_exc:
                with self._lock:
                    self._range_state = "FAULT"
                    self._robot_state = "FAULT"
                    self._error = (
                        "Safe-range file could not be saved and ID3-ID6 could not be "
                        "returned to torque-free mode. Support the arm and use the physical "
                        f"power switch: save={exc}; release={release_exc}"
                    )
                    self._message = self._error
                raise ControlError(self._error) from release_exc
            with self._lock:
                self._range_current = dict(released_positions)
                self._range_state = "WAITING_RETURN"
                self._range_started_at = None
                self._robot_state = "RANGE_WAITING_RETURN"
                self._safe_profile_error = str(exc)
                self._message = (
                    f"Safe range is still captured but could not be saved: {exc}. "
                    "ID3-ID6 were released again; correct the problem and click Save again."
                )
            raise ControlError(self._message) from exc

        with self._lock:
            self._controller = controller
            self._initial = dict(positions)
            self._last_action = controller.hold()
            self._range_current = dict(positions)
            self._range_state = "IDLE"
            self._range_started_at = None
            self._range_cancel_requested = None
            self._robot_state = "CONNECTED_DISARMED"
            self._safe_profile = profile
            self._safe_profile_error = None
            self._neutral = None
            self._clear_neutral_samples_locked()
            self._collecting_neutral = False
            self._message = (
                "Safe range saved; Agent gestures are available now; "
                "collect human neutral pose only before visual following"
            )
        LOGGER.info("safe range saved and ID3-ID6 locked | current=%s", positions)
        return {"ok": True, "message": self._message}

    def _cancel_safe_range(self, reason: str) -> dict[str, Any]:
        if self.config.safety.skip_safe_range_calibration:
            return {
                "ok": True,
                "message": "Safe-range capture is disabled in test mode",
            }
        LOGGER.warning("safe-range cancellation requested | reason=%s", reason)
        with self._lock:
            if self._range_state not in {
                "POSITIONING_FIXED",
                "RECORDING",
                "STARTING",
                "WAITING_RETURN",
                "FINISHING",
            }:
                return {"ok": True, "message": "Safe-range capture is not active"}
            adapter = self._robot
            self._range_state = "FINISHING"
        if adapter is None:
            raise ControlError("Robot is unavailable while restoring range-capture torque.")
        try:
            if adapter.fixed_positioning_active:
                positions = adapter.finish_fixed_positioning(
                    self.config.mapping.fixed_joints.names
                )
            elif adapter.manual_range_active:
                positions = adapter.finish_manual_range(
                    self.config.mapping.controlled_joints
                )
            else:
                positions = adapter.read_positions()
        except Exception as exc:
            with self._lock:
                self._range_state = "FAULT"
                self._robot_state = "FAULT"
                self._error = (
                    "Could not safely restore torque after cancelling range capture. "
                    "Support the arm and use the physical power switch: " + str(exc)
                )
                self._message = self._error
            raise ControlError(self._error) from exc
        try:
            controller = MotionController(positions, self.config.mapping, self.config.safety)
        except Exception:
            controller = None
        with self._lock:
            if controller is not None:
                self._controller = controller
            self._initial = dict(positions)
            self._last_action = dict(positions)
            self._range_current = dict(positions)
            self._range_state = "IDLE"
            self._range_started_at = None
            self._fixed_position_started_at = None
            self._range_cancel_requested = None
            self._robot_state = "CONNECTED_DISARMED"
            self._safe_profile = None
            self._range_mins.clear()
            self._range_maxes.clear()
            self._neutral = None
            self._clear_neutral_samples_locked()
            self._collecting_neutral = False
            self._message = f"Safe-range capture stopped: {reason}"
        LOGGER.info("safe-range capture cancelled and torque restored | reason=%s", reason)
        return {"ok": True, "message": self._message}

    def _disconnect_robot(
        self, confirmation: str, *, client_id: str | None = None
    ) -> dict[str, Any]:
        expected = CONTROL_CONFIRMATION_CODE
        if confirmation != expected:
            raise ControlError(f"Confirmation must exactly match: {expected}")
        with self._lock:
            range_active = self._range_state in {
                "POSITIONING_FIXED",
                "STARTING",
                "RECORDING",
                "WAITING_RETURN",
                "FINISHING",
            }
        if range_active:
            self._cancel_safe_range("disconnect requested; values were not saved")
        with self._send_lock:
            with self._lock:
                if not self._lease_valid_locked(client_id, time.monotonic()):
                    raise ControlError(
                        "The browser control lease expired before torque release."
                    )
                adapter = self._robot
                if adapter is None:
                    raise ControlError("Robot is not connected.")
                self._robot_state = "DISCONNECTING"
            self.disarm("disconnect requested")
            with self._lock:
                self._message = "Disconnecting and releasing torque"
        try:
            adapter.disconnect()
        except Exception as exc:
            with self._lock:
                self._robot_state = "FAULT"
                self._error = f"Robot disconnect failed: {exc}"
                self._message = self._error
            raise
        preview_controller = MotionController(
            preview_initial_positions(), self.config.mapping, self.config.safety
        )
        with self._lock:
            self._robot = None
            self._controller = preview_controller
            self._initial = dict(preview_controller.initial)
            self._connection_positions = None
            self._last_action = preview_controller.hold()
            self._calibration_fingerprint = None
            self._safe_profile = None
            self._safe_profile_error = None
            self._pending_fixed_positions = None
            self._range_state = "IDLE"
            self._range_started_at = None
            self._range_cancel_requested = None
            self._range_current = None
            self._agent_recovery_positions = None
            self._agent_recovery_joints = ()
            self._agent_return_verified = False
            self._agent_hold_pending = False
            self._range_mins.clear()
            self._range_maxes.clear()
            self._neutral = None
            self._clear_neutral_samples_locked()
            self._collecting_neutral = False
            self._pet_restore_positions = None
            self._agent_motion_state = "IDLE"
            self._agent_motion_message = "No Agent motion is active"
            self._agent_motion_error = None
            self._robot_state = "DISCONNECTED"
            self._error = None
            self._message = "Robot disconnected; torque was released"
        return {"ok": True, "message": self._message}

    def _start_neutral_collection(self) -> dict[str, Any]:
        with self._send_lock, self._lock:
            if self._agent_plan is not None:
                raise ControlError(
                    "Finish or cancel the active Agent motion before neutral-pose collection."
                )
            if self._release_pending:
                raise ControlError(
                    "Torque release is queued or in progress; neutral-pose collection is not allowed."
                )
            if self._preview_state != "RUNNING" or self._latest_features is None:
                raise ControlError("At least one reliable arm or hand channel is required.")
            if self.robot_allowed and self._safe_profile is None:
                raise ControlError("Complete and save the servo safe-range calibration first.")
            if self._range_state != "IDLE":
                raise ControlError("Finish or cancel safe-range capture first.")
            # Validation and capture start must stay ordered with camera disconnect.
            self.disarm("neutral collection started")
            self._neutral = None
            self._clear_neutral_samples_locked()
            self._collecting_neutral = True
            self._neutral_started_at = time.monotonic()
            self._neutral_last_features = self._latest_features
            self._neutral_collection_error = None
            self._message = "Collecting human neutral pose for 30 seconds"
        LOGGER.info("human neutral-pose capture started | duration_s=30")
        return {"ok": True, "message": self._message}

    def _arm(
        self,
        confirmation: str,
        *,
        client_id: str | None = None,
        submitted_epoch: int | None = None,
    ) -> dict[str, Any]:
        expected = CONTROL_CONFIRMATION_CODE
        if confirmation != expected:
            raise ControlError(f"Confirmation must exactly match: {expected}")
        now = time.monotonic()
        with self._lock:
            if (
                not client_id
                or self._operator_client != client_id
                or now - self._operator_heartbeat > self.lease_timeout_s
            ):
                raise ControlError("The browser control lease expired before final ARM.")
            if submitted_epoch is None or submitted_epoch != self._epoch:
                raise ControlError("ARM was cancelled by a newer STOP/DISARM action.")
            if self._agent_plan is not None:
                raise ControlError("Finish or cancel the active Agent motion before ARM.")
            if self._release_pending:
                raise ControlError("Torque release is queued or in progress; ARM is not allowed.")
            if self._robot is None or not self._robot.connected:
                raise ControlError("Robot is not connected.")
            if self._robot_state != "CONNECTED_DISARMED":
                raise ControlError(f"Robot state does not allow ARM: {self._robot_state}")
            if self._safe_profile is None:
                raise ControlError("Servo safe-range calibration is incomplete.")
            if self._collecting_neutral or self._neutral is None:
                raise ControlError("Neutral-pose collection is incomplete.")
            if self._preview_state != "RUNNING":
                raise ControlError("Camera/pose preview is not running for visual following.")
            if self._latest_features is None or self._last_pose_time is None:
                raise ControlError("No reliable pose is available.")
            if now - self._last_pose_time > self.config.safety.tracking_timeout_s:
                raise ControlError("Pose frame is stale.")
            if self._valid_streak < self.config.safety.valid_frames_before_arm:
                raise ControlError("Pose has not been stable for enough consecutive frames.")
            if self._error is not None:
                raise ControlError(f"A runtime fault must be resolved first: {self._error}")
            self._motion_stop.clear()
            self._epoch += 1
            self._armed = True
            self._robot_state = "ARMED"
            self._message = "ARMED: following human pose"
        LOGGER.warning("ARMED | human-following motion enabled")
        return {"ok": True, "message": self._message}

    def _lease_watchdog(self, now: float) -> None:
        expired = False
        with self._lock:
            if (
                self._operator_client is not None
                and now - self._operator_heartbeat > self.lease_timeout_s
            ):
                self._operator_client = None
                self._epoch += 1
                expired = self._armed or self._agent_plan is not None
                # Do not cancel torque-free/manual calibration here. Browser prompt()
                # blocks its own heartbeat timer, and the same local tab will
                # reacquire the lease on its next API request. Powered following is
                # still disarmed immediately when ``expired`` is true.
            expired_camera_owner = self._camera_owner if (
                self._camera_owner is not None and not self._lease_valid_locked(self._camera_owner, now)
            ) else None
        if expired:
            self.disarm("browser heartbeat lost")
        if expired_camera_owner is not None:
            self.disconnect_camera(expired_owner=expired_camera_owner)

    def _update_tracking(
        self, detection: PoseDetection, now: float, *, stop: threading.Event | None = None,
    ) -> PoseDetection:
        features = detection.features
        valid_features = frozenset(set(detection.valid_features) & set(CONTROL_FEATURES))
        proximity = estimate_visual_proximity(
            detection,
            aspect_ratio=self.config.camera.width / self.config.camera.height,
        )
        with self._lock:
            if stop is not None and stop.is_set():
                return detection
            if (
                features is not None
                and "center_x" in detection.valid_features
                and self._last_center_x is not None
            ):
                if abs(features.center_x - self._last_center_x) > self.config.safety.max_person_center_jump:
                    detection = PoseDetection(
                        None,
                        detection.landmarks,
                        detection.people_count,
                        "person changed",
                        detection.hand_landmarks,
                        detection.hand_label,
                        frozenset(),
                    )
                    features = None
                    valid_features = frozenset()
                    self._last_center_x = None

            hard_fault = detection.reason in {"person changed", "multiple people"}
            if features is not None and valid_features:
                if "center_x" in detection.valid_features:
                    self._last_center_x = features.center_x
                self._last_pose_time = now
                for feature in valid_features:
                    self._feature_last_seen[feature] = now
                self._valid_streak += 1
            else:
                self._valid_streak = 0

            self._detection = detection
            self._latest_features = features
            self._latest_valid_features = valid_features
            self._visual_proximity = proximity
            self._visual_proximity_time = now
            if proximity.state in {"SAFE", "SLOW"}:
                self._last_safe_visual_time = now
            powered_motion_active = self._armed
            timed_out = bool(
                self._armed
                and self._last_pose_time is not None
                and now - self._last_pose_time > self.config.safety.tracking_timeout_s
            )
            base_calibration_active = self._base_calibration_state in {
                "STARTING",
                "WAITING_MIDDLE",
                "MIDDLE_CAPTURED",
                "RECORDING",
                "RANGE_INCOMPLETE",
                "SAVING",
            }
            manual_servo_calibration_active = bool(
                base_calibration_active
                or self._range_state
                in {"POSITIONING_FIXED", "STARTING", "RECORDING", "WAITING_RETURN"}
            )

            if hard_fault and self._collecting_neutral:
                self._clear_neutral_samples_locked()
            elif self._collecting_neutral and features is not None:
                self._neutral_last_features = features
                for feature in valid_features:
                    self._neutral_samples[feature].append(float(getattr(features, feature)))

            neutral_time_complete = bool(
                self._collecting_neutral
                and self._neutral_started_at is not None
                and now - self._neutral_started_at >= NEUTRAL_CAPTURE_TIMEOUT_S
            )
            if neutral_time_complete:
                missing = [
                    feature
                    for feature in CONTROL_FEATURES
                    if not self._neutral_samples[feature]
                ]
                if missing or self._neutral_last_features is None:
                    self._collecting_neutral = False
                    self._neutral_started_at = None
                    self._neutral_collection_error = (
                        "Neutral capture lacked valid data for: " + ", ".join(missing)
                    )
                    self._message = self._neutral_collection_error
                    LOGGER.warning("neutral-pose capture incomplete | missing=%s", missing)
                else:
                    self._finish_neutral_locked(self._neutral_last_features)

        # Servo calibration is a manual operation and the camera is mounted on the arm.
        # Moving a joint therefore moves the camera and can transiently look like a
        # person switch or multiple people. Tracking faults must still DISARM normal
        # powered following, but must not abort either manual calibration workflow.
        visual_stop_reason = None
        if proximity.state == "TOO_CLOSE" and powered_motion_active:
            visual_stop_reason = "visual proximity TOO_CLOSE"
        elif hard_fault and not manual_servo_calibration_active:
            visual_stop_reason = detection.reason
        elif timed_out:
            visual_stop_reason = "all control channels timed out"
        if visual_stop_reason is not None:
            # Recheck the mode under the same write gate as STOP: an old camera
            # frame must never cancel a newly started conversation gesture.
            with self._send_lock, self._lock:
                if self._armed and (stop is None or not stop.is_set()):
                    self.disarm(visual_stop_reason)
        return detection

    def _send_if_due(
        self,
        features: PoseFeatures,
        now: float,
        last_control_time: float,
        valid_features: frozenset[str] | None = None,
    ) -> float:
        if now - last_control_time < 1.0 / self.config.robot.control_hz:
            return last_control_time
        if valid_features is None:
            valid_features = frozenset(CONTROL_FEATURES)
        valid_features = frozenset(set(valid_features) & set(CONTROL_FEATURES))
        if not valid_features:
            return last_control_time
        with self._lock:
            if (
                not self._armed
                or self._motion_stop.is_set()
                or self._robot is None
                or self._neutral is None
                or self._valid_streak < self.config.safety.valid_frames_before_arm
            ):
                return last_control_time
            epoch = self._epoch
            robot = self._robot
            controller = self._controller
            neutral = self._neutral
            previous = dict(self._last_action)

        planned = controller.compute(features, neutral, valid_features)
        with self._send_lock:
            with self._lock:
                cancelled = (
                    self._motion_stop.is_set()
                    or not self._armed
                    or epoch != self._epoch
                )
            if cancelled:
                controller.reset_to(previous)
                return last_control_time

            try:
                actual = robot.send(planned)
                if set(actual) != set(ALL_JOINTS):
                    raise RuntimeError("Robot returned an incomplete sent-action record.")
                actual = {joint: float(actual[joint]) for joint in ALL_JOINTS}
                if not all(math.isfinite(value) for value in actual.values()):
                    raise RuntimeError("Robot returned a non-finite sent-action value.")
            except Exception as exc:
                self.disarm("serial command failed")
                with self._lock:
                    self._robot_state = "FAULT"
                    self._error = f"Serial command failed: {exc}"
                    self._message = self._error
                return last_control_time

            controller.reset_to(actual)
            with self._lock:
                self._last_action = actual
                self._last_send_time = now
                if now - self._last_motion_telemetry_time >= 0.5:
                    self._last_motion_telemetry_time = now
                    LOGGER.info(
                        "motion telemetry | hand_open=%.3f valid=%s "
                        "id6_planned=%.2f id6_sent=%.2f",
                        float(features.hand_open),
                        "hand_open" in valid_features,
                        float(planned["gripper.pos"]),
                        float(actual["gripper.pos"]),
                    )
        return now

    def _send_agent_if_due(self, now: float, last_control_time: float) -> float:
        """Advance one semantic Agent keyframe on the sole serial-owner thread."""

        agent_hz = self._agent_timing.control_hz if self._agent_timing else self.config.robot.control_hz
        if now - last_control_time < 1.0 / agent_hz:
            return last_control_time

        cancel_reason: str | None = None
        with self._lock:
            plan = self._agent_plan
            if plan is None:
                return last_control_time
            if not self._lease_valid_locked(self._agent_plan_client, now):
                cancel_reason = "Agent control lease expired"
            elif (
                self._robot is None
                or not self._robot.connected
                or self._robot_state != "CONNECTED_DISARMED"
                or self._armed
            ):
                cancel_reason = "robot state no longer permits Agent motion"
            # Conversation gestures have no camera/recognition prerequisite.
            # Their guards are lease, STOP, encoder feedback and joint limits.

            audio_paused = bool(self._agent_playback and self._agent_playback["state"] == "paused")
            if (cancel_reason is None and audio_paused
                    and now - self._agent_playback["received_at"] > AGENT_PLAYBACK_TIMEOUT_S):
                cancel_reason = "语音暂停后的播放进度已失联，动作已保持停止，请重新发送指令。"
            if (
                cancel_reason is None
                and not audio_paused
                and self._agent_keyframe_started_at is not None
                and self._agent_keyframe_reached_at is None
                and now - self._agent_keyframe_started_at > self._agent_frame_duration_s + AGENT_KEYFRAME_TIMEOUT_S
            ):
                frame = plan.keyframes[self._agent_plan_index]
                observed = self._range_current or {}
                blocked = [
                    f"ID{ALL_JOINTS.index(joint) + 1} {joint.removesuffix('.pos')} "
                    f"目标 {frame.targets[joint]:.2f}、实测 {observed[joint]:.2f}"
                    for joint in frame.active_joints
                    if joint in observed and abs(frame.targets[joint] - observed[joint]) > _agent_feedback_tolerance(joint)
                ]
                detail = f"（{'；'.join(blocked)}）" if blocked else ""
                cancel_reason = (
                    f"第 {self._agent_plan_index + 1}/{len(plan.keyframes)} 步舵机未到位{detail}，"
                    "动作已停止。请检查负载、供电和活动范围。"
                )

            if cancel_reason is None and not audio_paused and self._agent_keyframe_reached_at is not None:
                frame = plan.keyframes[self._agent_plan_index]
                if now - self._agent_keyframe_reached_at < self._agent_frame_hold_s:
                    self._agent_frame_tick_at = now
                    return last_control_time
                LOGGER.info(
                    "agent keyframe completed | plan=%s step=%d/%d label=%s",
                    plan.name,
                    self._agent_plan_index + 1,
                    len(plan.keyframes),
                    frame.label,
                )
                local_index = getattr(frame, "action_index", 0)
                source_index = self._agent_source_indices[local_index] if local_index < len(self._agent_source_indices) else 0
                self._agent_completed_cycles[source_index] = max(
                    self._agent_completed_cycles.get(source_index, 0), getattr(frame, "completed_cycles", 0),
                )
                self._agent_plan_index += 1
                self._agent_keyframe_reached_at = None
                self._agent_keyframe_started_at = None
                self._agent_frame_origin = None
                self._agent_frame_joint_schedule = {}
                self._agent_frame_tick_at = None
                self._agent_feedback_sample = None
                self._trajectory_error_since.clear()
                if self._agent_plan_index >= len(plan.keyframes):
                    self._complete_agent_plan_locked()
                    return last_control_time

            if cancel_reason is None and not audio_paused:
                frame = plan.keyframes[self._agent_plan_index]
                robot = self._robot
                controller = self._controller
                epoch = self._agent_plan_epoch
                previous = dict(self._last_action)
                active_joints = tuple(frame.active_joints)
                fixed = {joint: self._agent_hold_positions.get(joint, previous[joint]) for joint in ALL_JOINTS if joint not in active_joints}
                if not set(active_joints) <= AGENT_JOINTS:
                    raise RuntimeError("Agent frame requested an unsupported joint.")
                remaining_speech = self._agent_speech_remaining_locked(now)
                if self._agent_frame_origin is None:
                    # Plans and resumes seed this goal from measured position.
                    # Later frames start at the last goal to avoid a command jump.
                    self._agent_frame_origin = previous
                    remaining_frames = plan.keyframes[self._agent_plan_index:]
                    if self._agent_audio_paused and self._agent_frame_joint_schedule:
                        frame = replace(frame, joint_timing=self._remaining_agent_joint_timing_locked(frame, now))
                        remaining_frames = (frame, *remaining_frames[1:])
                    timing = calculate_agent_motion_timing(
                        AgentMotionPlan(plan.name, remaining_frames),
                        self._agent_frame_origin,
                        max_step_per_cycle=self.config.safety.max_step_per_cycle,
                        max_relative_target=self.config.robot.max_relative_target,
                        base_control_hz=self.config.robot.control_hz,
                        speech_duration_s=remaining_speech,
                        measured_speeds=self._agent_measured_speeds,
                        cycle_time_s=self._agent_cycle_s,
                    )
                    self._agent_timing = timing
                    # Command pace is independent of a long spoken reply. The
                    # feedback-based ETA remains diagnostic, never a slowdown.
                    period_s = 1.0 / timing.control_hz
                    if frame.trajectory is None:
                        duration, self._agent_frame_hold_s, self._agent_frame_joint_schedule = agent_frame_schedule(
                            frame, previous, timing.max_speeds, period_s,
                        )
                    else:
                        duration = frame.trajectory.duration_s
                        self._agent_frame_hold_s = frame.hold_s
                        self._agent_frame_joint_schedule = {}
                    self._agent_frame_duration_s = max(period_s, duration)
                    self._agent_frame_elapsed_s = 0.0
                    self._agent_frame_tick_at = now
                    self._agent_keyframe_started_at = now
                    self._agent_audio_paused = False
                    LOGGER.info(
                        "agent frame timeline | motion=%s step=%d duration_s=%.3f speech_remaining_s=%s effective_speeds=%s reason=%s",
                        epoch, self._agent_plan_index + 1, self._agent_frame_duration_s,
                        remaining_speech, timing.effective_speeds, timing.reason,
                    )
                timing = self._agent_timing
                # A slow serial cycle must not skip ahead into the peak of the ramp.
                period_s = 1.0 / timing.control_hz
                tick_elapsed = max(0.0, now - self._agent_frame_tick_at)
                delta = min(period_s, tick_elapsed)
                if frame.trajectory is not None:
                    # Preserve normal serial jitter without skipping after a long stall.
                    replay_delta = min(2.0 * period_s, tick_elapsed)
                    replay_target = frame.trajectory.position_at(self._agent_frame_elapsed_s + replay_delta)
                    if all(
                        abs(replay_target[joint] - previous[joint]) <= timing.max_speeds[joint] * period_s
                        for joint in active_joints
                    ):
                        delta = replay_delta
                self._agent_frame_tick_at = now
                self._agent_frame_elapsed_s = min(self._agent_frame_duration_s, self._agent_frame_elapsed_s + delta)
                phase = self._agent_frame_elapsed_s / self._agent_frame_duration_s
                if frame.trajectory is not None:
                    interpolated_target = frame.trajectory.position_at(
                        frame.trajectory.duration_s if phase >= 1.0 else self._agent_frame_elapsed_s
                    )
                else:
                    progress = cosine_progress(phase)
                    joint_progress = {
                        joint: cosine_progress(min(1.0, self._agent_frame_elapsed_s / duration))
                        for joint, (duration, _) in self._agent_frame_joint_schedule.items()
                    }
                    interpolated_target = dict(frame.targets)
                    interpolated_target.update({
                        joint: self._agent_frame_origin[joint] + joint_progress.get(joint, progress) * (
                            frame.targets[joint] - self._agent_frame_origin[joint]
                        )
                        for joint in active_joints
                    })
                self._agent_motion_state = (
                    "RETURNING" if "restore" in frame.label.lower() or "return" in frame.label.lower()
                    else "RUNNING"
                )
                self._agent_motion_message = (
                    f"Agent motion {self._agent_plan_index + 1}/{len(plan.keyframes)}: "
                    f"{frame.label}"
                )

        if cancel_reason is not None:
            LOGGER.warning("agent safety interception | reason=%s", cancel_reason)
            self.disarm(cancel_reason)
            return last_control_time

        if audio_paused:
            try:
                return self._pause_agent_for_audio(now, plan)
            except Exception:
                LOGGER.exception("agent audio pause hold failed")
                self.disarm("Agent audio pause hold failed")
                return last_control_time

        try:
            effective_target = dict(previous)
            effective_target.update({joint: interpolated_target[joint] for joint in active_joints})
            planned, reached = step_toward_agent_target(
                previous,
                effective_target,
                fixed_positions=fixed,
                # The time profile already applies the pace; do not scale the
                # command cap twice.  Original configured caps still apply.
                max_step={joint: speed / timing.control_hz for joint, speed in timing.max_speeds.items()},
                speed_scale=1.0,
            )
        except Exception as exc:
            self.disarm("Agent motion planning failed")
            with self._lock:
                self._agent_motion_state = "FAULT"
                self._agent_motion_error = str(exc)
                self._agent_motion_message = f"Agent motion planning failed: {exc}"
                self._message = self._agent_motion_message
            return last_control_time

        # send_action() returns the goal actually SENT, not the measured pose.
        # Even when the last goal matches, read the encoders before advancing.
        needs_write = not all(
            abs(planned[joint] - previous[joint]) <= 1e-6 for joint in active_joints
        )

        with self._send_lock:
            with self._lock:
                cancelled = bool(
                    self._agent_plan is not plan
                    or self._agent_plan_epoch != epoch
                    or self._robot is not robot
                    or self._robot_state != "CONNECTED_DISARMED"
                    or self._armed
                    or self._motion_stop.is_set()
                )
            if cancelled:
                return last_control_time

            io_started_at = time.monotonic()
            try:
                if needs_write and hasattr(robot, "send_agent_targets"):
                    actual, observed = robot.send_agent_targets(planned, active_joints, fixed)
                elif needs_write and hasattr(robot, "send_observed"):
                    actual, observed = robot.send_observed(planned)
                else:
                    observed = robot.read_positions()
                    actual = robot.send(planned) if needs_write else previous
                io_finished_at = time.monotonic()
                if set(actual) != set(ALL_JOINTS):
                    raise RuntimeError("Robot returned an incomplete Agent action record.")
                actual = {joint: float(actual[joint]) for joint in ALL_JOINTS}
                if not all(math.isfinite(value) for value in actual.values()):
                    raise RuntimeError("Robot returned a non-finite Agent action value.")
                if set(observed) != set(ALL_JOINTS):
                    raise RuntimeError("Robot returned incomplete measured Agent positions.")
                observed = {joint: float(observed[joint]) for joint in ALL_JOINTS}
                if not all(math.isfinite(value) for value in observed.values()):
                    raise RuntimeError("Robot returned non-finite measured Agent positions.")
                position_error = max(
                    (abs(frame.targets[joint] - observed[joint]) for joint in active_joints),
                    default=0.0,
                )
                reached = bool(
                    phase >= 1.0
                    and
                    all(
                        abs(frame.targets[joint] - observed[joint]) <= _agent_feedback_tolerance(joint)
                        for joint in active_joints
                    )
                    and all(
                        abs(frame.targets[joint] - actual[joint]) <= AGENT_POSITION_TOLERANCE
                        for joint in active_joints
                    )
                )
                if frame.trajectory is not None:
                    blocked = []
                    for joint in active_joints:
                        tolerance = _agent_feedback_tolerance(joint) + timing.steps[joint]
                        if abs(previous[joint] - observed[joint]) <= tolerance:
                            self._trajectory_error_since.pop(joint, None)
                        elif now - self._trajectory_error_since.setdefault(joint, now) >= TRAJECTORY_TRACKING_TIMEOUT_S:
                            blocked.append(
                                f"ID{ALL_JOINTS.index(joint) + 1} {joint} "
                                f"指令 {previous[joint]:.2f}、实测 {observed[joint]:.2f}"
                            )
                    if blocked:
                        cancel_reason = f"从臂轨迹跟踪持续偏差，已停止（{'，'.join(blocked)}）。"
            except Exception as exc:
                LOGGER.exception("agent serial tick failed | motion=%s step=%d io_ms=%.1f",
                                 epoch, self._agent_plan_index + 1,
                                 (time.monotonic() - io_started_at) * 1000)
                self.disarm("Agent serial command failed")
                with self._lock:
                    self._robot_state = "FAULT"
                    self._error = f"Agent serial command failed: {exc}"
                    self._agent_motion_state = "FAULT"
                    self._agent_motion_error = str(exc)
                    self._agent_motion_message = self._error
                    self._message = self._error
                return last_control_time

            if cancel_reason is not None:
                LOGGER.warning("follower trajectory tracking stopped | motion=%s reason=%s", epoch, cancel_reason)
                self.disarm(cancel_reason)
                return last_control_time

            controller.reset_to(actual)
            with self._lock:
                if self._agent_plan is plan and self._agent_plan_epoch == epoch:
                    self._last_action = actual
                    self._agent_hold_positions.update({joint: actual[joint] for joint in active_joints})
                    self._range_current = observed
                    self._last_send_time = now
                    self._sample_agent_feedback_locked(
                        observed, actual, active_joints, io_started_at, timing,
                    )
                    if reached:
                        self._agent_keyframe_reached_at = now
                        LOGGER.info(
                            "agent keyframe measured arrival | plan=%s step=%d/%d label=%s max_error=%.2f measured=%s",
                            plan.name,
                            self._agent_plan_index + 1,
                            len(plan.keyframes),
                            frame.label,
                            position_error,
                            observed,
                        )
                    self._log_motion_telemetry(
                        "follower_replay" if frame.trajectory is not None else "agent",
                        started_at=self._agent_plan_started_at or now, tick_at=io_started_at,
                        io_elapsed_s=io_finished_at - io_started_at, control_hz=timing.control_hz,
                        target=effective_target, planned=planned, previous=previous, sent=actual,
                        measured=observed, active_joints=active_joints, wrote=needs_write,
                        details={"motion": epoch, "step": f"{self._agent_plan_index + 1}/{len(plan.keyframes)}",
                                 "label": frame.label, "end_target": frame.targets,
                                 "end_error": round(position_error, 2), "remaining_speech_s": remaining_speech,
                                 "estimated_s": round(timing.estimated_duration_s, 2),
                                 "frame_elapsed_s": round(self._agent_frame_elapsed_s, 3),
                                 "frame_duration_s": round(self._agent_frame_duration_s, 3)},
                    )
        return now

    def _publish_frame(self, frame, detection: PoseDetection, now: float, stop: threading.Event) -> None:
        import cv2

        with self._lock:
            status_lines = [
                f"WEB | {self._robot_state} | {'ARMED' if self._armed else 'DISARMED'}",
                (
                    f"tracking: {detection.reason} | active: "
                    f"{','.join(sorted(set(detection.valid_features) & set(CONTROL_FEATURES))) or 'none'}"
                ),
                (
                    "neutral: "
                    + ",".join(
                        f"{name}={len(self._neutral_samples[name])}/{self.config.pose.neutral_sample_count}"
                        for name in CONTROL_FEATURES
                    )
                    if self._collecting_neutral
                    else f"neutral: {'ready' if self._neutral is not None else 'not ready'}"
                ),
                (
                    f"proximity: {self._visual_proximity.state} | agent: "
                    f"{self._agent_motion_state}"
                ),
                "ID1-2 FIXED | ID3-6 INDEPENDENT | physical power is the real E-stop",
            ]
        annotated = annotate_frame(frame, detection, status_lines)
        ok, encoded = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            raise RuntimeError("JPEG encoding failed.")
        with self._frame_condition:
            if stop.is_set():
                return
            self._latest_jpeg = encoded.tobytes()
            self._frame_revision += 1
            self._last_frame_time = now
            self._preview_state = "RUNNING"
            self._frame_condition.notify_all()

    def _vision_loop(self, name: str, stop: threading.Event) -> None:
        """Own the capture and release it on the same thread that reads frames."""
        camera = None
        detector = None
        error = None
        try:
            camera = self._open_camera(name)
            if stop.is_set():
                return
            detector = MediaPipePoseDetector(self.config.pose)
            while not stop.is_set():
                ok, frame = camera.read()
                if stop.is_set():
                    break
                if not ok:
                    raise RuntimeError("相机没有返回画面，请检查设备连接后重试。")
                frame = orient_camera_frame(
                    frame,
                    rotation_degrees=self.config.camera.rotation_degrees,
                    flip_horizontal=self.config.camera.flip_horizontal,
                    flip_vertical=self.config.camera.flip_vertical,
                )
                detection = detector.detect(frame)
                now = time.monotonic()
                detection = self._update_tracking(detection, now, stop=stop)
                if not stop.is_set():
                    self._publish_frame(frame, detection, now, stop)
        except Exception as exc:
            if not stop.is_set():
                error = str(exc)
                LOGGER.warning("optional vision worker failed | error=%s", exc)
        finally:
            with self._send_lock, self._frame_condition:
                self._preview_state = "STOPPING"
                if self._armed:
                    self.disarm("视频已停止，人体跟随已解除。")
                self._clear_camera_tracking_locked("camera stopped")
            if camera is not None:
                try:
                    camera.release()
                except Exception:
                    error = "释放相机失败，请检查设备或重新启动本地服务。"
                    LOGGER.exception("camera release failed")
            if detector is not None:
                try:
                    detector.close()
                except Exception:
                    LOGGER.exception("pose detector cleanup failed")
            with self._frame_condition:
                self._vision_thread = None
                self._vision_stop = None
                self._camera_owner = None
                self._preview_error = error
                self._preview_state = "FAULT" if error else "STOPPED"
                if not error:
                    self._camera_name = None
                self._frame_condition.notify_all()

    def _run(self) -> None:
        last_control_time = 0.0
        try:
            with self._lock:
                self._message = "控制服务已就绪；对话动作不依赖摄像头。"
                self._error = None
            while not self._shutdown.is_set():
                self._process_commands()
                if self._shutdown.is_set():
                    break
                now = time.monotonic()
                self._lease_watchdog(now)
                self._poll_base_calibration(now)
                self._poll_safe_range(now)
                self._hold_agent_if_pending()
                self._poll_teach(now)
                with self._lock:
                    features = self._latest_features
                    valid_features = self._latest_valid_features
                    human_active = self._armed
                    camera_fresh = bool(
                        self._preview_state == "RUNNING"
                        and self._last_frame_time is not None
                        and now - self._last_frame_time <= self.config.safety.tracking_timeout_s
                    )
                if human_active and not camera_fresh:
                    self.disarm("visual preview is stale")
                elif human_active and features is not None:
                    last_control_time = self._send_if_due(
                        features,
                        now,
                        last_control_time,
                        valid_features=valid_features,
                    )
                last_control_time = self._send_agent_if_due(now, last_control_time)
                self._shutdown.wait(0.005)
        except Exception as exc:
            with self._lock:
                base_active = self._base_calibration is not None
            if base_active:
                try:
                    self._cancel_base_calibration("runtime fault")
                except Exception as cleanup_exc:
                    exc = RuntimeError(
                        f"{exc}; base-calibration cleanup failed: {cleanup_exc}"
                    )
            with self._lock:
                range_active = self._range_state in {
                    "POSITIONING_FIXED",
                    "STARTING",
                    "RECORDING",
                    "WAITING_RETURN",
                    "FINISHING",
                }
            if range_active:
                try:
                    self._cancel_safe_range("runtime fault; range values were not saved")
                except Exception as restore_exc:
                    exc = RuntimeError(
                        f"{exc}; safe-range torque restore failed: {restore_exc}. "
                        "Support the arm and use the physical power switch."
                    )
            self.disarm("runtime fault")
            self.disconnect_camera()
            with self._lock:
                self._error = str(exc)
                self._message = f"Runtime fault: {exc}"
            while not self._shutdown.is_set():
                self._process_commands()
                now = time.monotonic()
                self._lease_watchdog(now)
                self._poll_teach(now)
                time.sleep(0.05)
        finally:
            self.disarm("control thread stopping")
            try:
                if self._teach.recording:
                    self._stop_teach_recording()
                elif self._teach.active:
                    self._teach.cancel()
            except Exception:
                LOGGER.exception("teach shutdown cleanup failed")
            with self._lock:
                adapter = self._robot
                calibration_session = self._base_calibration
            if calibration_session is not None:
                try:
                    calibration_session.cancel()
                except Exception as exc:
                    print(f"Base calibration shutdown warning: {exc}")
            if adapter is not None:
                if adapter.manual_range_active:
                    try:
                        adapter.finish_manual_range(self.config.mapping.controlled_joints)
                    except Exception as exc:
                        print(f"Safe-range shutdown restore warning: {exc}")
                try:
                    adapter.disconnect()
                except Exception as exc:
                    print(f"Robot shutdown warning: {exc}")
            with self._lock:
                self._robot = None
                self._connection_positions = None
                self._robot_state = "DISABLED" if not self.robot_allowed else "DISCONNECTED"
                self._message = "Control thread stopped"
            with self._frame_condition:
                self._frame_condition.notify_all()

