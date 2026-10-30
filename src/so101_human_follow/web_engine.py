from __future__ import annotations

import os
import math
import logging
import queue
import statistics
import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import ALL_JOINTS, CONTROLLED_JOINTS, FIXED_JOINTS, AppConfig
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
    """Single-owner runtime for camera, pose model, controller and serial bus."""

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
        self._epoch = 0

        self._preview_state = "STOPPED"
        self._robot_state = "DISABLED" if not robot_allowed else "DISCONNECTED"
        self._armed = False
        self._message = "Waiting to start"
        self._error: str | None = None

        self._operator_client: str | None = None
        self._operator_heartbeat = 0.0

        self._robot: SO101RobotAdapter | None = None
        self._release_pending = False
        self._controller = MotionController(
            preview_initial_positions(), config.mapping, config.safety
        )
        self._initial = dict(self._controller.initial)
        self._last_action = self._controller.hold()
        self._last_send_time: float | None = None
        self._last_motion_telemetry_time = 0.0

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
            self._preview_state = "STARTING"
            self._message = "Starting camera and pose model"
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
                if expired and self._armed:
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
                self._armed = False
                self._epoch += 1
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
        LOGGER.info("DISARM | reason=%s | was_armed=%s", reason, was_armed)

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
            can_arm = bool(
                operator
                and self._robot is not None
                and self._robot.connected
                and self._robot_state == "CONNECTED_DISARMED"
                and profile_ready
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
                "robot_allowed": self.robot_allowed,
                "robot_state": self._robot_state,
                "robot_connected": bool(self._robot is not None and self._robot.connected),
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
                    and self._preview_state == "RUNNING"
                    and self._robot is None
                    and self._robot_state in {"DISCONNECTED", "FAULT"}
                    and not base_active
                ),
                "can_start_base_calibration": bool(
                    operator
                    and self.robot_allowed
                    and self._preview_state == "RUNNING"
                    and self._robot is None
                    and not base_active
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

    def wait_for_frame(self, after_revision: int, timeout_s: float = 2.0) -> tuple[int, bytes | None]:
        with self._frame_condition:
            if self._frame_revision <= after_revision and not self._shutdown.is_set():
                self._frame_condition.wait(timeout_s)
            return self._frame_revision, self._latest_jpeg

    def _open_camera(self):
        import cv2

        backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
        camera = cv2.VideoCapture(self.config.camera.index, backend)
        if not camera.isOpened():
            camera.release()
            raise RuntimeError(f"Could not open camera index {self.config.camera.index}.")
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.camera.width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.camera.height)
        camera.set(cv2.CAP_PROP_FPS, self.config.camera.fps)
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
            "start_safe_range",
            "start_base_calibration",
            "start_base_range",
        } and command.submitted_epoch != current_epoch:
            action = {
                "arm": "ARM",
                "start_safe_range": "Safe-range start",
                "start_base_calibration": "Base-calibration start",
                "start_base_range": "Base-range start",
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
        if name == "connect":
            return self._connect_robot(
                str(payload.get("confirmation", "")), client_id=command.client_id
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
        raise ControlError(f"Unknown control command: {name}")

    def _start_base_calibration(
        self,
        confirmation: str,
        *,
        client_id: str | None = None,
        submitted_epoch: int | None = None,
    ) -> dict[str, Any]:
        expected = CONTROL_CONFIRMATION_CODE
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
                self._base_calibration_state = "STARTING"
                self._base_calibration_cancel_requested = None
                self._base_calibration_error = None
                self._robot_state = "CALIBRATING"
                self._message = "Opening COM4 for base calibration"

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
        self, confirmation: str, *, client_id: str | None = None
    ) -> dict[str, Any]:
        expected = CONTROL_CONFIRMATION_CODE
        if not self.robot_allowed:
            raise ControlError("This server was started in preview-only mode.")
        if confirmation != expected:
            raise ControlError(f"Confirmation must exactly match: {expected}")
        with self._lock:
            if not self._lease_valid_locked(client_id, time.monotonic()):
                raise ControlError("The browser control lease expired before connection.")
            if self._preview_state != "RUNNING":
                raise ControlError("Camera/pose preview is not running.")
            if self._robot is not None:
                raise ControlError("Robot is already connected or connecting.")
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
            self._last_action = controller.hold()
            self._calibration_fingerprint = fingerprint
            self._safe_profile = profile
            self._safe_profile_error = profile_error
            self._pending_fixed_positions = pending_fixed
            self._range_state = "IDLE"
            self._range_started_at = None
            self._range_cancel_requested = None
            self._range_current = dict(initial)
            self._range_mins.clear()
            self._range_maxes.clear()
            self._neutral = None
            self._clear_neutral_samples_locked()
            self._collecting_neutral = False
            self._robot_state = "CONNECTED_DISARMED"
            self._message = (
                "Robot connected in TEST MODE; safe-range capture skipped, collect neutral pose"
                if self.config.safety.skip_safe_range_calibration
                else "Robot connected; safe range ready, collect neutral pose"
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
            self._message = "Safe range saved; collect the human neutral pose next"
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
            self._last_action = preview_controller.hold()
            self._calibration_fingerprint = None
            self._safe_profile = None
            self._safe_profile_error = None
            self._pending_fixed_positions = None
            self._range_state = "IDLE"
            self._range_started_at = None
            self._range_cancel_requested = None
            self._range_current = None
            self._range_mins.clear()
            self._range_maxes.clear()
            self._neutral = None
            self._clear_neutral_samples_locked()
            self._collecting_neutral = False
            self._robot_state = "DISCONNECTED"
            self._error = None
            self._message = "Robot disconnected; torque was released"
        return {"ok": True, "message": self._message}

    def _start_neutral_collection(self) -> dict[str, Any]:
        self.disarm("neutral collection started")
        with self._lock:
            if self._preview_state != "RUNNING" or self._latest_features is None:
                raise ControlError("At least one reliable arm or hand channel is required.")
            if self.robot_allowed and self._safe_profile is None:
                raise ControlError("Complete and save the servo safe-range calibration first.")
            if self._range_state != "IDLE":
                raise ControlError("Finish or cancel safe-range capture first.")
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
            if self._robot is None or not self._robot.connected:
                raise ControlError("Robot is not connected.")
            if self._robot_state != "CONNECTED_DISARMED":
                raise ControlError(f"Robot state does not allow ARM: {self._robot_state}")
            if self._safe_profile is None:
                raise ControlError("Servo safe-range calibration is incomplete.")
            if self._collecting_neutral or self._neutral is None:
                raise ControlError("Neutral-pose collection is incomplete.")
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
                expired = self._armed
                # Do not cancel torque-free/manual calibration here. Browser prompt()
                # blocks its own heartbeat timer, and the same local tab will
                # reacquire the lease on its next API request. Powered following is
                # still disarmed immediately when ``expired`` is true.
        if expired:
            self.disarm("browser heartbeat lost")

    def _update_tracking(self, detection: PoseDetection, now: float) -> PoseDetection:
        features = detection.features
        valid_features = frozenset(set(detection.valid_features) & set(CONTROL_FEATURES))
        with self._lock:
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
        if hard_fault and not manual_servo_calibration_active:
            self.disarm(detection.reason)
        elif timed_out:
            self.disarm("all control channels timed out")
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

    def _publish_frame(self, frame, detection: PoseDetection, now: float) -> None:
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
                "ID1-2 FIXED | ID3-6 INDEPENDENT | physical power is the real E-stop",
            ]
        annotated = annotate_frame(frame, detection, status_lines)
        ok, encoded = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            raise RuntimeError("JPEG encoding failed.")
        with self._frame_condition:
            self._latest_jpeg = encoded.tobytes()
            self._frame_revision += 1
            self._last_frame_time = now
            self._frame_condition.notify_all()

    def _run(self) -> None:
        camera = None
        detector = None
        last_control_time = 0.0
        try:
            detector = MediaPipePoseDetector(self.config.pose)
            camera = self._open_camera()
            with self._lock:
                self._preview_state = "RUNNING"
                self._message = (
                    "Preview running; robot control is disabled"
                    if not self.robot_allowed
                    else "Preview running; robot is not connected"
                )
                self._error = None

            while not self._shutdown.is_set():
                self._process_commands()
                if self._shutdown.is_set():
                    break
                now = time.monotonic()
                self._lease_watchdog(now)
                self._poll_base_calibration(now)
                self._poll_safe_range(now)
                ok, frame = camera.read()
                if not ok:
                    raise RuntimeError("Camera read failed.")
                frame = orient_camera_frame(
                    frame,
                    rotation_degrees=self.config.camera.rotation_degrees,
                    flip_horizontal=self.config.camera.flip_horizontal,
                    flip_vertical=self.config.camera.flip_vertical,
                )
                detection = detector.detect(frame)
                # Freshness must be measured after both inference passes finish.
                now = time.monotonic()
                detection = self._update_tracking(detection, now)
                if detection.features is not None:
                    last_control_time = self._send_if_due(
                        detection.features,
                        now,
                        last_control_time,
                        valid_features=detection.valid_features,
                    )
                self._publish_frame(frame, detection, now)
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
            with self._lock:
                self._preview_state = "FAULT"
                self._error = str(exc)
                self._message = f"Runtime fault: {exc}"
            while not self._shutdown.is_set():
                self._process_commands()
                self._lease_watchdog(time.monotonic())
                time.sleep(0.05)
        finally:
            self.disarm("control thread stopping")
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
            if detector is not None:
                try:
                    detector.close()
                except Exception as exc:
                    print(f"Pose detector shutdown warning: {exc}")
            if camera is not None:
                try:
                    camera.release()
                except Exception as exc:
                    print(f"Camera shutdown warning: {exc}")
            with self._lock:
                self._robot = None
                self._preview_state = "STOPPED"
                self._robot_state = "DISABLED" if not self.robot_allowed else "DISCONNECTED"
                self._message = "Control thread stopped"
            with self._frame_condition:
                self._frame_condition.notify_all()

