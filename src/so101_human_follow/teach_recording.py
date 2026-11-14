"""SO-101 leader calibration and recording on the control thread."""

from __future__ import annotations

import logging
import math
import time
import uuid
from collections.abc import Mapping
from typing import Any

from .config import ALL_JOINTS, RobotSettings
from .robot_adapter import SO101CalibrationSession, _make_lerobot_importable
from .teach_trajectory import MAX_RECORDING_SAMPLES, MAX_RECORDING_SECONDS


LOGGER = logging.getLogger("so101_human_follow.teach")
LEADER_ID = "my_awesome_leader_arm"
GRIPPER = "gripper.pos"
CALIBRATION_INTERVAL_S = 0.15
CALIBRATION_DURATION_S = 90.0
MIN_CALIBRATION_SPAN = 20
RECORD_INTERVAL_S = 1 / 30


def _make_leader(port: str) -> Any:
    _make_lerobot_importable()
    from lerobot.teleoperators.so101_leader import SO101Leader, SO101LeaderConfig

    return SO101Leader(SO101LeaderConfig(port=port, id=LEADER_ID, use_degrees=False))


def _raw_by_joint(raw: Mapping[str, int]) -> dict[str, int]:
    return {joint: raw[joint.removesuffix(".pos")] for joint in ALL_JOINTS}


def _extend_ranges(
    ranges: dict[str, tuple[float, float]], positions: Mapping[str, float]
) -> dict[str, tuple[float, float]]:
    result = dict(ranges)
    for joint, position in positions.items():
        low, high = result.get(joint, (position, position))
        result[joint] = min(low, position), max(high, position)
    return result


def _valid_raw(value: int) -> bool:
    return 0 <= value < (1 << 15)


class TeachRecorder:
    """Own leader serial I/O, while the engine alone can command the follower."""

    def __init__(self, robot_settings: RobotSettings):
        self._robot_settings = robot_settings
        self._calibration: SO101CalibrationSession | None = None
        self._leader: Any | None = None
        self.port: str | None = None
        self._state = "IDLE"
        self._message = "填写主臂串口后可以校准或录制。"
        self._error: str | None = None
        self._sync = False
        self._directions = dict.fromkeys(ALL_JOINTS, 1)
        self._started_at = 0.0
        self._last_poll = 0.0
        self._calibration_port: str | None = None
        self._calibration_positions: dict[str, int] = {}
        self._calibration_ranges: dict[str, tuple[float, float]] = {}
        self._previous_calibration: dict[str, Any] = {}
        self._samples: list[dict[str, Any]] = []
        self._follower_samples: list[dict[str, Any]] = []
        self._ranges: dict[str, tuple[float, float]] = {}
        self._recording_metadata: dict[str, str] = {}
        self._result: dict[str, Any] | None = None

    @property
    def active(self) -> bool:
        return self._calibration is not None or self.recording

    @property
    def recording(self) -> bool:
        return self._leader is not None

    @property
    def sync_active(self) -> bool:
        return self.recording and self._sync

    def status(self) -> dict[str, Any]:
        samples = self._samples
        latest = samples[-1] if samples else None
        state = self._state
        return {
            "state": state,
            "active": self.active,
            "recording": self.recording,
            "sync_active": self.sync_active,
            "port": self.port,
            "leader_id": LEADER_ID,
            "message": self._message,
            "error": self._error,
            "duration_s": latest["time_s"] if latest else 0.0,
            "sample_count": len(samples),
            "recording_id": self._recording_metadata.get("id"),
            "positions": dict(latest["positions"]) if latest else {},
            "ranges": {joint: list(pair) for joint, pair in self._ranges.items()},
            "calibration_port": self._calibration_port,
            "calibration_positions": dict(self._calibration_positions),
            "calibration_ranges": {joint: list(pair) for joint, pair in self._calibration_ranges.items()},
            "calibration_remaining_s": (
                max(0.0, CALIBRATION_DURATION_S - (time.monotonic() - self._started_at))
                if state == "CALIBRATING_RANGE" else None
            ),
        }

    def start_calibration(self, port: str) -> dict[str, Any]:
        if self.active:
            raise RuntimeError("请先结束当前主臂校准或录制。")
        # The follower's single-turn wrist guard protects its camera cable.
        # The unpowered leader uses encoder values solely as input.
        session = SO101CalibrationSession(
            self._robot_settings,
            device_factory=lambda: _make_leader(port),
            single_turn_wrist=False,
        )
        try:
            current = _raw_by_joint(session.start())
        except Exception as exc:
            if session.active:
                session.cancel()
            self._state = "FAULT"
            self._error = str(exc)
            self._message = "主臂校准未能开始。"
            raise
        self._calibration = session
        self._calibration_port = port
        self._calibration_positions = current
        self._calibration_ranges = {}
        self._previous_calibration = {}
        self.port = port
        self._state = "WAITING_MIDDLE"
        self._error = None
        self._message = "主臂已释放扭矩。将六轴放在各自行程中间，再确认中位。"
        LOGGER.info("leader calibration started | port=%s", port)
        return self.status()

    def capture_middle(self) -> dict[str, Any]:
        session = self._calibration
        if session is None or self._state != "WAITING_MIDDLE":
            raise RuntimeError("请先开始主臂校准。")
        if not self._previous_calibration:
            self._previous_calibration = session.robot.bus.read_calibration()
        _, current = session.capture_middle()
        self._calibration_positions = _raw_by_joint(current)
        self._state = "MIDDLE_CAPTURED"
        self._error = None
        self._message = "中位已记录。开始范围采集后，依次摆动六个关节。"
        return self.status()

    def start_range(self) -> dict[str, Any]:
        session = self._calibration
        if session is None or self._state != "MIDDLE_CAPTURED":
            raise RuntimeError("请先确认主臂六轴中位。")
        self._calibration_ranges = {}
        self._observe_calibration(session)
        self._started_at = self._last_poll = time.monotonic()
        self._state = "CALIBRATING_RANGE"
        self._error = None
        self._message = "正在采集六轴范围，完成后可以提前保存。"
        return self.status()

    def _observe_calibration(self, session: SO101CalibrationSession) -> None:
        current = _raw_by_joint(session.read_raw_positions())
        self._calibration_positions = current
        self._calibration_ranges = _extend_ranges(
            self._calibration_ranges,
            {joint: value for joint, value in current.items() if _valid_raw(value)},
        )

    def save_calibration(self) -> dict[str, Any]:
        return self._save_calibration(automatic=False)

    def _save_calibration(self, *, automatic: bool) -> dict[str, Any]:
        session = self._calibration
        if session is None or self._state != "CALIBRATING_RANGE":
            raise RuntimeError("主臂范围采集尚未开始。")
        self._observe_calibration(session)
        ranges = self._calibration_ranges
        incomplete = [
            str(index) for index, joint in enumerate(ALL_JOINTS, 1)
            if joint not in ranges or ranges[joint][1] - ranges[joint][0] < MIN_CALIBRATION_SPAN
        ]
        if incomplete:
            self._error = f"主臂舵机 {', '.join(incomplete)} 的采集范围太小。"
            self._message = (
                "采集时间结束。请重新开始范围采集。" if automatic else "继续摆动上述关节，再保存校准。"
            )
            if automatic:
                self._state = "MIDDLE_CAPTURED"
                return self.status()
            raise RuntimeError(self._error)

        mins = {joint.removesuffix(".pos"): int(pair[0]) for joint, pair in ranges.items()}
        maxes = {joint.removesuffix(".pos"): int(pair[1]) for joint, pair in ranges.items()}
        try:
            session.finish(mins, maxes)
        except Exception as exc:
            self._state = "FAULT"
            self._error = str(exc)
            self._message = "主臂校准保存失败。"
            raise
        finally:
            self._calibration = None
            self._previous_calibration = {}
            self.port = None
        self._state = "CALIBRATED"
        self._error = None
        self._message = "主臂六轴校准已保存。"
        LOGGER.info("leader calibration saved | ranges=%s", ranges)
        return self.status()

    def start_recording(
        self, port: str, *, sync: bool = False, directions: Mapping[str, int] | None = None,
        metadata: dict[str, str] | None = None,
        follower_start: Mapping[str, float] | None = None,
        follower_calibration_fingerprint: str | None = None,
    ) -> dict[str, Any]:
        if self.active:
            raise RuntimeError("请先结束当前主臂校准或录制。")
        if directions is not None and (
            not isinstance(directions, Mapping) or set(directions) != set(ALL_JOINTS)
            or any(type(value) is not int or value not in (-1, 1) for value in directions.values())
        ):
            raise ValueError("六个关节的方向都必须是 1 或 -1。")
        self._directions = dict(directions) if directions is not None else dict.fromkeys(ALL_JOINTS, 1)
        leader = None
        try:
            leader = _make_leader(port)
            expected = {joint.removesuffix(".pos") for joint in ALL_JOINTS}
            if set(leader.calibration) != expected:
                raise RuntimeError("未找到完整的主臂校准，请先校准主臂。")
            raw_ranges = {
                f"{motor}.pos": (int(calibration.range_min), int(calibration.range_max))
                for motor, calibration in leader.calibration.items()
            }
            for index, joint in enumerate(ALL_JOINTS, 1):
                low, high = raw_ranges[joint]
                if not _valid_raw(low) or not _valid_raw(high) or low >= high:
                    raise RuntimeError(f"主臂 ID{index} 校准范围 {low}～{high} 无效，请重新校准主臂。")
            leader.bus.connect()
            # MotorCalibration equality covers id, direction, offset and both limits.
            if leader.bus.read_calibration() != leader.calibration:
                raise RuntimeError("主臂舵机与保存的校准不匹配，请重新校准主臂。")
            leader.bus.disable_torque()
            positions = self._read_positions(leader)
        except Exception as exc:
            if leader is not None:
                try:
                    self._disconnect_leader(leader)
                except Exception:
                    LOGGER.exception("leader connection cleanup failed | port=%s", port)
            self._state = "FAULT"
            self._error = str(exc)
            self._message = "主臂录制未能开始。"
            raise

        self._leader = leader
        self._calibration_port = port
        self._calibration_positions = {}
        self._calibration_ranges = raw_ranges
        self._samples = [{"time_s": 0.0, "positions": positions}]
        self._follower_samples = (
            [{"time_s": 0.0, "positions": dict(follower_start)}]
            if sync and follower_start is not None else []
        )
        self._ranges = _extend_ranges({}, positions)
        self._recording_metadata = {**metadata, "id": uuid.uuid4().hex} if metadata is not None else {}
        if sync and follower_calibration_fingerprint:
            self._recording_metadata["follower_calibration_fingerprint"] = follower_calibration_fingerprint
        self._result = None
        self._started_at = self._last_poll = time.monotonic()
        self._sync = sync
        self.port = port
        self._state = "RECORDING"
        self._error = None
        self._message = "正在录制主臂，从臂同步跟随。" if sync else "正在录制主臂。"
        LOGGER.info("leader recording started | port=%s sync=%s directions=%s", port, sync, self._directions)
        return self.status()

    def _read_positions(self, leader: Any) -> dict[str, float]:
        action = leader.get_action()
        if set(action) != set(ALL_JOINTS):
            raise RuntimeError("主臂未返回全部六个关节的位置。")
        positions = {}
        for joint in ALL_JOINTS:
            value = float(action[joint])
            low = 0 if joint == GRIPPER else -100
            if not math.isfinite(value) or not low <= value <= 100:
                raise RuntimeError(f"主臂关节 {joint} 的位置无效。")
            if self._directions[joint] == -1:
                value = 100 - value if joint == GRIPPER else -value
            positions[joint] = value
        return positions

    def poll(self, now: float) -> dict[str, float] | None:
        leader = self._leader
        if leader is not None:
            elapsed = now - self._started_at
            if elapsed >= MAX_RECORDING_SECONDS or len(self._samples) >= MAX_RECORDING_SAMPLES:
                self.stop_recording()
            elif now - self._last_poll >= RECORD_INTERVAL_S:
                positions = self._read_positions(leader)
                self._samples.append({"time_s": elapsed, "positions": positions})
                self._ranges = _extend_ranges(self._ranges, positions)
                self._last_poll = now
                return positions
        elif self._calibration is not None and now - self._last_poll >= CALIBRATION_INTERVAL_S:
            self._last_poll = now
            if self._state == "CALIBRATING_RANGE":
                if now - self._started_at >= CALIBRATION_DURATION_S:
                    self._save_calibration(automatic=True)
                else:
                    self._observe_calibration(self._calibration)
            else:
                self._calibration_positions = _raw_by_joint(self._calibration.read_raw_positions())
        return None

    @staticmethod
    def _disconnect_leader(leader: Any) -> None:
        if leader.bus.is_connected:
            leader.bus.disconnect(disable_torque=False)

    def record_follower_sample(self, positions: Mapping[str, float], now: float) -> None:
        samples = self._follower_samples
        if not self.sync_active or not samples or len(samples) >= MAX_RECORDING_SAMPLES:
            return
        elapsed = now - self._started_at
        if samples[-1]["time_s"] < elapsed <= MAX_RECORDING_SECONDS:
            samples.append({"time_s": elapsed, "positions": dict(positions)})

    def invalidate_follower_replay(self) -> None:
        self._recording_metadata.pop("follower_calibration_fingerprint", None)
        if self._result is not None:
            self._result.pop("follower_calibration_fingerprint", None)

    def stop_recording(self) -> dict[str, Any]:
        leader = self._leader
        if leader is None:
            if self._result is not None:
                return self._result
            raise RuntimeError("当前没有可用的主臂录制。")
        self._close_recording(leader)
        if len(self._samples) < 2:
            self._recording_metadata = {}
            self._state = "IDLE"
            self._message = "录制时间过短，请重新录制。"
            raise RuntimeError(self._message)
        self._result = {
            **self._recording_metadata,
            "duration_s": self._samples[-1]["time_s"],
            "samples": list(self._samples),
            "ranges": {joint: list(pair) for joint, pair in self._ranges.items()},
            "calibration_ranges": {joint: list(pair) for joint, pair in self._calibration_ranges.items()},
        }
        if len(self._follower_samples) >= 2:
            self._result["follower_samples"] = list(self._follower_samples)
        self._state = "RECORDED"
        self._message = "录制已完成，可以预览或生成动作步骤。"
        LOGGER.info(
            "leader recording finished | samples=%d follower_samples=%d duration_s=%.3f",
            len(self._samples), len(self._follower_samples), self._result["duration_s"],
        )
        return self._result

    def _close_recording(self, leader: Any) -> None:
        try:
            self._disconnect_leader(leader)
        finally:
            self._leader = None
            self._sync = False
            self.port = None

    def cancel(self, reason: str = "用户取消") -> dict[str, Any]:
        session = self._calibration
        leader = self._leader
        try:
            if session is not None:
                try:
                    if self._previous_calibration:
                        session.robot.bus.write_calibration(self._previous_calibration)
                finally:
                    session.cancel()
            elif leader is not None:
                self._close_recording(leader)
                self._samples = []
                self._follower_samples = []
                self._ranges = {}
                self._recording_metadata = {}
        finally:
            self._calibration = None
            self._previous_calibration = {}
            if session is not None:
                self._calibration_port = None
                self._calibration_positions = {}
                self._calibration_ranges = {}
            self.port = None
            self._state = "IDLE"
            self._error = None
            self._message = f"主臂操作已结束（{reason}）。"
        LOGGER.info("leader operation cancelled | reason=%s", reason)
        return self.status()
