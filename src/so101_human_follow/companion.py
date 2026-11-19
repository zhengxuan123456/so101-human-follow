"""Local companion behavior. The caller owns camera freshness and motion safety."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from .pose_math import Landmark
from .safety import gripper_position_for_hand

if TYPE_CHECKING:
    from .pose_detector import PoseDetection


_YAW = "shoulder_pan.pos"
_SHOULDER = "shoulder_lift.pos"
_ELBOW = "elbow_flex.pos"
_WRIST = "wrist_flex.pos"
_GRIPPER = "gripper.pos"
_GRIPPER_SPEED = 120.0
_TRAVEL = {_YAW: 20.0, _SHOULDER: 15.0, _ELBOW: 20.0, _WRIST: 24.0}
_SPEED = {_YAW: 12.0, _SHOULDER: 23.0, _ELBOW: 26.0, _WRIST: 22.0}
_SEARCH_SPEED = 9.0
# Nose, eyes, and corners of the mouth in MediaPipe Pose's 33-point layout.
_FACE_POINTS = (0, 2, 5, 9, 10)


class CompanionRangeError(ValueError):
    """A moving axis left its envelope; stop and rebuild the remaining axes."""


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def face_center(
    landmarks: Sequence[Landmark], *, min_confidence: float = 0.55,
) -> tuple[float, float] | None:
    """Aim at a visible face using the landmarks of the tracked person."""
    reliable = [
        landmarks[index] for index in _FACE_POINTS
        if index < len(landmarks)
        and all(math.isfinite(value) for value in (
            landmarks[index].x, landmarks[index].y, landmarks[index].visibility, landmarks[index].presence,
        ))
        and landmarks[index].visibility >= min_confidence
        and landmarks[index].presence >= min_confidence
        and 0.0 <= landmarks[index].x <= 1.0
        and 0.0 <= landmarks[index].y <= 1.0
    ]
    if len(reliable) < 2:
        return None
    if max(max(point.x for point in reliable) - min(point.x for point in reliable),
           max(point.y for point in reliable) - min(point.y for point in reliable)) < 0.015:
        return None
    return (
        sum(point.x for point in reliable) / len(reliable),
        sum(point.y for point in reliable) / len(reliable),
    )


def head_target(detection: PoseDetection) -> tuple[float, float] | None:
    """Prefer the independent head channel, including frames without a body."""
    if (getattr(detection, "head_source", None) == "ambiguous"
            or getattr(detection, "head_state", "") in {
                "confirming", "unverified", "unavailable", "ambiguous", "lost",
            }
            or detection.people_count > 1 or detection.reason == "person changed"):
        return None
    center = getattr(detection, "head_center", None)
    confidence = getattr(detection, "head_confidence", 0.0)
    if center is not None:
        try:
            x, y = map(float, center)
            confidence = float(confidence)
        except (TypeError, ValueError, OverflowError):
            pass
        else:
            if (all(math.isfinite(value) for value in (x, y, confidence))
                    and 0.55 <= confidence <= 1.0 and 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                return x, y
    return face_center(detection.landmarks)


def companion_hand_openness(detection: PoseDetection) -> float | None:
    """Return the current hand channel only if the detector marked it reliable."""
    if detection.features is None or "hand_open" not in detection.valid_features:
        return None
    openness = detection.features.hand_open
    return openness if 0.0 <= openness <= 1.0 else None


class CompanionMotion:
    """Combine a continuous shoulder/elbow trajectory and image-based gaze."""

    def __init__(
        self, initial_positions: Mapping[str, float], limits: Mapping[str, tuple[float, float]],
        *, gripper_open_is_high: bool = True,
    ) -> None:
        self._gripper_limits = limits.get(_GRIPPER)
        self._gripper_open_is_high = gripper_open_is_high
        self._gripper_position = initial_positions.get(_GRIPPER)
        self.gripper_active = False
        self._limits = {joint: limits[joint] for joint in _TRAVEL}
        self.blocked_joints: dict[str, str] = {}
        # Keep real encoder positions intact. An axis outside its envelope is
        # omitted from powered targets while the remaining axes can still work.
        for joint, (low, high) in tuple(self._limits.items()):
            position = self._position(initial_positions, joint)
            if not low <= position <= high:
                label = "舵机 1" if joint == _YAW else joint
                self.blocked_joints[joint] = (
                    f"{label} 当前位置 {position:.2f} 超出陪伴限位 "
                    f"[{low:.2f}, {high:.2f}]，已暂停该轴，请恢复到允许范围后重新开启陪伴。"
                )
                del self._limits[joint]
        self._origin = self.measured_positions(initial_positions)
        self._bounds: dict[str, tuple[float, float]] = {}
        for joint, (low, high) in self._limits.items():
            position = self._origin[joint]
            travel = _TRAVEL[joint]
            # Continuous motion should not command calibrated endpoints: normal
            # tracking error can otherwise put the next encoder sample outside.
            margin = min((high - low) * 0.05, travel * 0.25)
            self._bounds[joint] = (
                max(low + margin, position - travel), min(high - margin, position + travel),
            )
            if joint in {_YAW, _WRIST}:
                # A verified near-edge calibration ends slightly inward. Keep
                # that chosen resting pose instead of jumping to the wider
                # continuous-motion margin; outward travel stays bounded there.
                bounded_low, bounded_high = self._bounds[joint]
                self._bounds[joint] = min(bounded_low, position), max(bounded_high, position)
        self._last_positions = dict(self._origin)
        self._phase = 0.0
        self._search_toward_high: bool | None = None
        self._search_pitch_toward_high: bool | None = None
        self._body_gain = 0.0
        self._yaw_trim = 0.0
        self._pitch_trim = 0.0
        self._gaze_limit_reason = ""

    @property
    def tracks_yaw(self) -> bool:
        return _YAW in self._limits

    @property
    def tracks_pitch(self) -> bool:
        return _WRIST in self._limits

    def can_search(self, vertical: bool = False) -> bool:
        return self.tracks_yaw or (vertical and self.tracks_pitch)

    @property
    def yaw_block_reason(self) -> str:
        return self.blocked_joints.get(_YAW, "")

    @property
    def blocked_reason(self) -> str:
        return " ".join(self.blocked_joints.values())

    @property
    def gaze_limit_reason(self) -> str:
        return self._gaze_limit_reason

    @property
    def gaze_bounds(self) -> dict[str, tuple[float, float]]:
        """Expose a copy of the existing bounded envelope for the two gaze axes."""
        return {joint: self._bounds[joint] for joint in (_YAW, _WRIST) if joint in self._bounds}

    @staticmethod
    def _position(positions: Mapping[str, float], joint: str) -> float:
        try:
            position = float(positions[joint])
        except (KeyError, TypeError, ValueError, OverflowError):
            position = math.nan
        if not math.isfinite(position):
            raise ValueError(f"{joint} 缺少有效的实测位置。")
        return position

    def measured_positions(self, positions: Mapping[str, float]) -> dict[str, float]:
        """Validate encoder feedback on the axes used by the companion preset."""
        measured = {}
        for joint, (low, high) in self._limits.items():
            position = self._position(positions, joint)
            if not low <= position <= high:
                raise CompanionRangeError(
                    f"{joint} 当前位置 {position:.2f} 超出陪伴限位 [{low:.2f}, {high:.2f}]。"
                )
            measured[joint] = position
        return measured

    def resume_from(self, positions: Mapping[str, float]) -> None:
        """Resume from the encoders while preserving the original motion envelope."""
        # If the measured pose has drifted beyond the relative envelope, the
        # limiter will lead it back at the normal rate toward an in-bounds goal.
        self._last_positions = self.measured_positions(positions)
        self._gripper_position = positions.get(_GRIPPER)
        self.gripper_active = False
        self._search_toward_high = None
        self._search_pitch_toward_high = None
        if self.tracks_pitch:
            shoulder = self._last_positions.get(_SHOULDER, self._origin.get(_SHOULDER, 0.0)) - self._origin.get(_SHOULDER, 0.0)
            elbow = self._last_positions.get(_ELBOW, self._origin.get(_ELBOW, 0.0)) - self._origin.get(_ELBOW, 0.0)
            self._pitch_trim = self._joint_offset(
                _WRIST, self._last_positions[_WRIST] - self._origin[_WRIST] + 1.05 * shoulder + 1.02 * elbow,
            )
        if self.tracks_yaw:
            self._yaw_trim = self._joint_offset(_YAW, self._last_positions[_YAW] - self._origin[_YAW])

    def _joint_offset(self, joint: str, offset: float) -> float:
        if joint not in self._bounds:
            return 0.0
        low, high = self._bounds[joint]
        return _clamp(self._origin[joint] + offset, low, high) - self._origin[joint]

    def step(
        self, dt: float, *, face_target: tuple[float, float] | None,
        amplitude: float = 1.0, search: bool = False, max_step: float = math.inf,
        hand_open: float | None = None, search_vertical: bool = False,
        gaze_delta: tuple[float, float] | None = None,
    ) -> dict[str, float]:
        """Apply the built-in seven-second motion and bounded camera correction.

        Camera corrections come from a separately calibrated image controller.
        Without a correction this method never guesses a camera/motor sign.
        """
        if not math.isfinite(dt) or dt < 0:
            raise ValueError("Companion motion requires a finite nonnegative timestep.")
        if face_target is not None and not all(0.0 <= value <= 1.0 for value in face_target):
            face_target = None

        # A delayed control tick must never catch up with a large physical step.
        dt = min(dt, 0.10)
        self._gaze_limit_reason = ""
        if search:
            self._body_gain = 0.0
            self.gripper_active = False
            targets = self._search_yaw(dt, max_step)
            if search_vertical:
                targets.update(self._search_pitch(dt, max_step))
            else:
                self._search_pitch_toward_high = None
            return targets

        self._search_toward_high = None
        self._search_pitch_toward_high = None
        if face_target is None:
            self._body_gain = 0.0
            return self.gripper_targets(dt, hand_open, amplitude, max_step)

        blend = -math.expm1(-dt / 0.45)
        self._phase = (self._phase + math.tau * dt / 7.0) % math.tau
        x, y = face_target
        edge = max(abs(x - 0.5), abs(y - 0.5))
        # Keep the person in view before adding expressive body motion. A head
        # near an image edge needs the full available camera correction.
        desired_body_gain = amplitude * _clamp(1.0 - (edge - 0.10) / 0.20, 0.0, 1.0)
        yaw_delta, pitch_delta = (0.0, 0.0) if gaze_delta is None else gaze_delta
        if not all(math.isfinite(value) for value in (yaw_delta, pitch_delta)):
            raise ValueError("注视纠偏量必须为有限数值。")
        self._yaw_trim = self._joint_offset(_YAW, self._yaw_trim + yaw_delta) if self.tracks_yaw else 0.0
        self._pitch_trim = self._joint_offset(_WRIST, self._pitch_trim + pitch_delta)
        self._body_gain += blend * (desired_body_gain - self._body_gain)
        shoulder = self._joint_offset(_SHOULDER, 12.0 * self._body_gain * math.sin(self._phase))
        elbow = self._joint_offset(_ELBOW, 16.0 * self._body_gain * math.sin(self._phase + 2.55))
        # In the SO-101 model these three pitch axes are parallel. Compensate its
        # nominal pitch using the modeled normalized-to-angle ratios, then use
        # image feedback to trim the actual camera frame.
        wrist = -1.05 * shoulder - 1.02 * elbow + self._pitch_trim
        offsets = {_YAW: self._yaw_trim, _SHOULDER: shoulder, _ELBOW: elbow, _WRIST: wrist}
        goals = {joint: self._origin[joint] + self._joint_offset(joint, offset)
                 for joint, offset in offsets.items() if joint in self._bounds}
        # The image axes may be coupled. Diagnose the requested joint direction
        # against its final compensated goal, rather than assuming yaw fixes x.
        saturated = []
        if max(abs(x - 0.5), abs(y - 0.5)) > 0.04:
            for joint, correction, label in ((_YAW, yaw_delta, "底座"), (_WRIST, pitch_delta, "腕部")):
                if joint not in goals:
                    continue
                low, high = self._bounds[joint]
                if ((correction < 0.0 and goals[joint] <= low + 1e-6)
                        or (correction > 0.0 and goals[joint] >= high - 1e-6)):
                    saturated.append(label)
        if saturated:
            self._gaze_limit_reason = f"头部尚未居中，{'、'.join(saturated)}已到允许范围，请调整起始姿态后重标定。"
        targets = self._slew(goals, dt, max_step)
        targets.update(self.gripper_targets(dt, hand_open, amplitude, max_step))
        return targets

    def validate_gripper_position(self, position: float | None) -> float:
        if position is None or not math.isfinite(position):
            raise ValueError("夹爪缺少有效的实测位置。")
        if self._gripper_limits is None:
            raise ValueError("夹爪缺少有效的安全限位。")
        low, high = self._gripper_limits
        if not low <= position <= high:
            raise ValueError(f"夹爪当前位置 {position:.1f} 超出陪伴限位 [{low:.1f}, {high:.1f}]。")
        return position

    def gripper_targets(
        self, dt: float, hand_open: float | None, amplitude: float, max_step: float,
    ) -> dict[str, float]:
        self.gripper_active = hand_open is not None and 0.0 <= hand_open <= 1.0
        if not self.gripper_active:
            return {}
        previous = self.validate_gripper_position(self._gripper_position)
        low, high = self._gripper_limits
        # Keep a small margin from the calibrated endpoints during continuous use.
        margin = 0.05 * (high - low)
        goal = gripper_position_for_hand(
            hand_open, (low + margin, high - margin), open_is_high=self._gripper_open_is_high,
        )
        limit = min(max_step, _GRIPPER_SPEED * dt * amplitude)
        self._gripper_position = _clamp(goal, previous - limit, previous + limit)
        return {_GRIPPER: self._gripper_position}

    def _search_yaw(self, dt: float, max_step: float) -> dict[str, float]:
        if not self.tracks_yaw:
            return {}
        low, high = self._bounds[_YAW]
        position = self._last_positions[_YAW]
        if self._search_toward_high is None:
            self._search_toward_high = position <= (low + high) / 2.0
        edge = high if self._search_toward_high else low
        if position == edge:
            self._search_toward_high = not self._search_toward_high
            edge = high if self._search_toward_high else low
        step = min(max_step, _SEARCH_SPEED * dt)
        position += _clamp(edge - position, -step, step)
        self._last_positions[_YAW] = position
        self._yaw_trim = self._joint_offset(_YAW, position - self._origin[_YAW])
        return {_YAW: position}

    def _search_pitch(self, dt: float, max_step: float) -> dict[str, float]:
        """An explicitly allowed slow wrist scan can find a head below the frame."""
        if not self.tracks_pitch:
            return {}
        low, high = self._bounds[_WRIST]
        position = self._last_positions[_WRIST]
        if self._search_pitch_toward_high is None:
            self._search_pitch_toward_high = position <= (low + high) / 2.0
        edge = high if self._search_pitch_toward_high else low
        if position == edge:
            self._search_pitch_toward_high = not self._search_pitch_toward_high
            edge = high if self._search_pitch_toward_high else low
        step = min(max_step, min(_SEARCH_SPEED, 6.0) * dt)
        position += _clamp(edge - position, -step, step)
        self._last_positions[_WRIST] = position
        self._pitch_trim = self._joint_offset(_WRIST, position - self._origin[_WRIST])
        return {_WRIST: position}

    def _slew(self, goals: Mapping[str, float], dt: float, max_step: float) -> dict[str, float]:
        for joint, goal in goals.items():
            previous = self._last_positions[joint]
            step_limit = min(max_step, _SPEED[joint] * dt)
            self._last_positions[joint] = _clamp(goal, previous - step_limit, previous + step_limit)
        return dict(self._last_positions)
