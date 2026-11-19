"""Small, local image-space gaze control. No camera, robot, or filesystem IO.

The Jacobian maps normalized joint *position units* to normalized image x/y.
It is measured for a particular wrist-camera mounting and arm configuration;
neither a robot model nor a guessed motor sign is a substitute for that data.
The caller retains joint limits, proximity/lease checks, and the sole motor writer.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np


YAW = "shoulder_pan.pos"
PITCH = "wrist_flex.pos"
_POSE_LIMITS = {"shoulder_lift.pos": 25.0, "elbow_flex.pos": 30.0, "wrist_roll.pos": 8.0}
_REQUIRED = (YAW, PITCH, *_POSE_LIMITS)
_FRAME_MAX_AGE = 0.25


def _center(value: Sequence[float] | None) -> np.ndarray:
    point = np.asarray(value, dtype=float)
    if point.shape != (2,) or not np.all(np.isfinite(point)) or np.any(point < 0) or np.any(point > 1):
        raise ValueError("A visible, finite head center inside the image is required.")
    return point


def _positions(value: Mapping[str, float]) -> dict[str, float]:
    result = {joint: float(value[joint]) for joint in _REQUIRED}
    if not all(math.isfinite(position) for position in result.values()):
        raise ValueError("Joint feedback must be finite.")
    return result


def _frame_valid(frame_at: float, now: float) -> bool:
    return (math.isfinite(frame_at) and math.isfinite(now)
            and -0.02 <= now - frame_at <= _FRAME_MAX_AGE)


def _metadata(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("Gaze metadata must be an object.")
    try:
        return json.loads(json.dumps(dict(value), allow_nan=False))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Gaze metadata must contain finite JSON values.") from exc


@dataclass(frozen=True)
class GazeCalibration:
    jacobian: Any
    reference_positions: Mapping[str, float]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        matrix = np.array(self.jacobian, dtype=float, copy=True)
        if matrix.shape != (2, 2) or not np.all(np.isfinite(matrix)):
            raise ValueError("Gaze Jacobian must be a finite 2 by 2 matrix.")
        singular = np.linalg.svd(matrix, compute_uv=False)
        # A 2-unit probe must produce a measurable response, without implausible
        # image jumps. Near-collinear axes cannot control both image dimensions.
        if singular[-1] < 0.001 or singular[0] > 0.25 or singular[0] / singular[-1] > 30.0:
            raise ValueError("Gaze calibration response is too small, too large, or poorly observable.")
        try:
            reference = _positions(self.reference_positions)
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Gaze calibration requires finite reference joint positions.") from exc
        matrix.setflags(write=False)
        object.__setattr__(self, "jacobian", matrix)
        object.__setattr__(self, "reference_positions", reference)
        object.__setattr__(self, "metadata", _metadata(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {"version": 1, "jacobian": self.jacobian.tolist(),
                "reference_positions": dict(self.reference_positions), "metadata": _metadata(self.metadata)}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> GazeCalibration:
        if (not isinstance(value, Mapping) or type(value.get("version")) is not int
                or value.get("version") != 1):
            raise ValueError("Unsupported gaze calibration version.")
        try:
            return cls(value["jacobian"], value["reference_positions"], value.get("metadata", {}))
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"Invalid gaze calibration: {exc}") from exc


@dataclass(frozen=True)
class GazeResult:
    delta: tuple[float, float] = (0.0, 0.0)
    state: str = "waiting_frame"
    reason: str = "Waiting for a fresh head frame."
    error: tuple[float, float] | None = None
    fault_code: str | None = None


class ImageGazeController:
    """Damped 2-axis IBVS with conservative frame and feedback checks."""

    def __init__(self, calibration: GazeCalibration, *, gain: float = 1.8) -> None:
        if not isinstance(calibration, GazeCalibration) or not math.isfinite(gain) or not 0 < gain <= 5:
            raise ValueError("A validated calibration and positive finite gaze gain are required.")
        self.calibration = calibration
        self.gain = gain
        matrix = calibration.jacobian
        self._inverse = matrix.T @ np.linalg.inv(matrix @ matrix.T + 0.002 ** 2 * np.eye(2))
        self._fault: tuple[str, str] | None = None
        self._fault_code: str | None = None
        self.state = "waiting_frame"
        self.reason = "Waiting for a fresh head frame."
        self.reset_feedback()

    def reset_feedback(self, *, clear_fault: bool = False) -> None:
        """Forget temporal feedback; only an explicit clear releases a fault."""
        self._last_frame: float | None = None
        self._last_error: float | None = None
        self._last_positions: np.ndarray | None = None
        self._last_delta = np.zeros(2)
        self._target_id: Any = None
        self._has_target = False
        self._growing = 0
        self._growth = 0.0
        self._error: tuple[float, float] | None = None
        self._reset_progress()
        if clear_fault:
            self._fault = None
            self._fault_code = None
        self.state, self.reason = self._fault or ("waiting_frame", "Waiting for a fresh head frame.")

    def _result(
        self, state: str, reason: str, delta: Sequence[float] = (0.0, 0.0), *, remember: bool = True,
    ) -> GazeResult:
        self.state, self.reason = state, reason
        if remember:
            self._last_delta = np.asarray(delta, dtype=float)
        return GazeResult((float(delta[0]), float(delta[1])), state, reason, self._error, self._fault_code)

    def _reset_progress(self) -> None:
        self._progress_at: float | None = None
        self._progress_error: float | None = None
        self._progress_low: np.ndarray | None = None
        self._progress_high: np.ndarray | None = None
        self._progress_command = 0.0

    def _check_progress(
        self, frame_at: float, previous_frame: float, error: np.ndarray, measured: np.ndarray,
    ) -> tuple[str, str] | None:
        """Watch issued corrections against encoders and image improvement.

        Small residual error near the deadband needs time to settle. Duplicate
        frames never reach this method, and long gaps cannot count as observed
        time with a stalled motor. Progress above detector noise renews the window.
        """
        if (float(np.max(np.abs(error))) <= 0.065 or np.linalg.norm(self._last_delta) <= 1e-5
                or frame_at - previous_frame > _FRAME_MAX_AGE):
            self._reset_progress()
            return None
        magnitude = float(np.linalg.norm(error))
        improvement = max(0.005, 0.025 * (self._progress_error or magnitude))
        if self._progress_at is None or magnitude <= self._progress_error - improvement:
            self._progress_at, self._progress_error = frame_at, magnitude
            self._progress_low, self._progress_high = measured.copy(), measured.copy()
            self._progress_command = 0.0
            return None
        self._progress_low = np.minimum(self._progress_low, measured)
        self._progress_high = np.maximum(self._progress_high, measured)
        self._progress_command += float(np.linalg.norm(self._last_delta))
        elapsed = frame_at - self._progress_at
        # Do not diagnose a stalled servo from sub-resolution requested motion.
        if self._progress_command < 0.5:
            return None
        excursion = float(np.linalg.norm(self._progress_high - self._progress_low))
        if elapsed >= 2.0 and excursion < 0.15:
            return "stalled", "Gaze corrections produced almost no encoder motion for two seconds; gaze is held."
        if elapsed >= 3.0:
            return "not_converging", "Joints moved but the head did not approach the image center for three seconds; gaze is held."
        return None

    def update(
        self, center: Sequence[float] | None, positions: Mapping[str, float],
        frame_at: float, now: float, target_id: Any = None,
    ) -> GazeResult:
        if self._fault is not None:
            return self._result(*self._fault)
        try:
            measured = _positions(positions)
            frame_at, now = float(frame_at), float(now)
        except (KeyError, TypeError, ValueError, OverflowError):
            self._fault = ("invalid", "Joint feedback or frame time is invalid; gaze is held.")
            self._fault_code = "invalid_feedback"
            return self._result(*self._fault)
        for joint, tolerance in _POSE_LIMITS.items():
            if abs(measured[joint] - self.calibration.reference_positions[joint]) > tolerance:
                self._fault = ("invalid", f"Camera geometry changed at {joint}; recalibrate gaze.")
                self._fault_code = "pose_changed"
                return self._result(*self._fault)
        try:
            point = _center(center)
        except (TypeError, ValueError, OverflowError):
            self._error = None
            self._last_error = self._last_positions = None
            self._growing, self._growth = 0, 0.0
            self._reset_progress()
            return self._result("target_lost", "Head target is missing; gaze is held.")
        if not _frame_valid(frame_at, now):
            self._error = None
            self._last_error = self._last_positions = None
            self._growing, self._growth = 0, 0.0
            self._reset_progress()
            return self._result("stale", "Head frame is stale; gaze is held.")
        if self._last_frame is not None and frame_at <= self._last_frame:
            return self._result("waiting_frame", "No new head frame; no new gaze command.", remember=False)
        changed = self._has_target and target_id != self._target_id
        previous_frame = self._last_frame
        self._last_frame = frame_at
        self._target_id, self._has_target = target_id, True
        joint_vector = np.array([measured[YAW], measured[PITCH]])
        error = point - 0.5
        self._error = float(error[0]), float(error[1])
        magnitude = float(np.linalg.norm(error))
        if changed or previous_frame is None or self._last_positions is None:
            self._last_error, self._last_positions = magnitude, joint_vector
            self._growing, self._growth = 0, 0.0
            self._reset_progress()
            return self._result("target_changed" if changed else "waiting_frame", "Confirming head feedback before moving.")
        rise = magnitude - self._last_error
        moved = float(np.linalg.norm(joint_vector - self._last_positions))
        if rise > 0.003 and moved >= 0.05 and np.linalg.norm(self._last_delta) > 0.001:
            self._growing += 1
            self._growth += rise
        else:
            self._growing, self._growth = 0, 0.0
        self._last_error, self._last_positions = magnitude, joint_vector
        if self._growing >= 4 and self._growth >= 0.025:
            self._fault = ("fault", "Head error increased despite measured joint motion; check target and recalibrate direction.")
            self._fault_code = "diverging"
            return self._result(*self._fault)
        failure = self._check_progress(frame_at, previous_frame, error, joint_vector)
        if failure is not None:
            self._fault_code, reason = failure
            self._fault = ("fault", reason)
            return self._result(*self._fault)
        # Soft deadband avoids a discontinuous full-speed command at its edge.
        error = np.sign(error) * np.maximum(np.abs(error) - 0.04, 0.0)
        if not np.any(error):
            return self._result("centered", "Head is centered.")
        velocity = -self.gain * (self._inverse @ error)
        velocity /= max(1.0, float(np.max(np.abs(velocity) / np.array([10.0, 8.0]))))
        delta = velocity * min(0.1, frame_at - previous_frame)
        delta /= max(1.0, float(np.max(np.abs(delta))) / 0.5)
        return self._result("tracking", "Centering the head using calibrated image feedback.", delta)


@dataclass(frozen=True)
class GazeCalibrationStep:
    targets: dict[str, float] = field(default_factory=dict)
    state: str = "idle"
    reason: str = "Gaze calibration has not started."
    calibration: GazeCalibration | None = None

    @property
    def progress(self) -> float:
        return {"baseline": 0.0, "probe_yaw": 0.15, "return_yaw": 0.35,
                "baseline_pitch": 0.5, "probe_pitch": 0.6, "return_pitch": 0.85,
                "complete": 1.0}.get(self.state, 0.0)


class GazeCalibrator:
    """Explicit, bounded two-axis probes with no automatic motion on failure.

    The caller must freeze expressive motion and verify head identity/proximity,
    lease, limits, and STOP again before writing returned absolute targets. At
    normalized endpoints the caller must additionally prove raw feedback has not
    been clipped; a numeric -100/+100 alone cannot establish that observability.
    Every failure returns an empty dict: recovery never blindly returns to origin.
    """

    def __init__(
        self, bounds: Mapping[str, tuple[float, float]], *, metadata: Mapping[str, Any] | None = None,
        probe_distance: float = 2.0, max_step: float = 0.2,
    ) -> None:
        if not math.isfinite(probe_distance) or not 0.5 <= probe_distance <= 2.0:
            raise ValueError("Gaze probe distance must be between 0.5 and 2 position units.")
        if not math.isfinite(max_step) or max_step <= 0.0:
            raise ValueError("Gaze calibration step limit must be positive and finite.")
        self.max_step = min(0.2, max_step)
        self.bounds = {}
        for joint in (YAW, PITCH):
            try:
                low, high = map(float, bounds[joint])
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"Missing or invalid gaze calibration bounds: {joint}.") from exc
            if not math.isfinite(low) or not math.isfinite(high) or high - low <= 1.0:
                raise ValueError("Gaze calibration requires finite, nonempty safe bounds.")
            self.bounds[joint] = (low, high)
        self.metadata = _metadata({} if metadata is None else metadata)
        self.probe_distance = probe_distance
        self.state, self.reason = "idle", "Gaze calibration has not started."
        self.calibration: GazeCalibration | None = None
        self._samples: list[tuple[np.ndarray, dict[str, float]]] = []

    @property
    def active(self) -> bool:
        return self.state not in {"idle", "complete", "aborted"}

    def _result(self, targets: dict[str, float] | None = None) -> GazeCalibrationStep:
        return GazeCalibrationStep(targets or {}, self.state, self.reason, self.calibration)

    def abort(self, reason: str = "Gaze calibration was cancelled.") -> GazeCalibrationStep:
        self.state, self.reason, self.calibration = "aborted", reason, None
        self._samples.clear()
        return self._result()

    def cancel(self, reason: str = "Gaze calibration was cancelled.") -> GazeCalibrationStep:
        return self.abort(reason)

    def begin(
        self, positions: Mapping[str, float], center: Sequence[float] | None,
        frame_at: float, now: float, target_id: Any = None,
    ) -> GazeCalibrationStep:
        self.calibration = None
        self._samples.clear()
        try:
            self._origin = _positions(positions)
            _center(center)
            frame_at, now = float(frame_at), float(now)
            if not _frame_valid(frame_at, now):
                raise ValueError("A fresh head frame is required to begin gaze calibration.")
            self._probe = {}
            self._probe_travel = {}
            self._return = {}
            for joint, (low, high) in self.bounds.items():
                value = self._origin[joint]
                if not low <= value <= high:
                    raise ValueError(f"Joint feedback is outside its calibration bounds: {joint}.")
                direction = 1.0 if high - value >= value - low else -1.0
                available = high - value if direction > 0 else value - low
                distance = min(self.probe_distance, available - 0.1)
                if distance < 0.5:
                    raise ValueError(f"Insufficient safe probe travel at {joint}.")
                self._probe[joint] = value + direction * distance
                self._probe_travel[joint] = distance
                # Do not write the recorded endpoint again on return. A small
                # interior rest point remains observable after a one-sided probe.
                self._return[joint] = max(low + 0.5, min(high - 0.5, value))
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            return self.abort(str(exc))
        self.state, self.reason = "baseline", "Keep the same head still while measuring gaze direction."
        self._started_at = self._last_tick = now
        self._last_frame = frame_at
        self._target_id = target_id
        self._last_point = _center(center)
        self._command = {joint: self._origin[joint] for joint in (YAW, PITCH)}
        self._rest = dict(self._command)
        self._columns: list[np.ndarray] = []
        self._arrived_at: float | None = None
        return self._result()

    def _active_axis(self) -> str | None:
        if self.state in {"probe_yaw", "return_yaw"}:
            return YAW
        if self.state in {"probe_pitch", "return_pitch"}:
            return PITCH
        return None

    def _destination(self) -> dict[str, float]:
        joint = self._active_axis()
        if joint is None:
            return {}
        goal = self._probe[joint] if self.state.startswith("probe_") else self._return[joint]
        return {joint: goal}

    def _sample(self, point: np.ndarray, measured: dict[str, float]) -> bool:
        if self._samples:
            first_point, first_positions = self._samples[0]
            # Compare the whole window, not successive pairs: slow head drift
            # must not pass as three stable samples simply by taking tiny steps.
            if (np.linalg.norm(point - first_point) > 0.015
                    or any(abs(measured[joint] - first_positions[joint]) > 0.15 for joint in (YAW, PITCH))):
                self._samples.clear()
        self._samples.append((point, measured))
        return len(self._samples) >= 3

    def _mean(self) -> tuple[np.ndarray, dict[str, float]]:
        return (np.mean([point for point, _ in self._samples], axis=0),
                {joint: float(np.mean([position[joint] for _, position in self._samples]))
                 for joint in (YAW, PITCH)})

    def update(
        self, positions: Mapping[str, float], center: Sequence[float] | None,
        frame_at: float, now: float, target_id: Any = None,
    ) -> GazeCalibrationStep:
        if not self.active:
            return self._result()
        try:
            measured, point = _positions(positions), _center(center)
            frame_at, now = float(frame_at), float(now)
            if not _frame_valid(frame_at, now):
                raise ValueError("Head frame became stale during gaze calibration.")
            if now - self._started_at > 20.0 or now < self._last_tick:
                raise ValueError("Gaze calibration timed out or its clock changed.")
            if target_id != self._target_id:
                raise ValueError("Head identity changed during gaze calibration.")
            for joint in _POSE_LIMITS:
                if abs(measured[joint] - self._origin[joint]) > 0.5:
                    raise ValueError(f"Non-probed joint moved during calibration: {joint}.")
            active_axis = self._active_axis()
            for joint, (low, high) in self.bounds.items():
                if not low <= measured[joint] <= high:
                    raise ValueError(f"Joint feedback left its safe calibration bounds: {joint}.")
                if joint != active_axis and abs(measured[joint] - self._rest[joint]) > 0.4:
                    raise ValueError(f"Non-probed gaze axis moved: {joint}.")
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            return self.abort(str(exc))
        if frame_at <= self._last_frame:
            return self._result()
        if np.linalg.norm(point - self._last_point) > 0.18:
            return self.abort("Head position jumped during gaze calibration.")
        self._last_point, self._last_frame = point, frame_at
        dt = min(0.05, max(0.0, now - self._last_tick))
        self._last_tick = now
        destination = self._destination()
        arrival_positions = destination or self._rest
        arrived = all(
            abs(measured[joint] - value) <= 0.25
            and (joint not in destination or abs(self._command[joint] - value) <= 1e-9)
            for joint, value in arrival_positions.items()
        )
        if arrived and self._arrived_at is None:
            self._arrived_at = now
        # Only use images captured after encoder arrival, rather than a queued
        # image of an axis that was still moving when that image was taken.
        settled = arrived and frame_at >= self._arrived_at + 0.05 - 1e-9
        if settled and self._sample(point, measured):
            mean_point, mean_positions = self._mean()
            previous_state = self.state
            if previous_state in {"baseline", "baseline_pitch"}:
                self._baseline = mean_point, mean_positions
                self.state = "probe_yaw" if previous_state == "baseline" else "probe_pitch"
            elif previous_state in {"probe_yaw", "probe_pitch"}:
                joint = YAW if previous_state == "probe_yaw" else PITCH
                base_point, base_positions = self._baseline
                displacement = mean_positions[joint] - base_positions[joint]
                if abs(displacement) < self._probe_travel[joint] * 0.7:
                    return self.abort("The gaze probe did not produce sufficient measured joint travel.")
                self._columns.append((mean_point - base_point) / displacement)
                self.state = "return_yaw" if joint == YAW else "return_pitch"
            else:
                joint = YAW if previous_state == "return_yaw" else PITCH
                base_point, base_positions = self._baseline
                column = self._columns[0 if joint == YAW else 1]
                expected = base_point + column * (mean_positions[joint] - base_positions[joint])
                if np.linalg.norm(mean_point - expected) > 0.025:
                    return self.abort("Head moved while returning from the gaze probe; retry with a still target.")
                self._rest[joint] = mean_positions[joint]
                if previous_state == "return_yaw":
                    self.state = "baseline_pitch"
                else:
                    try:
                        reference = dict(measured)
                        reference.update(mean_positions)
                        self.calibration = GazeCalibration(np.column_stack(self._columns), reference, self.metadata)
                    except ValueError as exc:
                        return self.abort(str(exc))
                    self.state, self.reason = "complete", "Gaze direction calibration completed."
                    return self._result()
            self._samples.clear()
            self._arrived_at = None
            self.reason = f"Gaze calibration: {self.state}. Keep the same head still."
            destination = self._destination()
        elif not arrived:
            self._samples.clear()
            self._arrived_at = None
        # Even on a long processing pause never jump a target by more than 0.2.
        step = min(self.max_step, 4.0 * dt)
        targets = {}
        for joint, goal in destination.items():
            previous = self._command[joint]
            self._command[joint] = previous + max(-step, min(step, goal - previous))
            low, high = self.bounds[joint]
            if not low <= self._command[joint] <= high:
                return self.abort("Gaze probe target would leave its calibration bounds.")
            targets[joint] = self._command[joint]
        return self._result(targets)
