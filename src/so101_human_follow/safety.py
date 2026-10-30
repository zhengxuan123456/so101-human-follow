from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

from .config import ALL_JOINTS, MappingSettings, SafetySettings
from .pose_math import PoseFeatures


FEATURE_BY_JOINT = {
    # ID3 follows the real angle between the operator's upper and lower arm.
    "elbow_flex.pos": "elbow",
    # ID4 follows lifting/lowering of the operator's upper arm.
    "wrist_flex.pos": "lift",
    "wrist_roll.pos": "hand_roll",
    "gripper.pos": "hand_open",
}


def signed_deadband(value: float, width: float) -> float:
    if not math.isfinite(value) or not math.isfinite(width) or width < 0:
        raise ValueError("Deadband inputs must be finite and width must be non-negative.")
    if abs(value) <= width:
        return 0.0
    return math.copysign(abs(value) - width, value)


def clamp(value: float, low: float, high: float) -> float:
    if not low <= high:
        raise ValueError("Invalid clamp bounds.")
    return max(low, min(high, value))


def ema(previous: float, target: float, alpha: float) -> float:
    return alpha * target + (1.0 - alpha) * previous


def slew_limit(previous: float, target: float, max_step: float) -> float:
    return previous + clamp(target - previous, -max_step, max_step)


@dataclass
class MotionController:
    initial: dict[str, float]
    mapping: MappingSettings
    safety: SafetySettings
    dynamic_limits: Mapping[str, tuple[float, float]] | None = None
    fixed_override: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        if set(self.initial) != set(ALL_JOINTS):
            raise ValueError("Initial position must contain all six SO-101 joints.")
        if not all(math.isfinite(float(value)) for value in self.initial.values()):
            raise ValueError("Initial joint positions must be finite.")

        # Connection/startup acceptance is deliberately wider than the automatic
        # motion envelope.  The operator may plug the arm in at any mechanically
        # valid pose inside ``startup_body_abs_limit``; automatic commands are
        # still clamped to each joint's tighter absolute limits below.  ID6 keeps
        # its native 0..100 scale.
        for joint in ALL_JOINTS:
            if joint == "gripper.pos":
                low, high = self.safety.absolute_limits[joint]
            else:
                low = -self.safety.startup_body_abs_limit
                high = self.safety.startup_body_abs_limit
            value = float(self.initial[joint])
            if not low <= value <= high:
                raise ValueError(
                    f"{joint} starts at {value:.2f}, outside its configured startup range "
                    f"[{low:.2f}, {high:.2f}]. Move that joint inside this range before saving."
                )

        self.initial = {key: float(value) for key, value in self.initial.items()}
        self.dynamic_limits = self._validated_dynamic_limits()
        self.fixed = self._fixed_positions()
        self.previous = dict(self.initial)

    def _validated_dynamic_limits(self) -> dict[str, tuple[float, float]] | None:
        if self.dynamic_limits is None:
            return None
        if set(self.dynamic_limits) != set(self.mapping.controlled_joints):
            raise ValueError("Dynamic limits must contain every controlled joint exactly once.")
        checked: dict[str, tuple[float, float]] = {}
        for joint in self.mapping.controlled_joints:
            low, high = self.dynamic_limits[joint]
            low, high = float(low), float(high)
            absolute_low, absolute_high = self.safety.absolute_limits[joint]
            if not all(math.isfinite(value) for value in (low, high)) or low >= high:
                raise ValueError(f"Invalid dynamic limits for {joint}: {(low, high)}")
            if low < absolute_low or high > absolute_high:
                raise ValueError(
                    f"Dynamic limits for {joint} exceed configured absolute limits."
                )
            # A startup pose may sit just outside the automatic envelope.  It is
            # accepted here and, once ARMED, the slew limiter brings a controlled
            # joint back toward the envelope without a one-cycle jump.
            checked[joint] = (low, high)
        return checked

    def _fixed_positions(self) -> dict[str, float]:
        if self.fixed_override is not None and set(self.fixed_override) != set(
            self.mapping.fixed_joints.names
        ):
            raise ValueError("Fixed override must contain every fixed joint exactly once.")
        fixed = {}
        for joint in self.mapping.fixed_joints.names:
            if self.fixed_override is not None:
                value = float(self.fixed_override[joint])
                if abs(value - self.initial[joint]) > self.safety.max_step_per_cycle:
                    raise ValueError(
                        f"Fixed joint {joint} is not at its confirmed position. "
                        "Re-confirm the fixed position before following."
                    )
            elif self.mapping.fixed_joints.mode == "startup":
                value = self.initial[joint]
            else:
                value = self.mapping.fixed_joints.constant_values[joint]
                if abs(value - self.initial[joint]) > self.safety.max_step_per_cycle:
                    raise ValueError(
                        f"Refusing to jump fixed joint {joint} from {self.initial[joint]:.2f} "
                        f"to {value:.2f}. Place it at the configured constant before startup."
                    )
            if self.mapping.fixed_joints.mode == "startup" or self.fixed_override is not None:
                # ID1-ID2 must remain exactly where they were connected/confirmed,
                # even when that fixed pose is outside the automatic motion range.
                fixed[joint] = float(value)
            else:
                low, high = self.safety.absolute_limits[joint]
                fixed[joint] = clamp(float(value), low, high)
        return fixed

    def compute(
        self,
        features: PoseFeatures,
        neutral: PoseFeatures,
        valid_features: frozenset[str] | set[str] | None = None,
    ) -> dict[str, float]:
        action = dict(self.fixed)

        for joint in self.mapping.controlled_joints:
            feature_name = FEATURE_BY_JOINT[joint]
            if valid_features is not None and feature_name not in valid_features:
                action[joint] = self.previous[joint]
                continue
            absolute_low, absolute_high = self.safety.absolute_limits[joint]
            if joint == "gripper.pos":
                # A neutral-centred delta only moved the gripper a few units.
                # Expand the useful camera openness band to the complete physical
                # 0..100 gripper envelope so a fist and an open palm are visibly
                # different.  The configured sign still allows reversal.
                # Deliberately use a steep gesture band: <=0.35 is a full close
                # target and >=0.55 is a full open target.  The small transition
                # band converts a visibly large fist/open gesture into the complete
                # ID6 stroke instead of a small mid-range wobble.
                openness = clamp((float(features.hand_open) - 0.35) / 0.20, 0.0, 1.0)
                if self.mapping.signs[joint] < 0.0:
                    openness = 1.0 - openness
                raw = absolute_low + openness * (absolute_high - absolute_low)
            else:
                feature_delta = getattr(features, feature_name) - getattr(neutral, feature_name)
                if feature_name == "hand_roll":
                    # hand_roll is a normalized circular angle: -1 and +1 meet.
                    feature_delta = (feature_delta + 1.0) % 2.0 - 1.0
                feature_delta = signed_deadband(
                    feature_delta, self.mapping.feature_deadbands[joint]
                )
                raw = (
                    self.initial[joint]
                    + self.mapping.signs[joint]
                    * self.mapping.gains[joint]
                    * feature_delta
                )
                relative_limit = self.safety.relative_limits[joint]
                raw = clamp(
                    raw,
                    self.initial[joint] - relative_limit,
                    self.initial[joint] + relative_limit,
                )
            raw = clamp(raw, absolute_low, absolute_high)
            if self.dynamic_limits is not None:
                dynamic_low, dynamic_high = self.dynamic_limits[joint]
                raw = clamp(raw, dynamic_low, dynamic_high)
            smoothed = ema(self.previous[joint], raw, self.safety.ema_alpha)
            # Hand landmarks can be reliable for only a few frames during a fast
            # fist/open gesture.  The old global six-unit step let ID6 move only a
            # few millimetres before the channel disappeared.  Give the gripper an
            # independent fast-but-bounded step while leaving ID3-ID5 unchanged.
            max_step = (
                max(self.safety.max_step_per_cycle, 25.0)
                if joint == "gripper.pos"
                else self.safety.max_step_per_cycle
            )
            action[joint] = slew_limit(
                self.previous[joint], smoothed, max_step
            )

        ordered = {joint: float(action[joint]) for joint in ALL_JOINTS}
        if not all(math.isfinite(value) for value in ordered.values()):
            raise ValueError("Motion controller generated a non-finite action.")
        self.previous = ordered
        return ordered

    def hold(self) -> dict[str, float]:
        return dict(self.previous)

    def reset_to(self, positions: Mapping[str, float]) -> None:
        if set(positions) != set(ALL_JOINTS):
            raise ValueError("Reset positions must contain all six joints.")
        self.previous = {joint: float(positions[joint]) for joint in ALL_JOINTS}

