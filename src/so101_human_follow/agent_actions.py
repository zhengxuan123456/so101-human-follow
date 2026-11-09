from __future__ import annotations

import math
import random
import threading
from collections import deque
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .config import ALL_JOINTS, CONTROLLED_JOINTS, FIXED_JOINTS
from .pose_detector import PoseDetection
from .pose_math import HAND_INDEX, POSE_INDEX, Landmark


# Conversational gestures are independent of the tuned human-follow mapping.
# Head gestures start at the measured pose.  Never move to an artificial raised
# pose just to manufacture a large stroke near a joint limit.
PET_TRAVEL_FRACTION = 0.50
HEAD_UP_MAX_TRAVEL = {"elbow_flex.pos": 8.0, "wrist_flex.pos": 10.0}
AGENT_RANGE_MARGIN_FRACTION = 0.05
WRIST_WAVE_MAX_AMPLITUDE = 18.0

# Apparent-size thresholds are only a conservative RGB-camera proximity guard;
# they are not metric distances.  Shoulder size is normalized to frame height
# (x is aspect-corrected), while palm width uses the same coordinate system.
SHOULDER_SLOW_SIZE = 0.42
SHOULDER_STOP_SIZE = 0.62
PALM_SLOW_SIZE = 0.16
PALM_STOP_SIZE = 0.24

ALLOWED_AGENT_ACTIONS = frozenset(
    {
        "perform_greeting",
        "enter_pet_pose",
        "leave_pet_pose",
        "open_gripper",
        "close_gripper",
        "happy_wiggle",
        "wave_wrist",
        "nod",
        "perform_response_gesture",
        "return_to_neutral",
        "return_to_action_start",
        "wait",
    }
)


@dataclass(frozen=True)
class VisualProximity:
    state: str
    reason: str
    shoulder_size: float | None = None
    palm_size: float | None = None


@dataclass(frozen=True)
class AgentKeyframe:
    label: str
    targets: dict[str, float]
    hold_s: float = 0.0
    active_joints: tuple[str, ...] = ()


@dataclass(frozen=True)
class AgentMotionPlan:
    name: str
    keyframes: tuple[AgentKeyframe, ...]
    pet_restore_to_save: dict[str, float] | None = None
    clear_pet_restore_on_complete: bool = False


@dataclass(frozen=True)
class AgentMotionTiming:
    control_hz: float
    steps: dict[str, float]
    min_speeds: dict[str, float]
    max_speeds: dict[str, float]
    speeds: dict[str, float]
    requested_scale: float | None
    applied_scale: float
    estimated_duration_s: float
    reason: str


@dataclass(frozen=True)
class ResponseGestureTemplate:
    name: str
    categories: tuple[str, ...]
    # Fractions of the calibrated working span, not model-provided angles.
    phases: tuple[Mapping[str, float], ...]
    hold_s: float = 0.12


RESPONSE_CATEGORIES = frozenset({"neutral", "happy", "curious", "cute", "encouraging"})
RESPONSE_GESTURES = (
    ResponseGestureTemplate("attentive_lift", ("neutral", "curious"), ({"wrist_flex.pos": 0.12},)),
    ResponseGestureTemplate("gentle_ack", ("neutral", "encouraging"), ({"elbow_flex.pos": 0.12},)),
    ResponseGestureTemplate("thinking_tilt", ("neutral", "curious", "cute"), ({"wrist_roll.pos": 0.14},), 0.22),
    ResponseGestureTemplate(
        "happy_clap", ("happy", "encouraging"),
        ({"wrist_roll.pos": 0.12, "gripper.pos": 0.40},
         {"wrist_roll.pos": -0.12, "gripper.pos": -0.30}),
    ),
    ResponseGestureTemplate(
        "curious_peek", ("curious", "neutral"),
        ({"elbow_flex.pos": 0.12, "wrist_flex.pos": -0.12},),
    ),
    ResponseGestureTemplate(
        "shy_sway", ("cute", "neutral"),
        ({"wrist_flex.pos": 0.10, "wrist_roll.pos": 0.12},
         {"wrist_flex.pos": -0.06, "wrist_roll.pos": -0.12}),
    ),
    ResponseGestureTemplate(
        "cheerful_bounce", ("happy", "encouraging"),
        ({"elbow_flex.pos": -0.12, "wrist_roll.pos": 0.14, "gripper.pos": 0.40},
         {"elbow_flex.pos": 0.08, "wrist_roll.pos": -0.14, "gripper.pos": -0.30}),
    ),
    ResponseGestureTemplate(
        "soft_nuzzle", ("cute", "encouraging"),
        ({"elbow_flex.pos": -0.12, "wrist_flex.pos": -0.12, "gripper.pos": 0.30},),
    ),
    ResponseGestureTemplate(
        "eager_turn", ("curious", "happy"),
        ({"elbow_flex.pos": 0.10, "wrist_flex.pos": 0.10, "wrist_roll.pos": 0.14},),
    ),
    ResponseGestureTemplate(
        "cute_twirl", ("cute", "happy"),
        ({"wrist_flex.pos": 0.10, "wrist_roll.pos": 0.12, "gripper.pos": 0.35},
         {"wrist_flex.pos": -0.05, "wrist_roll.pos": -0.12, "gripper.pos": -0.20}),
    ),
)
_RECENT_RESPONSE_GESTURES: deque[str] = deque(maxlen=2)
_RESPONSE_GESTURE_LOCK = threading.Lock()


def calculate_agent_motion_timing(
    plan: AgentMotionPlan,
    current_positions: Mapping[str, float],
    *,
    max_step_per_cycle: float,
    max_relative_target: float | Mapping[str, float],
    base_control_hz: float,
    speech_duration_s: float | None = None,
    proximity_speed_scale: float = 1.0,
) -> AgentMotionTiming:
    """Fit remaining sequential motion to speech, bounded by local speed caps.

    Speeds are normalized LeRobot position units/second, not servo-rated deg/s.
    Hardware feedback/serial latency can lengthen the actual duration.  If the
    requested pace is outside the supported range, use the allowed maximum;
    never cut short a gesture or widen a position/relative-target limit.
    """
    if not math.isfinite(base_control_hz) or base_control_hz <= 0:
        raise ValueError("Agent control frequency must be positive.")
    if not math.isfinite(max_step_per_cycle) or max_step_per_cycle <= 0:
        raise ValueError("Agent step cap must be positive.")
    if not math.isfinite(proximity_speed_scale) or not 0 < proximity_speed_scale <= 1:
        raise ValueError("Agent proximity speed scale is invalid.")
    if speech_duration_s is not None and (
        isinstance(speech_duration_s, bool)
        or not math.isfinite(speech_duration_s)
        or not 0 <= speech_duration_s <= 120
    ):
        raise ValueError("Agent speech duration must be between 0 and 120 seconds.")

    # A faster Agent cadence is independent of tuned camera-follow parameters.
    # Body steps remain within their existing caps; ID6 uses its separate cap.
    hz = min(30.0, max(base_control_hz, base_control_hz * 2.0))
    step_caps: dict[str, float] = {}
    for joint in CONTROLLED_JOINTS:
        raw_cap = (
            max_relative_target[joint.removesuffix(".pos")]
            if isinstance(max_relative_target, Mapping) else max_relative_target
        )
        cap = float(raw_cap)
        if not math.isfinite(cap) or cap <= 0:
            raise ValueError(f"Invalid relative step cap for {joint}.")
        cap = cap if joint == "gripper.pos" else min(cap, max_step_per_cycle)
        step_caps[joint] = cap * proximity_speed_scale
    max_speeds = {joint: cap * hz for joint, cap in step_caps.items()}
    min_speeds = {joint: speed * 0.30 for joint, speed in max_speeds.items()}
    previous = _validated_positions(current_positions, "Timing start positions")
    movement_s = 0.0
    stationary_s = 0.0
    for frame in plan.keyframes:
        target = _validated_positions(frame.targets, "Timing keyframe")
        active = tuple(frame.active_joints)
        if set(active) - set(CONTROLLED_JOINTS):
            raise ValueError("Timing frame contains an unsupported active joint.")
        movement_s += max(
            (
                abs(target[joint] - previous[joint]) / max_speeds[joint]
                for joint in active
            ),
            default=0.0,
        )
        stationary_s += frame.hold_s + 1.0 / hz
        previous.update({joint: target[joint] for joint in active})
    scale = 1.0
    requested_scale = None
    reason = "no speech duration; allowed maximum"
    if speech_duration_s is not None:
        budget = speech_duration_s - stationary_s
        requested_scale = movement_s / budget if budget > 0 else None
        if requested_scale is not None and 0.30 <= requested_scale <= 1.0:
            scale = requested_scale
            reason = "fitted to remaining speech duration"
        else:
            reason = "requested speed outside allowed range; using allowed maximum"
    return AgentMotionTiming(
        control_hz=hz,
        steps={joint: cap * scale for joint, cap in step_caps.items()},
        min_speeds=min_speeds,
        max_speeds=max_speeds,
        speeds={joint: speed * scale for joint, speed in max_speeds.items()},
        requested_scale=requested_scale,
        applied_scale=scale,
        estimated_duration_s=movement_s / scale + stationary_s,
        reason=reason,
    )


def _screen_distance(a: Landmark, b: Landmark, aspect_ratio: float) -> float:
    return math.hypot((float(a.x) - float(b.x)) * aspect_ratio, float(a.y) - float(b.y))


def estimate_visual_proximity(
    detection: PoseDetection,
    *,
    aspect_ratio: float,
) -> VisualProximity:
    """Return a conservative apparent-size proximity state for an RGB camera.

    This intentionally never claims a centimetre distance.  With one ordinary
    camera, apparent shoulder/palm size is suitable only as an additional stop
    guard.  Missing or unreliable scale data remains UNKNOWN and therefore
    prevents a new Agent motion from starting.
    """

    if not math.isfinite(aspect_ratio) or aspect_ratio <= 0.0:
        return VisualProximity("UNKNOWN", "invalid camera geometry")
    if detection.people_count > 1:
        return VisualProximity("UNKNOWN", "multiple people")

    shoulder_size: float | None = None
    if len(detection.landmarks) > max(
        POSE_INDEX["left_shoulder"], POSE_INDEX["right_shoulder"]
    ):
        left = detection.landmarks[POSE_INDEX["left_shoulder"]]
        right = detection.landmarks[POSE_INDEX["right_shoulder"]]
        confidence = min(
            float(left.visibility),
            float(left.presence),
            float(right.visibility),
            float(right.presence),
        )
        if confidence >= 0.5:
            value = _screen_distance(left, right, aspect_ratio)
            if math.isfinite(value) and value >= 0.03:
                shoulder_size = value

    palm_size: float | None = None
    if len(detection.hand_landmarks) > max(
        HAND_INDEX["index_mcp"], HAND_INDEX["pinky_mcp"]
    ):
        index_mcp = detection.hand_landmarks[HAND_INDEX["index_mcp"]]
        pinky_mcp = detection.hand_landmarks[HAND_INDEX["pinky_mcp"]]
        value = _screen_distance(index_mcp, pinky_mcp, aspect_ratio)
        if math.isfinite(value) and value >= 0.015:
            palm_size = value

    if shoulder_size is None and palm_size is None:
        return VisualProximity("UNKNOWN", "no reliable visual scale")

    too_close = bool(
        (shoulder_size is not None and shoulder_size >= SHOULDER_STOP_SIZE)
        or (palm_size is not None and palm_size >= PALM_STOP_SIZE)
    )
    slow = bool(
        (shoulder_size is not None and shoulder_size >= SHOULDER_SLOW_SIZE)
        or (palm_size is not None and palm_size >= PALM_SLOW_SIZE)
    )
    if too_close:
        state, reason = "TOO_CLOSE", "person or hand appears too close"
    elif slow:
        state, reason = "SLOW", "person or hand is inside the visual slow zone"
    else:
        state, reason = "SAFE", "visual apparent size is inside the safe zone"
    return VisualProximity(state, reason, shoulder_size, palm_size)


def _number(
    arguments: Mapping[str, Any],
    name: str,
    default: float,
    low: float,
    high: float,
) -> float:
    raw = arguments.get(name, default)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError(f"{name} must be a number.")
    value = float(raw)
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{name} must be between {low:g} and {high:g}.")
    return value


def _count(arguments: Mapping[str, Any], default: int = 2) -> int:
    raw = arguments.get("count", default)
    if isinstance(raw, bool) or not isinstance(raw, int) or not 1 <= raw <= 3:
        raise ValueError("count must be an integer from 1 to 3.")
    return raw


def _validated_positions(values: Mapping[str, float], label: str) -> dict[str, float]:
    if set(values) != set(ALL_JOINTS):
        raise ValueError(f"{label} must contain all six joints.")
    result = {joint: float(values[joint]) for joint in ALL_JOINTS}
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError(f"{label} contains a non-finite value.")
    return result


def _clamped_target(
    base: Mapping[str, float],
    updates: Mapping[str, float],
    *,
    limits: Mapping[str, tuple[float, float]],
    fixed_positions: Mapping[str, float],
) -> dict[str, float]:
    target = _validated_positions(base, "Agent target base")
    unknown = set(updates) - set(CONTROLLED_JOINTS)
    if unknown:
        raise ValueError(f"Agent actions cannot command these joints: {sorted(unknown)}")
    for joint, raw in updates.items():
        low, high = limits[joint]
        value = float(raw)
        if not math.isfinite(value):
            raise ValueError(f"Non-finite target for {joint}.")
        target[joint] = max(float(low), min(float(high), value))
    for joint in FIXED_JOINTS:
        target[joint] = float(fixed_positions[joint])
    return {joint: target[joint] for joint in ALL_JOINTS}


def _gripper_goal(
    limits: Mapping[str, tuple[float, float]], *, open_gripper: bool, open_is_high: bool
) -> float:
    low, high = limits["gripper.pos"]
    # Stay slightly inside the captured/absolute endpoints instead of repeatedly
    # driving a mechanical stop during conversational gestures.
    margin = max(0.5, (high - low) * 0.05)
    low_goal = min(high, low + margin)
    high_goal = max(low, high - margin)
    choose_high = open_gripper == open_is_high
    return high_goal if choose_high else low_goal


def _restore_updates(
    positions: Mapping[str, float],
    joints: Sequence[str],
    limits: Mapping[str, tuple[float, float]],
) -> dict[str, float]:
    """Never silently clamp an exact-return target into a different pose."""
    result = {joint: float(positions[joint]) for joint in joints}
    for joint, value in result.items():
        low, high = limits[joint]
        if not math.isfinite(value) or not float(low) <= value <= float(high):
            raise ValueError(
                f"Cannot return {joint} to its saved position {value:.2f}: "
                f"outside the configured range [{low:.2f}, {high:.2f}]."
            )
    return result


def _inner_limits(joint: str, limits: Mapping[str, tuple[float, float]]) -> tuple[float, float]:
    low, high = map(float, limits[joint])
    margin = (high - low) * AGENT_RANGE_MARGIN_FRACTION
    return low + margin, high - margin


def _pet_pair(
    reference: Mapping[str, float],
    limits: Mapping[str, tuple[float, float]],
    *,
    travel_scale: float = 1.0,
) -> tuple[dict[str, float], dict[str, float]]:
    """Make a bounded downward stroke and an exact measured return pose.

    Limited remaining travel means a smaller stroke, never a preparatory motion
    in the opposite direction.  Calibration and visual-follow mapping stay as is.
    """
    joints = ("elbow_flex.pos", "wrist_flex.pos")
    up: dict[str, float] = {}
    down: dict[str, float] = {}
    for joint in joints:
        low, high = limits[joint]
        travel = min(
            (float(high) - float(low)) * PET_TRAVEL_FRACTION * travel_scale,
            float(high) - float(low),
        )
        up[joint] = float(reference[joint])
        if not float(low) <= up[joint] <= float(high):
            raise ValueError(f"Head gesture starts outside the configured range: {joint}.")
        inner_low, _ = _inner_limits(joint, limits)
        # At/below the inner margin, keep this axis still instead of reversing
        # direction just to force it back inside the preferred gesture envelope.
        down[joint] = min(up[joint], max(inner_low, up[joint] - travel))
    return up, down


def _response_template(
    category: str,
    original: Mapping[str, float],
    limits: Mapping[str, tuple[float, float]],
) -> ResponseGestureTemplate:
    if category not in RESPONSE_CATEGORIES:
        raise ValueError(f"Response category must be one of {sorted(RESPONSE_CATEGORIES)}.")
    candidates: list[ResponseGestureTemplate] = []
    for template in RESPONSE_GESTURES:
        if category not in template.categories:
            continue
        touched = set().union(*(phase.keys() for phase in template.phases))
        try:
            _restore_updates(original, tuple(touched), limits)
        except ValueError:
            continue
        candidates.append(template)
    if not candidates:
        raise ValueError("No response gesture can safely return to the current measured pose.")
    with _RESPONSE_GESTURE_LOCK:
        choices = [item for item in candidates if item.name not in _RECENT_RESPONSE_GESTURES]
        if not choices:
            choices = [
                item for item in candidates
                if not _RECENT_RESPONSE_GESTURES or item.name != _RECENT_RESPONSE_GESTURES[-1]
            ] or candidates
        selected = random.choice(choices)
        _RECENT_RESPONSE_GESTURES.append(selected.name)
    return selected


def build_agent_motion_plan(
    actions: Sequence[Mapping[str, Any]],
    *,
    current_positions: Mapping[str, float],
    home_positions: Mapping[str, float],
    limits: Mapping[str, tuple[float, float]],
    fixed_positions: Mapping[str, float],
    gripper_open_is_high: bool,
    pet_restore_positions: Mapping[str, float] | None,
    recovery_joints: Sequence[str] | None = None,
) -> AgentMotionPlan:
    """Validate semantic actions and expand them into safe joint-space keyframes."""

    if not isinstance(actions, Sequence) or isinstance(actions, (str, bytes)):
        raise ValueError("actions must be an array.")
    if not 1 <= len(actions) <= 12:
        raise ValueError("An Agent motion plan must contain 1 to 12 actions.")
    if set(limits) != set(CONTROLLED_JOINTS):
        raise ValueError("Agent limits must contain exactly ID3-ID6.")
    if set(fixed_positions) != set(FIXED_JOINTS):
        raise ValueError("Agent fixed positions must contain exactly ID1-ID2.")
    for joint, (low, high) in limits.items():
        if not math.isfinite(float(low)) or not math.isfinite(float(high)) or float(low) >= float(high):
            raise ValueError(f"Invalid working limits for {joint}.")

    current = _validated_positions(current_positions, "Current positions")
    # ``home_positions`` is the robot pose captured at connection time.  It is
    # deliberately independent of ControlEngine._neutral, which is a human-pose
    # feature sample used only by the live visual-following controller.
    home = _validated_positions(home_positions, "Robot connection/home positions")
    # Retain the argument for compatibility.  Lowering and raising remain
    # independently callable and do not require a prior pet interaction.
    if pet_restore_positions is not None:
        _validated_positions(pet_restore_positions, "Legacy pet-pose positions")
    save_restore: dict[str, float] | None = None
    clear_restore = False
    frames: list[AgentKeyframe] = []
    action_names: list[str] = []

    def add(label: str, updates: Mapping[str, float], hold_s: float = 0.08) -> None:
        nonlocal current
        current = _clamped_target(
            current,
            updates,
            limits=limits,
            fixed_positions=fixed_positions,
        )
        active = tuple(joint for joint in CONTROLLED_JOINTS if joint in updates)
        frames.append(AgentKeyframe(label, dict(current), max(0.0, float(hold_s)), active))

    for item in actions:
        if not isinstance(item, Mapping):
            raise ValueError("Every Agent action must be an object.")
        name = item.get("name")
        arguments = item.get("arguments", {})
        if not isinstance(name, str) or name not in ALLOWED_AGENT_ACTIONS:
            raise ValueError(f"Unsupported Agent action: {name!r}")
        if not isinstance(arguments, Mapping):
            raise ValueError(f"arguments for {name} must be an object.")
        action_names.append(name)

        if name == "perform_greeting":
            closed = _gripper_goal(
                limits, open_gripper=False, open_is_high=gripper_open_is_high
            )
            opened = _gripper_goal(
                limits, open_gripper=True, open_is_high=gripper_open_is_high
            )
            original = dict(current)
            count = _count(arguments, default=2)
            for index in range(count):
                add(f"greeting close {index + 1}/{count}", {"gripper.pos": closed}, 0.06)
                add(f"greeting open {index + 1}/{count}", {"gripper.pos": opened}, 0.06)
            add("greeting restore", {"gripper.pos": original["gripper.pos"]}, 0.08)
        elif name == "open_gripper":
            add(
                "open gripper",
                {
                    "gripper.pos": _gripper_goal(
                        limits, open_gripper=True, open_is_high=gripper_open_is_high
                    )
                },
            )
        elif name == "close_gripper":
            add(
                "close gripper",
                {
                    "gripper.pos": _gripper_goal(
                        limits, open_gripper=False, open_is_high=gripper_open_is_high
                    )
                },
            )
        elif name in {"wave_wrist", "happy_wiggle"}:
            count = _count(arguments, default=3 if name == "wave_wrist" else 2)
            original = dict(current)
            return_targets = _restore_updates(original, ("wrist_roll.pos",), limits)
            low, high = limits["wrist_roll.pos"]
            amplitude_scale = _number(arguments, "amplitude", 1.0, 0.25, 1.0)
            amplitude = min(WRIST_WAVE_MAX_AMPLITUDE, (high - low) * 0.18) * amplitude_scale
            for index in range(count):
                add(
                    f"{name} left {index + 1}/{count}",
                    {"wrist_roll.pos": original["wrist_roll.pos"] - amplitude},
                )
                add(
                    f"{name} right {index + 1}/{count}",
                    {"wrist_roll.pos": original["wrist_roll.pos"] + amplitude},
                )
            add(f"{name} restore", return_targets)
        elif name == "nod":
            count = _count(arguments, default=2)
            intensity = arguments.get("intensity", "normal")
            if intensity not in {"small", "normal"}:
                raise ValueError("Nod intensity must be small or normal.")
            original = dict(current)
            return_targets = _restore_updates(
                original, ("elbow_flex.pos", "wrist_flex.pos"), limits
            )
            up, down = _pet_pair(
                original, limits, travel_scale=0.55 if intensity == "small" else 1.0
            )
            for index in range(count):
                add(f"nod down {index + 1}/{count}", down, 0.08)
                add(f"nod up {index + 1}/{count}", up, 0.08)
            add("nod restore measured start", return_targets, 0.10)
        elif name == "perform_response_gesture":
            category = arguments.get("category", "neutral")
            if not isinstance(category, str):
                raise ValueError("Response category must be a string.")
            original = dict(current)
            template = _response_template(category, original, limits)
            touched = tuple(
                joint for joint in CONTROLLED_JOINTS
                if any(joint in phase for phase in template.phases)
            )
            return_targets = _restore_updates(original, touched, limits)
            orientations: dict[str, float] = {}
            for index, phase in enumerate(template.phases):
                updates: dict[str, float] = {}
                for joint, fraction in phase.items():
                    low, high = limits[joint]
                    inner_low, inner_high = _inner_limits(joint, limits)
                    offset = (float(high) - float(low)) * fraction
                    start = original[joint]
                    if joint not in orientations:
                        proposed = max(inner_low, min(inner_high, start + offset))
                        reverse = max(inner_low, min(inner_high, start - offset))
                        orientation = (
                            -1.0 if abs(proposed - start) < abs(offset) * 0.25
                            and abs(reverse - start) > abs(proposed - start) else 1.0
                        )
                        orientations[joint] = orientation
                    updates[joint] = max(
                        inner_low, min(inner_high, start + offset * orientations[joint])
                    )
                add(
                    f"response {template.name} phase {index + 1}/{len(template.phases)}",
                    updates, template.hold_s,
                )
            add(f"response {template.name} restore measured start", return_targets, 0.10)
            action_names[-1] = f"perform_response_gesture:{template.name}"
        elif name == "enter_pet_pose":
            original = dict(current)
            return_targets, down = _pet_pair(original, limits)
            add("lower head directly and hold two seconds", down, 2.0)
            add("pet restore measured start", return_targets, 0.10)
            clear_restore = True
        elif name == "leave_pet_pose":
            up = {}
            for joint, max_travel in HEAD_UP_MAX_TRAVEL.items():
                low, high = map(float, limits[joint])
                if not low <= current[joint] <= high:
                    raise ValueError(f"Head gesture starts outside the configured range: {joint}.")
                up[joint] = min(high, current[joint] + min(max_travel, (high - low) * 0.08))
            add("raise head a small step from measured pose", up, 0.15)
            clear_restore = True
        elif name == "return_to_action_start":
            joints = tuple(CONTROLLED_JOINTS if recovery_joints is None else recovery_joints)
            if not set(joints) <= set(CONTROLLED_JOINTS):
                raise ValueError("Recovery may only move previously controlled joints.")
            add(
                "return to previous measured action start",
                _restore_updates(home, joints, limits),
                0.15,
            )
            clear_restore = True
        elif name == "return_to_neutral":
            add(
                "return to neutral",
                {joint: home[joint] for joint in CONTROLLED_JOINTS},
                0.15,
            )
            clear_restore = True
        elif name == "wait":
            seconds = _number(arguments, "seconds", 1.0, 0.0, 10.0)
            add("wait", {}, seconds)

    if not frames:
        raise ValueError("Agent plan did not produce any motion keyframes.")
    return AgentMotionPlan(
        name=" + ".join(action_names),
        keyframes=tuple(frames),
        pet_restore_to_save=save_restore,
        clear_pet_restore_on_complete=clear_restore,
    )


def step_toward_agent_target(
    current_positions: Mapping[str, float],
    target_positions: Mapping[str, float],
    *,
    fixed_positions: Mapping[str, float],
    max_step: float | Mapping[str, float],
    speed_scale: float,
) -> tuple[dict[str, float], bool]:
    current = _validated_positions(current_positions, "Current Agent positions")
    target = _validated_positions(target_positions, "Agent keyframe target")
    steps = (
        {joint: float(max_step[joint]) for joint in CONTROLLED_JOINTS}
        if isinstance(max_step, Mapping)
        else {joint: float(max_step) for joint in CONTROLLED_JOINTS}
    )
    if not all(math.isfinite(value) and value > 0 for value in steps.values()):
        raise ValueError("Agent max_step must be positive and finite for every controlled joint.")
    if not math.isfinite(speed_scale) or not 0.0 < speed_scale <= 1.0:
        raise ValueError("Agent speed_scale must be in (0, 1].")

    result: dict[str, float] = {}
    reached = True
    for joint in ALL_JOINTS:
        if joint in FIXED_JOINTS:
            result[joint] = float(fixed_positions[joint])
            continue
        step = steps[joint] * speed_scale
        delta = target[joint] - current[joint]
        if abs(delta) > step:
            reached = False
            delta = math.copysign(step, delta)
        result[joint] = current[joint] + delta
        if abs(target[joint] - result[joint]) > 1e-6:
            reached = False
    return {joint: result[joint] for joint in ALL_JOINTS}, reached
