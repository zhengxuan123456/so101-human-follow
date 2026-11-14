from __future__ import annotations

import math
import random
import threading
from collections import deque
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .config import ALL_JOINTS, CONTROLLED_JOINTS, FIXED_JOINTS
from .motion_profile import COSINE_PEAK_SPEED_FACTOR
from .pose_detector import PoseDetection
from .pose_math import HAND_INDEX, POSE_INDEX, Landmark
from .teach_trajectory import FollowerTrajectory, follower_trajectory


# Agent/manual choreography may use six axes.  Camera following keeps its
# original CONTROLLED_JOINTS/FIXED_JOINTS configuration in config.py.
AGENT_JOINTS = frozenset(ALL_JOINTS)
MAX_AGENT_KEYFRAMES = 512
MAX_LOCAL_AGENT_ACTIONS = MAX_AGENT_KEYFRAMES
MAX_EXPLICIT_PLAN_SECONDS = 180.0
# Up to 32 editor groups, each expanded into six ordered single-axis frames.
MAX_SAVED_ACTION_FRAMES = 32 * len(ALL_JOINTS) * 2


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
        "return_to_plan_start",
        "wait",
        "move_joints",
        "play_saved_action",
        "play_follower_trajectory",
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
    targets: Mapping[str, float]
    hold_s: float = 0.0
    active_joints: tuple[str, ...] = ()
    action_index: int = 0
    completed_cycles: int = 0
    duration_s: float | None = None
    joint_timing: Mapping[str, Mapping[str, float]] | None = None
    trajectory: FollowerTrajectory | None = None

    def __post_init__(self) -> None:
        targets = _validated_updates(self.targets)
        if set(self.active_joints) - targets.keys():
            raise ValueError("Every active joint must have a target.")
        if len(set(self.active_joints)) != len(self.active_joints):
            raise ValueError("An active joint may only appear once.")
        if (isinstance(self.hold_s, bool) or not isinstance(self.hold_s, (int, float))
                or not math.isfinite(self.hold_s) or not 0 <= self.hold_s <= 10):
            raise ValueError("Keyframe hold must be between 0 and 10 seconds.")
        if self.duration_s is not None and (
            isinstance(self.duration_s, bool) or not isinstance(self.duration_s, (int, float))
            or not math.isfinite(self.duration_s) or not 0.05 <= self.duration_s <= 10
        ):
            raise ValueError("Keyframe duration must be between 0.05 and 10 seconds.")
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0
               for value in (self.action_index, self.completed_cycles)):
            raise ValueError("Keyframe progress indexes must be non-negative integers.")
        object.__setattr__(self, "targets", MappingProxyType(targets))
        object.__setattr__(self, "active_joints", tuple(self.active_joints))
        if self.joint_timing is not None:
            if (not isinstance(self.joint_timing, Mapping)
                    or not self.active_joints or set(self.joint_timing) != set(self.active_joints)):
                raise ValueError("Joint timing must specify each active joint exactly once.")
            timings = {}
            for joint, timing in self.joint_timing.items():
                if not isinstance(timing, Mapping) or set(timing) != {"duration_s", "hold_s"}:
                    raise ValueError("Joint timing requires motion and hold times.")
                timings[joint] = MappingProxyType({
                    "duration_s": _number(timing, "duration_s", 0.1, 0.1, 10.0),
                    "hold_s": _number(timing, "hold_s", 0.0, 0.0, 10.0),
                })
            duration, hold = _parallel_group_times({
                joint: (timing["duration_s"], timing["hold_s"])
                for joint, timing in timings.items()
            })
            object.__setattr__(self, "joint_timing", MappingProxyType(timings))
            object.__setattr__(self, "duration_s", duration)
            object.__setattr__(self, "hold_s", hold)


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
    frame_motion_s: tuple[float, ...] = ()
    frame_durations_s: tuple[float, ...] = ()
    effective_speeds: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class ResponseGestureTemplate:
    name: str
    categories: tuple[str, ...]
    # Fractions of the calibrated working span, not model-provided angles.
    phases: tuple[Mapping[str, float], ...]
    hold_s: float = 0.12


RESPONSE_CATEGORIES = frozenset({"neutral", "happy", "curious", "cute", "encouraging"})
RESPONSE_GESTURES = (
    ResponseGestureTemplate("attentive_lift", ("neutral", "curious"),
                            ({"wrist_flex.pos": 0.12, "elbow_flex.pos": 0.06,
                              "wrist_roll.pos": 0.06, "shoulder_pan.pos": 0.03},)),
    ResponseGestureTemplate("gentle_ack", ("neutral", "encouraging"),
                            ({"elbow_flex.pos": 0.12, "wrist_flex.pos": -0.05,
                              "wrist_roll.pos": 0.04, "shoulder_lift.pos": 0.03},)),
    ResponseGestureTemplate("thinking_tilt", ("neutral", "curious", "cute"),
                            ({"wrist_roll.pos": 0.14, "wrist_flex.pos": 0.05,
                              "elbow_flex.pos": 0.06},), 0.22),
    ResponseGestureTemplate(
        "happy_clap", ("happy", "encouraging"),
        ({"wrist_roll.pos": 0.12, "gripper.pos": 0.40,
          "elbow_flex.pos": -0.08, "shoulder_pan.pos": 0.03},
         {"wrist_roll.pos": -0.12, "gripper.pos": -0.30,
          "elbow_flex.pos": 0.04, "shoulder_pan.pos": -0.03}),
    ),
    ResponseGestureTemplate(
        "curious_peek", ("curious", "neutral"),
        ({"elbow_flex.pos": 0.12, "wrist_flex.pos": -0.12,
          "wrist_roll.pos": 0.08, "shoulder_lift.pos": 0.03},),
    ),
    ResponseGestureTemplate(
        "shy_sway", ("cute", "neutral"),
        ({"wrist_flex.pos": 0.10, "wrist_roll.pos": 0.12,
          "elbow_flex.pos": -0.04, "shoulder_pan.pos": 0.04},
         {"wrist_flex.pos": -0.06, "wrist_roll.pos": -0.12,
          "elbow_flex.pos": 0.04, "shoulder_pan.pos": -0.04}),
    ),
    ResponseGestureTemplate(
        "cheerful_bounce", ("happy", "encouraging"),
        ({"elbow_flex.pos": -0.12, "wrist_roll.pos": 0.14, "gripper.pos": 0.40,
          "wrist_flex.pos": -0.04, "shoulder_lift.pos": 0.03},
         {"elbow_flex.pos": 0.08, "wrist_roll.pos": -0.14, "gripper.pos": -0.30,
          "wrist_flex.pos": 0.04, "shoulder_lift.pos": -0.03}),
    ),
    ResponseGestureTemplate(
        "soft_nuzzle", ("cute", "encouraging"),
        ({"elbow_flex.pos": -0.12, "wrist_flex.pos": -0.12,
          "gripper.pos": 0.30, "shoulder_pan.pos": 0.03},),
    ),
    ResponseGestureTemplate(
        "eager_turn", ("curious", "happy"),
        ({"elbow_flex.pos": 0.10, "wrist_flex.pos": 0.10, "wrist_roll.pos": 0.14,
          "shoulder_pan.pos": 0.04, "shoulder_lift.pos": 0.025},),
    ),
    ResponseGestureTemplate(
        "cute_twirl", ("cute", "happy"),
        ({"wrist_flex.pos": 0.10, "wrist_roll.pos": 0.12, "gripper.pos": 0.35,
          "elbow_flex.pos": -0.04, "shoulder_pan.pos": 0.035},
         {"wrist_flex.pos": -0.05, "wrist_roll.pos": -0.12, "gripper.pos": -0.20,
          "elbow_flex.pos": 0.04, "shoulder_pan.pos": -0.035}),
    ),
)
_RECENT_RESPONSE_GESTURES: deque[str] = deque(maxlen=2)
_RESPONSE_GESTURE_LOCK = threading.Lock()


def _parallel_group_times(schedule: Mapping[str, tuple[float, float]]) -> tuple[float, float]:
    duration = max(move for move, _ in schedule.values())
    hold = max(hold - (duration - move) for move, hold in schedule.values())
    return duration, hold


def agent_frame_schedule(
    frame: AgentKeyframe,
    start: Mapping[str, float],
    speed_limits: Mapping[str, float],
    period_s: float,
) -> tuple[float, float, dict[str, tuple[float, float]]]:
    """Keep the cosine peak within all speed limits and sample short ramps."""
    minimum_moves = {}
    for joint in frame.active_joints:
        travel = abs(frame.targets[joint] - start[joint]) / speed_limits[joint]
        # Three intervals give a nonzero move two intermediate position commands.
        minimum_moves[joint] = max(3 * period_s, travel * COSINE_PEAK_SPEED_FACTOR) if travel else 0.0
    if frame.joint_timing is None:
        return max(frame.duration_s or 0.0, max(minimum_moves.values(), default=0.0)), frame.hold_s, {}
    schedule = {
        joint: (max(minimum_moves[joint], timing["duration_s"]), timing["hold_s"])
        for joint, timing in frame.joint_timing.items()
    }
    duration, hold = _parallel_group_times(schedule)
    return duration, hold, schedule


def agent_frame_motion_duration(
    frame: AgentKeyframe,
    start: Mapping[str, float],
    speed_limits: Mapping[str, float],
    period_s: float,
) -> float:
    return agent_frame_schedule(frame, start, speed_limits, period_s)[0]


def calculate_agent_motion_timing(
    plan: AgentMotionPlan,
    current_positions: Mapping[str, float],
    *,
    max_step_per_cycle: float,
    max_relative_target: float | Mapping[str, float],
    base_control_hz: float,
    speech_duration_s: float | None = None,
    proximity_speed_scale: float = 1.0,
    measured_speeds: Mapping[str, float] | None = None,
    cycle_time_s: float | None = None,
) -> AgentMotionTiming:
    """Estimate sequential motion without slowing it to fill speech playback.

    Speeds are normalized LeRobot position units/second, not servo-rated deg/s.
    The command-rate limit is not a measured servo speed.  Until qualified
    feedback is available, use a conservative starting estimate; later calls
    can supply measured speeds and serial-cycle time.  Speech length is retained
    for diagnostics only: commands retain their existing maximum step caps.
    """
    if not math.isfinite(base_control_hz) or base_control_hz <= 0:
        raise ValueError("Agent control frequency must be positive.")
    if not math.isfinite(max_step_per_cycle) or max_step_per_cycle <= 0:
        raise ValueError("Agent step cap must be positive.")
    if not math.isfinite(proximity_speed_scale) or not 0 < proximity_speed_scale <= 1:
        raise ValueError("Agent proximity speed scale is invalid.")
    if cycle_time_s is not None and (
        isinstance(cycle_time_s, bool)
        or not math.isfinite(cycle_time_s)
        or cycle_time_s <= 0
    ):
        raise ValueError("Agent serial-cycle time must be positive.")
    if speech_duration_s is not None and (
        isinstance(speech_duration_s, bool)
        or not math.isfinite(speech_duration_s)
        or not 0 <= speech_duration_s <= 120
    ):
        raise ValueError("Agent speech duration must be between 0 and 120 seconds.")

    # A faster Agent cadence is independent of tuned camera-follow parameters.
    # Body steps remain within their existing caps; ID6 uses its separate cap.
    hz = min(30.0, max(base_control_hz, base_control_hz * 2.0))
    cycle_s = max(1.0 / hz, cycle_time_s or 0.0)
    step_caps: dict[str, float] = {}
    for joint in ALL_JOINTS:
        if isinstance(max_relative_target, Mapping) and joint.removesuffix(".pos") not in max_relative_target:
            continue
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
    # These are software pacing rates, not motor-rated minimum speeds.
    min_speeds = {joint: 0.0 for joint in step_caps}
    effective_speeds = {}
    for joint, cap in step_caps.items():
        attainable = min(max_speeds[joint], cap / cycle_s)
        measured = None if measured_speeds is None else measured_speeds.get(joint)
        if measured is not None and (
            isinstance(measured, bool) or not math.isfinite(measured) or measured <= 0
        ):
            raise ValueError(f"Invalid measured Agent speed for {joint}.")
        effective_speeds[joint] = (
            min(attainable, measured) if measured is not None else attainable * 0.30
        )
    previous = _validated_positions(current_positions, "Timing start positions")
    travel_times = []
    stationary_times = []
    for frame in plan.keyframes:
        target = _validated_positions(frame.targets, "Timing keyframe")
        active = tuple(frame.active_joints)
        if set(active) - step_caps.keys():
            raise ValueError("Timing frame contains an unsupported active joint.")
        if not math.isfinite(frame.hold_s) or frame.hold_s < 0:
            raise ValueError("Timing frame hold must be non-negative.")
        if frame.trajectory is not None:
            motion, hold = frame.trajectory.duration_s, frame.hold_s
        else:
            motion, hold, _ = agent_frame_schedule(frame, previous, effective_speeds, cycle_s)
        travel_times.append(motion)
        stationary_times.append(hold + cycle_s)
        previous.update({joint: target[joint] for joint in active})
    movement_s = sum(travel_times)
    stationary_s = sum(stationary_times)
    scale = 1.0
    requested_scale = None
    reason = "feedback-aware estimate; no speech duration"
    if speech_duration_s is not None:
        budget = speech_duration_s - stationary_s
        requested_scale = movement_s / budget if budget > 0 else None
        reason = "maximum configured command rate; speech does not stretch motion"
    frame_motion_s = tuple(travel / scale for travel in travel_times)
    frame_durations_s = tuple(
        travel + stationary for travel, stationary in zip(frame_motion_s, stationary_times)
    )
    return AgentMotionTiming(
        control_hz=hz,
        steps={joint: cap * scale for joint, cap in step_caps.items()},
        min_speeds=min_speeds,
        max_speeds=max_speeds,
        speeds={joint: speed * scale for joint, speed in max_speeds.items()},
        requested_scale=requested_scale,
        applied_scale=scale,
        estimated_duration_s=sum(frame_durations_s),
        reason=reason,
        frame_motion_s=frame_motion_s,
        frame_durations_s=frame_durations_s,
        effective_speeds=effective_speeds,
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
    if not isinstance(values, Mapping) or set(values) != set(ALL_JOINTS):
        raise ValueError(f"{label} must contain all six joints.")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values.values()):
        raise ValueError(f"{label} must contain numeric position values.")
    result = {joint: float(values[joint]) for joint in ALL_JOINTS}
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError(f"{label} contains a non-finite value.")
    return result


def _validated_updates(values: Mapping[str, float]) -> dict[str, float]:
    if not isinstance(values, Mapping) or set(values) - AGENT_JOINTS:
        raise ValueError("Joint targets must use known joint names with the .pos suffix.")
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) for value in values.values()):
        raise ValueError("Joint targets must be finite numeric positions.")
    return {joint: float(value) for joint, value in values.items()}


def _validate_agent_envelope(
    limits: Mapping[str, tuple[float, float]], fixed_positions: Mapping[str, float],
) -> None:
    if set(limits) not in (set(CONTROLLED_JOINTS), set(AGENT_JOINTS)):
        raise ValueError("Agent limits must contain either legacy ID3-ID6 or all six joints.")
    fixed = _validated_updates(fixed_positions)
    if set(limits) == set(CONTROLLED_JOINTS) and set(fixed) != set(FIXED_JOINTS):
        raise ValueError("Legacy four-axis limits require fixed ID1-ID2 positions.")
    for joint, bounds in limits.items():
        if len(bounds) != 2 or any(isinstance(value, bool) or not isinstance(value, (int, float))
                                   or not math.isfinite(value) for value in bounds):
            raise ValueError(f"Invalid working limits for {joint}.")
        low, high = map(float, bounds)
        if low >= high:
            raise ValueError(f"Invalid working limits for {joint}.")
        if joint in fixed and not low <= fixed[joint] <= high:
            raise ValueError(f"Fixed position for {joint} is outside its working limits.")


def move_joint(
    joint: int | str, position: float, *, hold_s: float = 0.0,
    duration_s: float | None = None, label: str | None = None,
) -> AgentKeyframe:
    """Create one reusable local joint primitive in normalized LeRobot units.

    Physical degrees must first be converted using this robot's calibration;
    this API deliberately does not label normalized positions as angles.
    """
    if isinstance(joint, bool):
        raise ValueError("Joint id must be an integer from 1 to 6 or a joint name.")
    if isinstance(joint, int):
        if not 1 <= joint <= 6:
            raise ValueError("Joint id must be between 1 and 6.")
        joint = ALL_JOINTS[joint - 1]
    if not isinstance(joint, str) or joint not in AGENT_JOINTS:
        raise ValueError("Unknown joint.")
    return AgentKeyframe(
        label or f"move {joint}", {joint: position}, hold_s, (joint,),
        duration_s=duration_s,
    )


def move_id1(position: float, **options: Any) -> AgentKeyframe:
    return move_joint(1, position, **options)


def move_id2(position: float, **options: Any) -> AgentKeyframe:
    return move_joint(2, position, **options)


def move_id3(position: float, **options: Any) -> AgentKeyframe:
    return move_joint(3, position, **options)


def move_id4(position: float, **options: Any) -> AgentKeyframe:
    return move_joint(4, position, **options)


def move_id5(position: float, **options: Any) -> AgentKeyframe:
    return move_joint(5, position, **options)


def move_id6(position: float, **options: Any) -> AgentKeyframe:
    return move_joint(6, position, **options)


def parallel(
    *frames: AgentKeyframe, label: str = "parallel", hold_s: float | None = None,
    duration_s: float | None = None,
) -> AgentKeyframe:
    """Merge simultaneous disjoint joint primitives into one keyframe."""
    if not frames:
        raise ValueError("parallel requires at least one joint primitive.")
    targets: dict[str, float] = {}
    for frame in frames:
        if not isinstance(frame, AgentKeyframe):
            raise ValueError("parallel accepts joint primitives only.")
        if set(frame.active_joints) & targets.keys():
            raise ValueError("parallel cannot command one joint more than once.")
        targets.update({joint: frame.targets[joint] for joint in frame.active_joints})
    durations = [frame.duration_s for frame in frames if frame.duration_s is not None]
    return AgentKeyframe(
        label, targets, max(frame.hold_s for frame in frames) if hold_s is None else hold_s,
        tuple(joint for joint in ALL_JOINTS if joint in targets),
        duration_s=duration_s if duration_s is not None else max(durations, default=None),
    )


def sequence(
    *frames: AgentKeyframe, current_positions: Mapping[str, float],
    limits: Mapping[str, tuple[float, float]],
    fixed_positions: Mapping[str, float] | None = None,
    return_to_start: bool = False, name: str = "custom joint sequence",
) -> AgentMotionPlan:
    """Compile local primitives into immutable full-position keyframes.

    Every inactive joint retains its preceding target.  The serial owner holds
    inactive axes from encoder readings while executing each frame.
    """
    fixed = {} if fixed_positions is None else fixed_positions
    _validate_agent_envelope(limits, fixed)
    current = _validated_positions(current_positions, "Sequence start")
    original = dict(current)
    if not frames or len(frames) > MAX_SAVED_ACTION_FRAMES:
        raise ValueError(f"A sequence must contain 1 to {MAX_SAVED_ACTION_FRAMES} expanded keyframes.")
    if not isinstance(return_to_start, bool):
        raise ValueError("return_to_start must be a boolean.")
    expanded: list[AgentKeyframe] = []
    touched: set[str] = set()
    for index, frame in enumerate(frames):
        if not isinstance(frame, AgentKeyframe):
            raise ValueError("sequence accepts joint primitives only.")
        updates = {joint: frame.targets[joint] for joint in frame.active_joints}
        _restore_updates(updates, tuple(updates), limits)
        current = _clamped_target(current, updates, limits=limits, fixed_positions=fixed)
        touched.update(frame.active_joints)
        expanded.append(AgentKeyframe(
            frame.label, current, frame.hold_s, frame.active_joints,
            index, frame.completed_cycles, frame.duration_s, frame.joint_timing,
        ))
    if return_to_start and touched:
        updates = _restore_updates(original, tuple(touched), limits)
        current = _clamped_target(current, updates, limits=limits, fixed_positions=fixed)
        expanded.append(AgentKeyframe(
            "return sequence to measured start", current, 0.0,
            tuple(joint for joint in ALL_JOINTS if joint in touched), len(frames) - 1,
        ))
    if sum(frame.hold_s + (frame.duration_s or 0.0) for frame in expanded) > MAX_EXPLICIT_PLAN_SECONDS:
        raise ValueError(f"Explicit motion and hold times may total at most {MAX_EXPLICIT_PLAN_SECONDS:g} seconds.")
    return AgentMotionPlan(name, tuple(expanded))


def _clamped_target(
    base: Mapping[str, float],
    updates: Mapping[str, float],
    *,
    limits: Mapping[str, tuple[float, float]],
    fixed_positions: Mapping[str, float],
) -> dict[str, float]:
    target = _validated_positions(base, "Agent target base")
    updates = _validated_updates(updates)
    unknown = set(updates) - set(limits)
    if unknown:
        raise ValueError(f"Agent actions cannot command these joints: {sorted(unknown)}")
    if set(updates) & set(fixed_positions):
        raise ValueError("A fixed joint cannot also be an active joint.")
    for joint, raw in updates.items():
        low, high = limits[joint]
        value = float(raw)
        if not math.isfinite(value):
            raise ValueError(f"Non-finite target for {joint}.")
        target[joint] = max(float(low), min(float(high), value))
    target.update(_validated_updates(fixed_positions))
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
    if set(joints) - set(limits):
        raise ValueError("A requested joint has no configured working limits.")
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
    available_joints: Sequence[str] | None = None,
) -> ResponseGestureTemplate:
    if category not in RESPONSE_CATEGORIES:
        raise ValueError(f"Response category must be one of {sorted(RESPONSE_CATEGORIES)}.")
    candidates: list[ResponseGestureTemplate] = []
    for template in RESPONSE_GESTURES:
        if category not in template.categories:
            continue
        available = set(limits) if available_joints is None else set(available_joints)
        touched = set().union(*(phase.keys() for phase in template.phases)) & available
        if not touched:
            continue
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
    if not 1 <= len(actions) <= MAX_LOCAL_AGENT_ACTIONS:
        raise ValueError(f"A local Agent plan must contain 1 to {MAX_LOCAL_AGENT_ACTIONS} actions.")
    _validate_agent_envelope(limits, fixed_positions)
    available = set(limits) - set(fixed_positions)

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
    action_starts: list[dict[str, float]] = []

    def add(
        label: str, updates: Mapping[str, float], hold_s: float = 0.08,
        *, completed_cycles: int = 0, duration_s: float | None = None,
        joint_timing: Mapping[str, Mapping[str, float]] | None = None,
    ) -> None:
        nonlocal current
        current = _clamped_target(
            current,
            updates,
            limits=limits,
            fixed_positions=fixed_positions,
        )
        active = tuple(joint for joint in ALL_JOINTS if joint in updates)
        frames.append(AgentKeyframe(
            label, current, hold_s, active, action_index, completed_cycles, duration_s, joint_timing,
        ))
        if len(frames) > MAX_AGENT_KEYFRAMES:
            raise ValueError(f"A local Agent plan may contain at most {MAX_AGENT_KEYFRAMES} keyframes.")

    def offsets(reference: Mapping[str, float], fractions: Mapping[str, float]) -> dict[str, float]:
        """Small, locally defined companions stay within their intended stroke."""
        result = {}
        for joint, fraction in fractions.items():
            if joint not in available:
                continue
            low, high = map(float, limits[joint])
            start = float(reference[joint])
            if not low <= start <= high:
                raise ValueError(f"Gesture starts outside the configured range: {joint}.")
            offset = (high - low) * fraction
            proposed = max(low, min(high, start + offset))
            reverse = max(low, min(high, start - offset))
            if abs(proposed - start) < abs(offset) * 0.25 and abs(reverse - start) > abs(proposed - start):
                proposed = reverse
            result[joint] = proposed
        return result

    for action_index, item in enumerate(actions):
        if not isinstance(item, Mapping):
            raise ValueError("Every Agent action must be an object.")
        name = item.get("name")
        arguments = item.get("arguments", {})
        if not isinstance(name, str) or name not in ALLOWED_AGENT_ACTIONS:
            raise ValueError(f"Unsupported Agent action: {name!r}")
        if not isinstance(arguments, Mapping):
            raise ValueError(f"arguments for {name} must be an object.")
        action_names.append(name)
        action_starts.append(current)

        if name == "perform_greeting":
            closed = _gripper_goal(
                limits, open_gripper=False, open_is_high=gripper_open_is_high
            )
            opened = _gripper_goal(
                limits, open_gripper=True, open_is_high=gripper_open_is_high
            )
            original = dict(current)
            close_companions = offsets(original, {
                "elbow_flex.pos": -0.035, "wrist_flex.pos": -0.025, "wrist_roll.pos": 0.04,
            })
            open_companions = offsets(original, {
                "elbow_flex.pos": 0.035, "wrist_flex.pos": 0.025, "wrist_roll.pos": -0.04,
            })
            count = _count(arguments, default=2)
            for index in range(count):
                add(f"greeting close {index + 1}/{count}", {**close_companions, "gripper.pos": closed}, 0.06)
                add(f"greeting open {index + 1}/{count}", {**open_companions, "gripper.pos": opened}, 0.06,
                    completed_cycles=index + 1)
            add("greeting restore", _restore_updates(original, (*close_companions, "gripper.pos"), limits), 0.08,
                completed_cycles=count)
        elif name in {"open_gripper", "close_gripper"}:
            original = dict(current)
            companions = offsets(original, {"elbow_flex.pos": 0.035, "wrist_flex.pos": -0.025})
            add(name.replace("_", " "), {
                **companions,
                "gripper.pos": _gripper_goal(
                    limits, open_gripper=name == "open_gripper", open_is_high=gripper_open_is_high,
                ),
            })
            if companions:
                add("gripper companions restore; retain requested gripper state",
                    _restore_updates(original, tuple(companions), limits))
        elif name in {"wave_wrist", "happy_wiggle"}:
            count = _count(arguments, default=3 if name == "wave_wrist" else 2)
            original = dict(current)
            companion_joints = ("shoulder_pan.pos", "elbow_flex.pos", "wrist_flex.pos")
            if name == "happy_wiggle":
                companion_joints = (*companion_joints, "shoulder_lift.pos")
            left_companions = offsets(original, {joint: 0.025 for joint in companion_joints})
            right_companions = offsets(original, {joint: -0.025 for joint in companion_joints})
            return_targets = _restore_updates(original, (*left_companions, "wrist_roll.pos"), limits)
            low, high = limits["wrist_roll.pos"]
            amplitude_scale = _number(arguments, "amplitude", 1.0, 0.25, 1.0)
            amplitude = min(WRIST_WAVE_MAX_AMPLITUDE, (high - low) * 0.18) * amplitude_scale
            for index in range(count):
                add(
                    f"{name} left {index + 1}/{count}",
                    {**left_companions, "wrist_roll.pos": original["wrist_roll.pos"] - amplitude},
                )
                add(
                    f"{name} right {index + 1}/{count}",
                    {**right_companions, "wrist_roll.pos": original["wrist_roll.pos"] + amplitude},
                    completed_cycles=index + 1,
                )
            add(f"{name} restore", return_targets, completed_cycles=count)
        elif name == "nod":
            count = _count(arguments, default=2)
            intensity = arguments.get("intensity", "normal")
            if intensity not in {"small", "normal"}:
                raise ValueError("Nod intensity must be small or normal.")
            original = dict(current)
            companions = offsets(original, {"shoulder_lift.pos": -0.025, "wrist_roll.pos": 0.04})
            return_targets = _restore_updates(
                original, (*companions, "elbow_flex.pos", "wrist_flex.pos"), limits,
            )
            up, down = _pet_pair(
                original, limits, travel_scale=0.55 if intensity == "small" else 1.0
            )
            for index in range(count):
                add(f"nod down {index + 1}/{count}", {**companions, **down}, 0.08)
                add(f"nod up {index + 1}/{count}", return_targets, 0.08,
                    completed_cycles=index + 1)
            add("nod restore measured start", return_targets, 0.10, completed_cycles=count)
        elif name == "perform_response_gesture":
            category = arguments.get("category", "neutral")
            if not isinstance(category, str):
                raise ValueError("Response category must be a string.")
            original = dict(current)
            template = _response_template(category, original, limits, tuple(available))
            touched = tuple(
                joint for joint in ALL_JOINTS if joint in available
                if any(joint in phase for phase in template.phases)
            )
            return_targets = _restore_updates(original, touched, limits)
            orientations: dict[str, float] = {}
            for index, phase in enumerate(template.phases):
                updates: dict[str, float] = {}
                for joint, fraction in phase.items():
                    if joint not in available:
                        continue
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
                    proposed = max(
                        inner_low, min(inner_high, start + offset * orientations[joint])
                    )
                    # A preferred inner margin must not manufacture a larger
                    # auxiliary stroke when the measured pose is near an edge.
                    updates[joint] = max(start - abs(offset), min(start + abs(offset), proposed))
                add(
                    f"response {template.name} phase {index + 1}/{len(template.phases)}",
                    updates, template.hold_s,
                )
            add(f"response {template.name} restore measured start", return_targets, 0.10)
            action_names[-1] = f"perform_response_gesture:{template.name}"
        elif name == "enter_pet_pose":
            original = dict(current)
            return_targets, down = _pet_pair(original, limits)
            companions = offsets(original, {"shoulder_lift.pos": -0.02, "wrist_roll.pos": 0.03})
            return_targets.update(_restore_updates(original, tuple(companions), limits))
            add("lower head directly and hold two seconds", {**companions, **down}, 2.0)
            add("pet restore measured start", return_targets, 0.10)
            clear_restore = True
        elif name == "leave_pet_pose":
            up = offsets(current, {"shoulder_lift.pos": 0.02})
            for joint, max_travel in HEAD_UP_MAX_TRAVEL.items():
                low, high = map(float, limits[joint])
                if not low <= current[joint] <= high:
                    raise ValueError(f"Head gesture starts outside the configured range: {joint}.")
                up[joint] = min(high, current[joint] + min(max_travel, (high - low) * 0.08))
            add("raise head a small step from measured pose", up, 0.15)
            clear_restore = True
        elif name == "return_to_action_start":
            joints = tuple(available if recovery_joints is None else recovery_joints)
            if not set(joints) <= available:
                raise ValueError("Recovery may only move previously controlled joints.")
            add(
                "return to previous measured action start",
                _restore_updates(home, joints, limits),
                0.15,
            )
            clear_restore = True
        elif name == "return_to_neutral":
            add(
                "return to connection pose",
                _restore_updates(home, tuple(joint for joint in ALL_JOINTS if joint in available), limits),
                0.15,
            )
            clear_restore = True
        elif name == "return_to_plan_start":
            start_index = arguments.get("start_action_index", 0)
            if type(start_index) is not int or not 0 <= start_index < action_index:
                raise ValueError("Invalid action group start index.")
            touched = {
                joint for frame in frames if frame.action_index >= start_index
                for joint in frame.active_joints
            }
            joints = tuple(joint for joint in ALL_JOINTS if joint in touched)
            add(
                "return to this plan's measured start",
                _restore_updates(action_starts[start_index], joints, limits),
                0.0,
            )
        elif name == "wait":
            seconds = _number(arguments, "seconds", 1.0, 0.0, 10.0)
            add("wait", {}, seconds)
        elif name == "play_follower_trajectory":
            if set(arguments) - {"samples", "return_to_start", "calibration_fingerprint"}:
                raise ValueError("Unknown follower-trajectory argument.")
            trajectory = follower_trajectory(arguments.get("samples"), limits)
            return_to_start = arguments.get("return_to_start", False)
            if type(return_to_start) is not bool:
                raise ValueError("Return-to-start must be a boolean.")
            entry = dict(current)
            add("approach recorded start", trajectory.position_at(0.0), 0.0)
            add("play recorded follower trajectory", trajectory.position_at(trajectory.duration_s), 0.0)
            frames[-1] = AgentKeyframe(
                frames[-1].label, frames[-1].targets, active_joints=frames[-1].active_joints,
                action_index=action_index, trajectory=trajectory,
            )
            if return_to_start:
                add("return to recording entry pose", _restore_updates(entry, ALL_JOINTS, limits), 0.0)
        elif name == "play_saved_action":
            if set(arguments) - {"frames", "return_to_start"}:
                raise ValueError("Unknown saved-action argument.")
            saved_frames = arguments.get("frames")
            if (not isinstance(saved_frames, Sequence) or isinstance(saved_frames, (str, bytes))
                    or not 1 <= len(saved_frames) <= MAX_SAVED_ACTION_FRAMES):
                raise ValueError(f"A saved action must contain 1 to {MAX_SAVED_ACTION_FRAMES} expanded frames.")
            primitives: list[AgentKeyframe] = []
            for index, saved_frame in enumerate(saved_frames):
                if not isinstance(saved_frame, Mapping) or set(saved_frame) - {"targets", "hold_s", "duration_s", "joint_timing"}:
                    raise ValueError("Invalid saved-action frame.")
                updates = _validated_updates(saved_frame.get("targets", {}))
                if not updates:
                    raise ValueError("Every saved-action frame requires at least one joint target.")
                duration = None if saved_frame.get("duration_s") is None else _number(
                    saved_frame, "duration_s", 0.1, 0.1, 10.0,
                )
                primitives.append(AgentKeyframe(
                    f"saved action frame {index + 1}/{len(saved_frames)}", updates,
                    _number(saved_frame, "hold_s", 0.0, 0.0, 10.0),
                    tuple(joint for joint in ALL_JOINTS if joint in updates), duration_s=duration,
                    joint_timing=saved_frame.get("joint_timing"),
                ))
            # Capture this action's entry pose, after earlier semantic actions.
            # A full-plan start or connection pose would undo an earlier open
            # gripper/pose command when a compound interaction reaches here.
            compiled = sequence(
                *primitives, current_positions=current, limits=limits,
                fixed_positions=fixed_positions,
                return_to_start=arguments.get("return_to_start", False),
            )
            for frame in compiled.keyframes:
                add(frame.label, {joint: frame.targets[joint] for joint in frame.active_joints},
                    frame.hold_s, duration_s=frame.duration_s, joint_timing=frame.joint_timing)
        elif name == "move_joints":
            if set(arguments) - {"targets", "hold_s", "duration_s", "return_to_start", "joint_timing"}:
                raise ValueError("Unknown local joint-motion argument.")
            updates = _validated_updates(arguments.get("targets", {}))
            if not updates:
                raise ValueError("move_joints requires at least one joint target.")
            _restore_updates(updates, tuple(updates), limits)
            hold_s = _number(arguments, "hold_s", 0.0, 0.0, 10.0)
            duration_s = None if arguments.get("duration_s") is None else _number(arguments, "duration_s", 0.1, 0.05, 10.0)
            return_to_start = arguments.get("return_to_start", False)
            if not isinstance(return_to_start, bool):
                raise ValueError("return_to_start must be a boolean.")
            original = dict(current)
            add("local joint targets", updates, hold_s, duration_s=duration_s,
                joint_timing=arguments.get("joint_timing"))
            if return_to_start:
                add("local joints restore measured start", _restore_updates(original, tuple(updates), limits), 0.0,
                    duration_s=duration_s)

    if not frames:
        raise ValueError("Agent plan did not produce any motion keyframes.")
    explicit_seconds = sum(
        frame.hold_s + (frame.trajectory.duration_s if frame.trajectory else frame.duration_s or 0.0)
        for frame in frames
    )
    if explicit_seconds > MAX_EXPLICIT_PLAN_SECONDS:
        raise ValueError(f"Explicit motion and hold times may total at most {MAX_EXPLICIT_PLAN_SECONDS:g} seconds.")
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
    fixed = _validated_updates(fixed_positions)
    moving = AGENT_JOINTS - fixed.keys()
    if isinstance(max_step, Mapping) and moving - max_step.keys():
        raise ValueError("Agent max_step is missing an active joint's limit.")
    raw_steps = [max_step[joint] for joint in moving] if isinstance(max_step, Mapping) else [max_step]
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in raw_steps):
        raise ValueError("Agent max_step must contain numeric step limits.")
    steps = (
        {joint: float(max_step[joint]) for joint in moving}
        if isinstance(max_step, Mapping)
        else {joint: float(max_step) for joint in moving}
    )
    if not all(math.isfinite(value) and value > 0 for value in steps.values()):
        raise ValueError("Agent max_step must be positive and finite for every controlled joint.")
    if not math.isfinite(speed_scale) or not 0.0 < speed_scale <= 1.0:
        raise ValueError("Agent speed_scale must be in (0, 1].")

    result: dict[str, float] = {}
    reached = True
    for joint in ALL_JOINTS:
        if joint in fixed:
            result[joint] = fixed[joint]
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
