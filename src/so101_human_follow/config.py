from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ALL_JOINTS = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
)

CONTROLLED_JOINTS = frozenset(
    {
        "elbow_flex.pos",
        "wrist_flex.pos",
        "wrist_roll.pos",
        "gripper.pos",
    }
)
FIXED_JOINTS = frozenset({"shoulder_pan.pos", "shoulder_lift.pos"})


@dataclass(frozen=True)
class CameraSettings:
    index: int
    width: int
    height: int
    fps: int
    rotation_degrees: float
    flip_horizontal: bool
    flip_vertical: bool


@dataclass(frozen=True)
class PoseSettings:
    model_path: Path
    hand_model_path: Path
    side: str
    min_detection_confidence: float
    min_presence_confidence: float
    min_tracking_confidence: float
    min_landmark_visibility: float
    hand_min_detection_confidence: float
    hand_min_presence_confidence: float
    hand_min_tracking_confidence: float
    max_hand_wrist_distance: float
    neutral_sample_count: int


@dataclass(frozen=True)
class RobotSettings:
    port: str
    id: str
    control_hz: float
    max_relative_target: float | dict[str, float]
    disable_torque_on_disconnect: bool


@dataclass(frozen=True)
class FixedJointSettings:
    mode: str
    names: tuple[str, ...]
    constant_values: dict[str, float]


@dataclass(frozen=True)
class MappingSettings:
    controlled_joints: tuple[str, ...]
    gains: dict[str, float]
    signs: dict[str, float]
    feature_deadbands: dict[str, float]
    fixed_joints: FixedJointSettings


@dataclass(frozen=True)
class SafetySettings:
    ema_alpha: float
    max_step_per_cycle: float
    relative_limits: dict[str, float]
    absolute_limits: dict[str, tuple[float, float]]
    startup_body_abs_limit: float
    valid_frames_before_arm: int
    tracking_timeout_s: float
    max_person_center_jump: float
    skip_safe_range_calibration: bool


@dataclass(frozen=True)
class ViewerSettings:
    enabled: bool
    memory_limit: str


@dataclass(frozen=True)
class AppConfig:
    camera: CameraSettings
    pose: PoseSettings
    robot: RobotSettings
    mapping: MappingSettings
    safety: SafetySettings
    viewer: ViewerSettings


def _float_dict(raw: dict[str, Any]) -> dict[str, float]:
    return {str(key): float(value) for key, value in raw.items()}


def _relative_target(raw: Any) -> float | dict[str, float]:
    if isinstance(raw, dict):
        return _float_dict(raw)
    return float(raw)


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        raw = json.load(stream)

    camera_raw = raw["camera"]
    pose_raw = raw["pose"]
    robot_raw = raw["robot"]
    mapping_raw = raw["mapping"]
    fixed_raw = mapping_raw["fixed_joints"]
    safety_raw = raw["safety"]
    viewer_raw = raw.get("viewer", {"enabled": True, "memory_limit": "10%"})

    model_path = Path(pose_raw["model_path"]).expanduser()
    if not model_path.is_absolute():
        model_path = (config_path.parent / model_path).resolve()
    hand_model_path = Path(pose_raw["hand_model_path"]).expanduser()
    if not hand_model_path.is_absolute():
        hand_model_path = (config_path.parent / hand_model_path).resolve()

    config = AppConfig(
        camera=CameraSettings(
            index=int(camera_raw["index"]),
            width=int(camera_raw["width"]),
            height=int(camera_raw["height"]),
            fps=int(camera_raw["fps"]),
            rotation_degrees=float(camera_raw.get("rotation_degrees", 0.0)),
            flip_horizontal=bool(camera_raw.get("flip_horizontal", False)),
            flip_vertical=bool(camera_raw.get("flip_vertical", False)),
        ),
        pose=PoseSettings(
            model_path=model_path,
            hand_model_path=hand_model_path,
            side=str(pose_raw["side"]).lower(),
            min_detection_confidence=float(pose_raw["min_detection_confidence"]),
            min_presence_confidence=float(pose_raw["min_presence_confidence"]),
            min_tracking_confidence=float(pose_raw["min_tracking_confidence"]),
            min_landmark_visibility=float(pose_raw["min_landmark_visibility"]),
            hand_min_detection_confidence=float(pose_raw["hand_min_detection_confidence"]),
            hand_min_presence_confidence=float(pose_raw["hand_min_presence_confidence"]),
            hand_min_tracking_confidence=float(pose_raw["hand_min_tracking_confidence"]),
            max_hand_wrist_distance=float(pose_raw["max_hand_wrist_distance"]),
            neutral_sample_count=int(pose_raw["neutral_sample_count"]),
        ),
        robot=RobotSettings(
            port=str(robot_raw["port"]),
            id=str(robot_raw["id"]),
            control_hz=float(robot_raw["control_hz"]),
            max_relative_target=_relative_target(robot_raw["max_relative_target"]),
            disable_torque_on_disconnect=bool(robot_raw["disable_torque_on_disconnect"]),
        ),
        mapping=MappingSettings(
            controlled_joints=tuple(mapping_raw["controlled_joints"]),
            gains=_float_dict(mapping_raw["gains"]),
            signs=_float_dict(mapping_raw["signs"]),
            feature_deadbands=_float_dict(mapping_raw["feature_deadbands"]),
            fixed_joints=FixedJointSettings(
                mode=str(fixed_raw["mode"]).lower(),
                names=tuple(fixed_raw["names"]),
                constant_values=_float_dict(fixed_raw["constant_values"]),
            ),
        ),
        safety=SafetySettings(
            ema_alpha=float(safety_raw["ema_alpha"]),
            max_step_per_cycle=float(safety_raw["max_step_per_cycle"]),
            relative_limits=_float_dict(safety_raw["relative_limits"]),
            absolute_limits={
                str(key): (float(value[0]), float(value[1]))
                for key, value in safety_raw["absolute_limits"].items()
            },
            startup_body_abs_limit=float(safety_raw["startup_body_abs_limit"]),
            valid_frames_before_arm=int(safety_raw["valid_frames_before_arm"]),
            tracking_timeout_s=float(safety_raw["tracking_timeout_s"]),
            max_person_center_jump=float(safety_raw["max_person_center_jump"]),
            skip_safe_range_calibration=bool(
                safety_raw.get("skip_safe_range_calibration", False)
            ),
        ),
        viewer=ViewerSettings(
            enabled=bool(viewer_raw["enabled"]),
            memory_limit=str(viewer_raw["memory_limit"]),
        ),
    )
    validate_config(config)
    return config


def validate_config(config: AppConfig) -> None:
    controlled = set(config.mapping.controlled_joints)
    fixed = set(config.mapping.fixed_joints.names)
    expected = set(ALL_JOINTS)

    if controlled & fixed:
        raise ValueError("Controlled and fixed joint sets must not overlap.")
    if controlled | fixed != expected:
        raise ValueError(f"Controlled and fixed joints must cover exactly {sorted(expected)}.")
    if controlled != CONTROLLED_JOINTS or fixed != FIXED_JOINTS:
        raise ValueError(
            "This controller moves only elbow_flex, wrist_flex, wrist_roll and gripper; "
            "shoulder_pan and shoulder_lift must remain fixed."
        )
    if config.mapping.fixed_joints.mode not in {"startup", "constant"}:
        raise ValueError("fixed_joints.mode must be 'startup' or 'constant'.")

    for mapping_name, mapping in (
        ("gains", config.mapping.gains),
        ("signs", config.mapping.signs),
        ("feature_deadbands", config.mapping.feature_deadbands),
        ("relative_limits", config.safety.relative_limits),
    ):
        if set(mapping) != controlled:
            raise ValueError(f"{mapping_name} keys must equal the controlled joint set.")

    if set(config.safety.absolute_limits) != expected:
        raise ValueError("absolute_limits must contain all six joints.")
    if config.mapping.fixed_joints.mode == "constant":
        if set(config.mapping.fixed_joints.constant_values) != fixed:
            raise ValueError("constant_values must contain every fixed joint.")

    if config.pose.side not in {"left", "right"}:
        raise ValueError("pose.side must be 'left' or 'right'.")
    if config.camera.width <= 0 or config.camera.height <= 0 or config.camera.fps <= 0:
        raise ValueError("Camera width, height and fps must be positive.")
    if not math.isfinite(config.camera.rotation_degrees):
        raise ValueError("Camera rotation_degrees must be finite.")
    if config.pose.neutral_sample_count <= 0:
        raise ValueError("neutral_sample_count must be positive.")
    for name, value in (
        ("ema_alpha", config.safety.ema_alpha),
        ("min_detection_confidence", config.pose.min_detection_confidence),
        ("min_presence_confidence", config.pose.min_presence_confidence),
        ("min_tracking_confidence", config.pose.min_tracking_confidence),
        ("min_landmark_visibility", config.pose.min_landmark_visibility),
        ("hand_min_detection_confidence", config.pose.hand_min_detection_confidence),
        ("hand_min_presence_confidence", config.pose.hand_min_presence_confidence),
        ("hand_min_tracking_confidence", config.pose.hand_min_tracking_confidence),
    ):
        if not 0.0 < value <= 1.0:
            raise ValueError(f"{name} must be in (0, 1].")
    if not all(
        math.isfinite(value)
        for value in (
            config.robot.control_hz,
            config.safety.max_step_per_cycle,
            config.safety.startup_body_abs_limit,
            config.safety.tracking_timeout_s,
            config.safety.max_person_center_jump,
        )
    ):
        raise ValueError("All numeric safety and robot settings must be finite.")
    if config.robot.control_hz <= 0 or config.safety.max_step_per_cycle <= 0:
        raise ValueError("Control rate and maximum step must be positive.")
    relative_target = config.robot.max_relative_target
    if isinstance(relative_target, dict):
        expected_motors = {joint.removesuffix(".pos") for joint in ALL_JOINTS}
        if set(relative_target) != expected_motors:
            raise ValueError("max_relative_target motor keys must cover all six motors.")
        relative_values = tuple(relative_target.values())
    else:
        relative_values = (relative_target,)
    if not all(math.isfinite(value) and value > 0 for value in relative_values):
        raise ValueError("max_relative_target values must be positive finite numbers.")
    if config.safety.startup_body_abs_limit <= 0 or config.safety.tracking_timeout_s <= 0:
        raise ValueError("Startup limit and tracking timeout must be positive.")
    if not 0.0 < config.safety.max_person_center_jump <= 1.0:
        raise ValueError("max_person_center_jump must be in (0, 1].")
    if not math.isfinite(config.pose.max_hand_wrist_distance) or not (
        0.0 < config.pose.max_hand_wrist_distance <= 1.0
    ):
        raise ValueError("max_hand_wrist_distance must be in (0, 1].")
    if config.safety.valid_frames_before_arm <= 0:
        raise ValueError("valid_frames_before_arm must be positive.")

    for joint in controlled:
        values = (
            config.mapping.gains[joint],
            config.mapping.signs[joint],
            config.mapping.feature_deadbands[joint],
            config.safety.relative_limits[joint],
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"Mapping values for {joint} must be finite.")
        if config.mapping.gains[joint] <= 0 or config.safety.relative_limits[joint] <= 0:
            raise ValueError(f"Gain and relative limit for {joint} must be positive.")
        if config.mapping.feature_deadbands[joint] < 0:
            raise ValueError(f"Deadband for {joint} cannot be negative.")
        if config.mapping.signs[joint] not in {-1.0, 1.0}:
            raise ValueError(f"Sign for {joint} must be exactly -1.0 or 1.0.")

    for joint, (low, high) in config.safety.absolute_limits.items():
        if not math.isfinite(low) or not math.isfinite(high) or low >= high:
            raise ValueError(f"Invalid absolute limits for {joint}: {(low, high)}")
    if not all(math.isfinite(value) for value in config.mapping.fixed_joints.constant_values.values()):
        raise ValueError("Fixed joint constants must be finite.")

