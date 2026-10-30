from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

from .config import AppConfig, load_config
from .pose_detector import (
    MediaPipePoseDetector,
    PoseDetection,
    annotate_frame,
    orient_camera_frame,
)
from .pose_math import PoseFeatures, median_features
from .robot_adapter import SO101RobotAdapter, preview_initial_positions
from .safety import FEATURE_BY_JOINT, MotionController
from .viewer import RerunViewer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONTROL_FEATURES = frozenset(FEATURE_BY_JOINT.values())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Camera pose controller for one SO-101 follower arm. Robot control is disabled by default."
    )
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.json")
    parser.add_argument(
        "--enable-robot",
        action="store_true",
        help="Explicitly connect to COM4 and permit commands after manual arming.",
    )
    parser.add_argument("--camera-index", type=int, help="Override camera.index from config.json.")
    parser.add_argument("--port", help="Override robot.port from config.json.")
    parser.add_argument("--robot-id", help="Override robot.id from config.json.")
    parser.add_argument("--no-viewer", action="store_true", help="Do not launch Rerun.")
    return parser


def _apply_overrides(config: AppConfig, args: argparse.Namespace) -> AppConfig:
    camera = config.camera
    robot = config.robot
    viewer = config.viewer
    if args.camera_index is not None:
        camera = replace(camera, index=args.camera_index)
    if args.port:
        robot = replace(robot, port=args.port)
    if args.robot_id:
        robot = replace(robot, id=args.robot_id)
    if args.no_viewer:
        viewer = replace(viewer, enabled=False)
    return replace(config, camera=camera, robot=robot, viewer=viewer)


def _poll_keys() -> list[str]:
    if os.name != "nt":
        return []
    import msvcrt

    keys = []
    while msvcrt.kbhit():
        keys.append(msvcrt.getwch().lower())
    return keys


def _open_camera(config: AppConfig):
    import cv2

    backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
    camera = cv2.VideoCapture(config.camera.index, backend)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(f"Could not open camera index {config.camera.index}.")
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.camera.width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.camera.height)
    camera.set(cv2.CAP_PROP_FPS, config.camera.fps)
    return camera


def _status_lines(
    *,
    robot_enabled: bool,
    robot_port: str,
    armed: bool,
    detection: PoseDetection,
    neutral: PoseFeatures | None,
    collecting: int,
    required_samples: int,
) -> list[str]:
    mode = f"ROBOT {robot_port}" if robot_enabled else "PREVIEW ONLY"
    motion = "ARMED" if armed else "DISARMED"
    pose = f"pose: {detection.reason}"
    calibration = "neutral: ready" if neutral else "neutral: press C"
    if collecting:
        calibration = f"neutral: collecting {collecting}/{required_samples}"
    return [
        f"{mode} | {motion}",
        pose,
        calibration,
        "C=neutral  SPACE=arm  Q=quit  X=fast software stop",
    ]


def run(config: AppConfig, *, enable_robot: bool) -> int:
    import cv2

    detector = None
    camera = None
    viewer = None
    robot = None

    armed = False
    quit_requested = False
    exit_pending = False
    neutral: PoseFeatures | None = None
    neutral_samples: list[PoseFeatures] = []
    collecting = False
    valid_streak = 0
    last_valid_time: float | None = None
    last_center_x: float | None = None
    last_control_time = 0.0
    last_status_time = 0.0
    initial = preview_initial_positions()

    try:
        detector = MediaPipePoseDetector(config.pose)
        camera = _open_camera(config)
        viewer = RerunViewer(enabled=config.viewer.enabled, memory_limit=config.viewer.memory_limit)
        robot = SO101RobotAdapter(config.robot) if enable_robot else None

        if enable_robot:
            print("\nREAL ROBOT MODE requested.")
            print(f"Follower: {config.robot.id} on {config.robot.port}")
            print("ID3-6 (elbow_flex, wrist_flex, wrist_roll, gripper) can move independently.")
            print("ID1-2 (shoulder_pan, shoulder_lift) will be held at startup positions.")
            confirmation = input("Clear the workspace and type CONNECT to continue: ").strip()
            if confirmation != "CONNECT":
                print("Cancelled before connecting to the robot.")
                return 2
            initial = robot.connect()
            print("Robot connected. Motion is still DISARMED.")
            print("Startup positions:", initial)
        else:
            print("PREVIEW ONLY: no robot will be imported, connected or commanded.")

        controller = MotionController(initial, config.mapping, config.safety)
        last_action = controller.hold()
        control_period = 1.0 / config.robot.control_hz

        print("Focus this terminal for keyboard controls: C, Space, Q, X.")
        while not quit_requested:
            ok, frame = camera.read()
            now = time.monotonic()
            if not ok:
                armed = False
                raise RuntimeError("Camera read failed; motion was disarmed.")
            frame = orient_camera_frame(
                frame,
                rotation_degrees=config.camera.rotation_degrees,
                flip_horizontal=config.camera.flip_horizontal,
                flip_vertical=config.camera.flip_vertical,
            )

            detection = detector.detect(frame)
            features = detection.features
            valid_features = frozenset(set(detection.valid_features) & set(CONTROL_FEATURES))
            if (
                features is not None
                and "center_x" in detection.valid_features
                and last_center_x is not None
            ):
                if abs(features.center_x - last_center_x) > config.safety.max_person_center_jump:
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
                    last_center_x = None

            if detection.reason in {"person changed", "multiple people"} and armed:
                armed = False
                print(f"{detection.reason}: immediately DISARMED. Manual re-arm is required.")

            if features is not None and valid_features:
                if "center_x" in detection.valid_features:
                    last_center_x = features.center_x
                last_valid_time = now
                valid_streak += 1
            else:
                valid_streak = 0
                if collecting and neutral_samples:
                    neutral_samples.clear()
                if (
                    armed
                    and last_valid_time is not None
                    and now - last_valid_time > config.safety.tracking_timeout_s
                ):
                    armed = False
                    print("Tracking lost: automatically DISARMED. Manual re-arm is required.")

            keys = _poll_keys()
            for key in keys:
                if key == "x":
                    armed = False
                    print("Fast software stop requested. This is not a physical emergency stop.")
                    quit_requested = True
                    break
                if key in {"q", "\x1b"}:
                    armed = False
                    if robot is not None and not exit_pending:
                        exit_pending = True
                        print(
                            "DISARMED and holding. Lower/support the arm, then press Q again to disconnect "
                            "and release torque. Press Space to cancel exit."
                        )
                    else:
                        quit_requested = True
                        break
                    continue
                if exit_pending:
                    if key == " ":
                        exit_pending = False
                        print("Exit cancelled. Motion remains DISARMED.")
                    continue
                if key == "c":
                    if features is None:
                        print("Cannot collect neutral values: no reliable arm or hand channel.")
                    else:
                        armed = False
                        neutral = None
                        collecting = True
                        neutral_samples.clear()
                        print("Collecting neutral pose. Hold still...")
                if key == " ":
                    if armed:
                        armed = False
                        print("DISARMED: no new actions will be sent.")
                    elif collecting:
                        print("ARM refused: neutral-pose collection is still in progress.")
                    elif neutral is None:
                        print("ARM refused: press C and finish neutral-pose collection first.")
                    elif valid_streak < config.safety.valid_frames_before_arm:
                        print("ARM refused: pose has not been stable for enough consecutive frames.")
                    else:
                        armed = True
                        print("ARMED. Keep the physical power switch within reach.")

            if quit_requested:
                break

            if (
                collecting
                and features is not None
                and CONTROL_FEATURES.issubset(valid_features)
            ):
                neutral_samples.append(features)
                if len(neutral_samples) >= config.pose.neutral_sample_count:
                    neutral = median_features(neutral_samples)
                    collecting = False
                    neutral_samples.clear()
                    print(f"Neutral pose ready: {neutral}")

            if (
                armed
                and features is not None
                and neutral is not None
                and valid_streak >= config.safety.valid_frames_before_arm
                and now - last_control_time >= control_period
            ):
                planned_action = controller.compute(features, neutral, valid_features)
                if robot is not None:
                    actual_action = robot.send(planned_action)
                    last_action = {**planned_action, **actual_action}
                    controller.reset_to(last_action)
                else:
                    last_action = planned_action
                last_control_time = now

            lines = _status_lines(
                robot_enabled=enable_robot,
                robot_port=config.robot.port,
                armed=armed,
                detection=detection,
                neutral=neutral,
                collecting=len(neutral_samples) if collecting else 0,
                required_samples=config.pose.neutral_sample_count,
            )
            annotated = annotate_frame(frame, detection, lines)
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            viewer.log(
                annotated_rgb,
                action=last_action,
                armed=armed,
                pose_valid=features is not None,
            )

            if now - last_status_time >= 1.0:
                print(
                    f"{lines[0]} | {detection.reason} | valid_streak={valid_streak} | "
                    f"fixed lower joints={{{', '.join(f'{j}: {last_action[j]:.1f}' for j in config.mapping.fixed_joints.names)}}}"
                )
                last_status_time = now
        return 0
    finally:
        armed = False
        if robot is not None and robot.connected:
            print("Disconnecting robot. Torque release may let the arm fall; support or lower it.")
            try:
                robot.disconnect()
            except Exception as exc:
                print(f"Robot disconnect warning: {exc}", file=sys.stderr)
        if viewer is not None:
            viewer.close()
        if detector is not None:
            try:
                detector.close()
            except Exception as exc:
                print(f"Pose detector shutdown warning: {exc}", file=sys.stderr)
        if camera is not None:
            try:
                camera.release()
            except Exception as exc:
                print(f"Camera shutdown warning: {exc}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = _apply_overrides(load_config(args.config), args)
    try:
        return run(config, enable_robot=args.enable_robot)
    except KeyboardInterrupt:
        print("\nInterrupted; motion is stopping.")
        return 130
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

