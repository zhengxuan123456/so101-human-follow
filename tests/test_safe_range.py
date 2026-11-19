import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from so101_human_follow.config import load_config
from so101_human_follow.pose_math import PoseFeatures
from so101_human_follow.robot_adapter import SO101RobotAdapter, preview_initial_positions
from so101_human_follow.safe_range import (
    SafeRangeProfile,
    load_safe_range_profile,
    save_safe_range_profile,
)
from so101_human_follow.safety import MotionController
from so101_human_follow.web_engine import ControlEngine, ControlError


def make_profile() -> SafeRangeProfile:
    return SafeRangeProfile.create(
        robot_id="my_awesome_follower_arm",
        calibration_fingerprint="fingerprint-test-value",
        controlled_limits={
            "elbow_flex.pos": (-10.0, 10.0),
            "wrist_flex.pos": (-12.0, 12.0),
            "wrist_roll.pos": (-15.0, 15.0),
            "gripper.pos": (5.0, 95.0),
        },
        fixed_positions={
            "shoulder_pan.pos": 0.0,
            "shoulder_lift.pos": 0.0,
        },
        startup_positions=preview_initial_positions(),
    )


class SafeRangeProfileTest(unittest.TestCase):
    def create_with(self, **overrides):
        original = make_profile()
        arguments = {
            "robot_id": original.robot_id,
            "calibration_fingerprint": original.calibration_fingerprint,
            "controlled_limits": dict(original.controlled_limits),
            "fixed_positions": dict(original.fixed_positions),
            "startup_positions": dict(original.startup_positions),
        }
        arguments.update(overrides)
        return SafeRangeProfile.create(**arguments)

    def test_profile_round_trips_through_atomic_file(self):
        profile = make_profile()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "safety_profile.json"
            save_safe_range_profile(path, profile)
            loaded = load_safe_range_profile(path)

            self.assertEqual(loaded, profile)
            self.assertEqual([item.name for item in Path(directory).iterdir()], [path.name])

    def test_profile_rejects_tiny_range(self):
        limits = dict(make_profile().controlled_limits)
        limits["wrist_roll.pos"] = (1.0, 2.0)
        with self.assertRaisesRegex(ValueError, "too small"):
            SafeRangeProfile.create(
                robot_id="my_awesome_follower_arm",
                calibration_fingerprint="fingerprint-test-value",
                controlled_limits=limits,
                fixed_positions=make_profile().fixed_positions,
                startup_positions=preview_initial_positions(),
            )

    def test_captured_profile_defaults_to_strict_middle_margin_on_every_controlled_joint(self):
        original = make_profile()
        for joint, (low, high) in original.controlled_limits.items():
            for position in (low, low + 0.25, high - 0.25, high):
                with self.subTest(joint=joint, position=position):
                    startup = {**original.startup_positions, joint: position}
                    with self.assertRaisesRegex(ValueError, "comfortable middle"):
                        self.create_with(startup_positions=startup)

    def test_configured_envelope_keeps_exact_boundary_or_outside_connection_pose(self):
        limits = dict(make_profile().controlled_limits)
        limits["wrist_flex.pos"] = (-100.0, 100.0)
        for position in (-100.0, 100.0, -100.12, 100.12):
            with self.subTest(position=position):
                startup = {**preview_initial_positions(), "wrist_flex.pos": position}
                profile = self.create_with(
                    controlled_limits=limits, startup_positions=startup,
                    require_startup_middle=False,
                )
                self.assertEqual(profile.startup_positions, startup)
                self.assertEqual(profile.controlled_limits, limits)
                profile.validate(require_startup_middle=False)
                with self.assertRaisesRegex(ValueError, "comfortable middle"):
                    profile.validate()

    def test_configured_connection_flag_does_not_relax_profile_structure_or_limits(self):
        original = make_profile()
        invalid_inputs = (
            {"startup_positions": {**original.startup_positions, "wrist_flex.pos": float("nan")}},
            {"startup_positions": {"wrist_flex.pos": 0.0}},
            {"controlled_limits": {**original.controlled_limits, "wrist_flex.pos": (0.0, 1.0)}},
            {"controlled_limits": {**original.controlled_limits, "wrist_flex.pos": (0.0, float("inf"))}},
            {"fixed_positions": {**original.fixed_positions, "shoulder_pan.pos": float("nan")}},
        )
        for inputs in invalid_inputs:
            with self.subTest(inputs=inputs):
                with self.assertRaises(ValueError):
                    self.create_with(**inputs, require_startup_middle=False)

    def test_startup_middle_flag_requires_an_explicit_boolean(self):
        for value in (None, 0, 1, "false"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "must be a boolean"):
                    self.create_with(require_startup_middle=value)

    def test_saving_a_nonmiddle_configured_pose_still_requires_captured_profile_validation(self):
        profile = self.create_with(
            startup_positions={**preview_initial_positions(), "wrist_flex.pos": -12.1},
            require_startup_middle=False,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "safety_profile.json"
            with self.assertRaisesRegex(ValueError, "comfortable middle"):
                save_safe_range_profile(path, profile)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_loading_cannot_disable_captured_profile_validation_with_a_serialized_flag(self):
        profile = self.create_with(
            startup_positions={**preview_initial_positions(), "wrist_flex.pos": 12.1},
            require_startup_middle=False,
        )
        raw = {**profile.to_dict(), "require_startup_middle": False}
        with self.assertRaisesRegex(ValueError, "comfortable middle"):
            SafeRangeProfile.from_dict(raw)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "safety_profile.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "comfortable middle"):
                load_safe_range_profile(path)


class DynamicLimitTest(unittest.TestCase):
    def test_startup_safety_guards_remain_independent_of_profile_midpoint_validation(self):
        config = load_config("config.example.json")
        invalid_starts = (
            {"wrist_flex.pos": config.safety.startup_body_abs_limit + 0.1},
            {"wrist_flex.pos": -config.safety.startup_body_abs_limit - 0.1},
            {"gripper.pos": config.safety.absolute_limits["gripper.pos"][0] - 0.1},
            {"gripper.pos": config.safety.absolute_limits["gripper.pos"][1] + 0.1},
        )
        for updates in invalid_starts:
            with self.subTest(updates=updates):
                positions = {**preview_initial_positions(), **updates}
                with self.assertRaisesRegex(ValueError, "outside its configured startup range"):
                    MotionController(positions, config.mapping, config.safety)

    def test_motion_controller_never_exceeds_user_safe_range(self):
        config = load_config("config.example.json")
        limits = {
            "elbow_flex.pos": (-0.5, 0.5),
            "wrist_flex.pos": (-0.5, 0.5),
            "wrist_roll.pos": (-0.5, 0.5),
            "gripper.pos": (49.0, 51.0),
        }
        controller = MotionController(
            preview_initial_positions(),
            config.mapping,
            config.safety,
            dynamic_limits=limits,
        )
        neutral = PoseFeatures(0.0, 0.0, 0.0, 0.5, 0.0, 0.0)
        extreme = PoseFeatures(0.0, 1.0, 1.0, 0.5, 0.9, 1.0)

        for _ in range(100):
            action = controller.compute(extreme, neutral)

        for joint, (low, high) in limits.items():
            self.assertGreaterEqual(action[joint], low)
            self.assertLessEqual(action[joint], high)


class ManualRangeAdapterTest(unittest.TestCase):
    def test_only_controlled_joints_are_released_and_current_goal_precedes_enable(self):
        config = load_config("config.example.json")
        events = []
        observations = preview_initial_positions()
        observations.update(
            {
                "elbow_flex.pos": 3.0,
                "wrist_flex.pos": 4.0,
                "wrist_roll.pos": 5.0,
                "gripper.pos": 60.0,
            }
        )

        class FakeBus:
            def disable_torque(self, motors):
                events.append(("disable", list(motors)))

            def sync_write(self, register, goals):
                events.append(("write", register, dict(goals)))

            def enable_torque(self, motors):
                events.append(("enable", list(motors)))

        class FakeRobot:
            bus = FakeBus()

            @staticmethod
            def get_observation():
                return dict(observations)

        adapter = SO101RobotAdapter(config.robot)
        adapter.robot = FakeRobot()
        adapter.connected = True
        controlled = config.mapping.controlled_joints

        adapter.begin_manual_range(controlled)
        result = adapter.finish_manual_range(controlled)

        motor_names = [joint.removesuffix(".pos") for joint in controlled]
        self.assertEqual(events[0], ("disable", motor_names))
        self.assertEqual(events[1][0:2], ("write", "Goal_Position"))
        self.assertEqual(events[1][2], {name: observations[f"{name}.pos"] for name in motor_names})
        self.assertEqual(events[2], ("enable", motor_names))
        self.assertNotIn("shoulder_pan", motor_names)
        self.assertNotIn("shoulder_lift", motor_names)
        self.assertEqual(result, observations)
        self.assertFalse(adapter.manual_range_active)


class SafeRangeArmGateTest(unittest.TestCase):
    def test_arm_is_rejected_until_safe_profile_exists(self):
        config = load_config("config.example.json")
        engine = ControlEngine(config, robot_allowed=True)
        client_id = "client_safe_range_gate"
        engine.heartbeat(client_id)
        features = PoseFeatures(0.0, 0.0, 0.5, 0.5)
        engine._robot = SimpleNamespace(connected=True)
        engine._robot_state = "CONNECTED_DISARMED"
        engine._neutral = features
        engine._latest_features = features
        engine._last_pose_time = time.monotonic()
        engine._valid_streak = config.safety.valid_frames_before_arm

        with self.assertRaisesRegex(ControlError, "safe-range calibration is incomplete"):
            engine._arm(
                "123",
                client_id=client_id,
                submitted_epoch=engine._epoch,
            )
        self.assertFalse(engine._armed)


if __name__ == "__main__":
    unittest.main()
