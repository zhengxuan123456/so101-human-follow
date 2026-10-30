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


class DynamicLimitTest(unittest.TestCase):
    def test_motion_controller_never_exceeds_user_safe_range(self):
        config = load_config("config.json")
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
        config = load_config("config.json")
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
        config = load_config("config.json")
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
