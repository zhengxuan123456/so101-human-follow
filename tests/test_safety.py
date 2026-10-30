import unittest

from so101_human_follow.config import load_config
from so101_human_follow.pose_math import PoseFeatures
from so101_human_follow.robot_adapter import preview_initial_positions
from so101_human_follow.safety import MotionController, signed_deadband


class SafetyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config("config.json")

    def setUp(self):
        self.initial = preview_initial_positions()
        self.controller = MotionController(self.initial, self.config.mapping, self.config.safety)
        self.neutral = PoseFeatures(
            pan=0.0,
            lift=0.0,
            elbow=0.5,
            center_x=0.5,
            hand_roll=0.0,
            hand_open=0.5,
        )

    def moving_features(self):
        return PoseFeatures(
            pan=1.5,
            lift=1.5,
            elbow=1.0,
            center_x=0.5,
            hand_roll=1.0,
            hand_open=1.0,
        )

    def test_deadband(self):
        self.assertEqual(signed_deadband(0.02, 0.05), 0.0)
        self.assertAlmostEqual(signed_deadband(0.08, 0.05), 0.03)
        self.assertAlmostEqual(signed_deadband(-0.08, 0.05), -0.03)

    def test_neutral_keeps_non_gripper_startup_pose(self):
        action = self.controller.compute(self.neutral, self.neutral)
        for joint in (
            "shoulder_pan.pos",
            "shoulder_lift.pos",
            "elbow_flex.pos",
            "wrist_flex.pos",
            "wrist_roll.pos",
        ):
            self.assertEqual(action[joint], self.initial[joint])
        # ID6 intentionally uses an absolute open/close mapping rather than a
        # neutral-centred delta.  hand_open=0.5 maps toward 75 and is EMA-smoothed.
        self.assertAlmostEqual(action["gripper.pos"], 63.75)

    def test_two_base_joints_never_move(self):
        for _ in range(25):
            action = self.controller.compute(self.moving_features(), self.neutral)
            self.assertEqual(action["shoulder_pan.pos"], self.initial["shoulder_pan.pos"])
            self.assertEqual(action["shoulder_lift.pos"], self.initial["shoulder_lift.pos"])

    def test_each_control_feature_drives_its_expected_joint(self):
        cases = (
            ("elbow", "elbow_flex.pos"),
            ("lift", "wrist_flex.pos"),
            ("hand_roll", "wrist_roll.pos"),
            ("hand_open", "gripper.pos"),
        )
        for feature_name, expected_joint in cases:
            controller = MotionController(self.initial, self.config.mapping, self.config.safety)
            values = {
                "pan": self.neutral.pan,
                "lift": self.neutral.lift,
                "elbow": self.neutral.elbow,
                "center_x": self.neutral.center_x,
                "hand_roll": self.neutral.hand_roll,
                "hand_open": self.neutral.hand_open,
            }
            values[feature_name] += 0.5
            action = controller.compute(
                PoseFeatures(**values),
                self.neutral,
                frozenset({feature_name}),
            )
            changed = {
                joint
                for joint in self.config.mapping.controlled_joints
                if action[joint] != self.initial[joint]
            }
            self.assertEqual(changed, {expected_joint})

    def test_invalid_channel_holds_while_another_channel_moves(self):
        first = self.controller.compute(
            self.moving_features(),
            self.neutral,
            frozenset({"hand_roll"}),
        )
        held_gripper = first["gripper.pos"]
        held_elbow = first["elbow_flex.pos"]
        second = self.controller.compute(
            self.moving_features(),
            self.neutral,
            frozenset({"hand_open"}),
        )
        self.assertEqual(second["wrist_roll.pos"], first["wrist_roll.pos"])
        self.assertEqual(second["elbow_flex.pos"], held_elbow)
        self.assertNotEqual(second["gripper.pos"], held_gripper)

    def test_hand_roll_wraparound_takes_short_path(self):
        neutral = PoseFeatures(0.0, 0.0, 0.5, 0.5, hand_roll=0.98, hand_open=0.5)
        observed = PoseFeatures(0.0, 0.0, 0.5, 0.5, hand_roll=-0.98, hand_open=0.5)
        action = self.controller.compute(observed, neutral, frozenset({"hand_roll"}))
        self.assertLess(abs(action["wrist_roll.pos"] - self.initial["wrist_roll.pos"]), 1.0)

    def test_per_cycle_step_is_limited(self):
        previous = self.controller.hold()
        for _ in range(10):
            action = self.controller.compute(self.moving_features(), self.neutral)
            for joint in self.config.mapping.controlled_joints:
                max_step = (
                    max(self.config.safety.max_step_per_cycle, 25.0)
                    if joint == "gripper.pos"
                    else self.config.safety.max_step_per_cycle
                )
                self.assertLessEqual(
                    abs(action[joint] - previous[joint]),
                    max_step + 1e-9,
                )
            previous = action

    def test_relative_limits_are_never_exceeded(self):
        action = None
        for _ in range(200):
            action = self.controller.compute(self.moving_features(), self.neutral)
        for joint, limit in self.config.safety.relative_limits.items():
            self.assertLessEqual(abs(action[joint] - self.initial[joint]), limit + 1e-9)


if __name__ == "__main__":
    unittest.main()

