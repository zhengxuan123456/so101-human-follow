import unittest

from so101_human_follow.app import build_parser
from so101_human_follow.config import ALL_JOINTS, load_config
from so101_human_follow.web_app import build_parser as build_web_parser


class ConfigAndGateTest(unittest.TestCase):
    def test_default_cli_never_enables_robot(self):
        args = build_parser().parse_args([])
        self.assertFalse(args.enable_robot)

    def test_default_web_server_never_enables_robot(self):
        args = build_web_parser().parse_args([])
        self.assertFalse(args.enable_robot)

    def test_default_configuration_covers_six_joints(self):
        config = load_config("config.json")
        controlled = set(config.mapping.controlled_joints)
        fixed = set(config.mapping.fixed_joints.names)
        self.assertEqual(controlled | fixed, set(ALL_JOINTS))
        self.assertEqual(
            fixed,
            {"shoulder_pan.pos", "shoulder_lift.pos"},
        )
        self.assertEqual(
            controlled,
            {"elbow_flex.pos", "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"},
        )
        self.assertEqual(config.mapping.fixed_joints.mode, "startup")


if __name__ == "__main__":
    unittest.main()

