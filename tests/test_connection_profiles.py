"""Connection may observe a startup pose; motion needs separate authorization."""
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from so101_human_follow.config import load_config
from so101_human_follow.pose_math import PoseFeatures
from so101_human_follow.robot_adapter import preview_initial_positions
from so101_human_follow.web_engine import ControlEngine, ControlError


class ConfiguredConnectionTest(unittest.TestCase):
    def setUp(self):
        self.config = load_config("config.example.json")
        self.engine = ControlEngine(self.config, robot_allowed=True)
        self.client = "configured_connection_test_operator"
        self.engine.heartbeat(self.client)
        self.positions = preview_initial_positions()

    def connect(self):
        adapter = SimpleNamespace(
            connected=True, connect=lambda: dict(self.positions),
            calibration_fingerprint=lambda: "configured-fingerprint",
            normalized_joint_limits=lambda: {joint: ((0.0, 100.0) if joint == "gripper.pos" else (-100.0, 100.0))
                                             for joint in self.positions},
            disconnect=lambda: None, read_positions=lambda: dict(self.positions),
        )
        with patch("so101_human_follow.web_engine.SO101RobotAdapter", return_value=adapter):
            self.engine._connect_robot("123", client_id=self.client)
        return adapter

    def make_arm_ready(self):
        features = PoseFeatures(0.0, 0.0, 0.5, 0.5)
        self.engine._neutral = self.engine._latest_features = features
        self.engine._last_pose_time = time.monotonic()
        self.engine._valid_streak = self.config.safety.valid_frames_before_arm
        self.engine._preview_state = "RUNNING"

    def test_boundary_and_near_boundary_startup_keeps_real_pose_without_widening_limits(self):
        for position in (-100.0, -100.12, 100.12):
            with self.subTest(position=position):
                self.engine = ControlEngine(self.config, robot_allowed=True)
                self.engine.heartbeat(self.client)
                self.positions["wrist_flex.pos"] = position
                self.connect()
                self.assertEqual(self.engine._robot_state, "CONNECTED_DISARMED")
                self.assertEqual(self.engine._connection_positions["wrist_flex.pos"], position)
                self.assertEqual(self.engine._safe_profile.startup_positions["wrist_flex.pos"], position)
                self.assertEqual(self.engine.agent_joint_limits()["wrist_flex.pos"], (-100.0, 100.0))
                self.assertEqual(self.engine._last_action["wrist_flex.pos"], position)
                self.assertFalse(self.engine._armed)

    def test_startup_sanity_is_still_enforced(self):
        self.positions["wrist_flex.pos"] = 201.0
        with self.assertRaisesRegex(ValueError, "startup range"):
            self.connect()
        self.assertIsNone(self.engine._robot)

    def test_arm_rejects_current_outside_working_range_then_accepts_safe_feedback(self):
        adapter = self.connect()
        self.make_arm_ready()
        self.positions["wrist_flex.pos"] = -100.12
        with self.assertRaisesRegex(ControlError, "动作范围"):
            self.engine._arm("123", client_id=self.client, submitted_epoch=self.engine._epoch)
        self.assertFalse(self.engine._armed)
        self.positions["wrist_flex.pos"] = -90.0
        self.engine._arm("123", client_id=self.client, submitted_epoch=self.engine._epoch)
        self.assertTrue(self.engine._armed)
        self.assertEqual(self.engine._controller.previous["wrist_flex.pos"], -90.0)
        self.assertIs(self.engine._robot, adapter)

    def test_stop_during_arm_feedback_cannot_be_cleared(self):
        adapter = self.connect()
        self.make_arm_ready()
        def interrupted_read():
            self.engine._stop_sequence += 1
            self.engine._motion_stop.set()
            return dict(self.positions)
        adapter.read_positions = interrupted_read
        with self.assertRaisesRegex(ControlError, "cancelled"):
            self.engine._arm("123", client_id=self.client, submitted_epoch=self.engine._epoch)
        self.assertFalse(self.engine._armed)
        self.assertTrue(self.engine._motion_stop.is_set())


if __name__ == "__main__":
    unittest.main()
