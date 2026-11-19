import unittest

from so101_human_follow.agent_actions import build_agent_motion_plan
from so101_human_follow.config import ALL_JOINTS


ELBOW = "elbow_flex.pos"
WRIST_ROLL = "wrist_roll.pos"
GRIPPER = "gripper.pos"


class AgentBoundaryGesturesTest(unittest.TestCase):
    def setUp(self):
        self.positions = dict.fromkeys(ALL_JOINTS, 0.0)
        self.positions[GRIPPER] = 50.0
        self.limits = {joint: (-70.0, 70.0) for joint in ALL_JOINTS}
        self.limits["wrist_flex.pos"] = (-100.0, 100.0)
        self.limits[WRIST_ROLL] = (-100.0, 100.0)
        self.limits[GRIPPER] = (0.0, 100.0)

    def build(self, name, arguments=None, *, positions=None, home=None):
        current = dict(self.positions if positions is None else positions)
        return build_agent_motion_plan(
            [{"name": name, "arguments": arguments or {}}],
            current_positions=current,
            home_positions=dict(current if home is None else home),
            limits=self.limits,
            fixed_positions={},
            gripper_open_is_high=True,
            pet_restore_positions=None,
        )

    def assert_bounded_active_targets(self, plan):
        for frame in plan.keyframes:
            for joint in frame.active_joints:
                low, high = self.limits[joint]
                self.assertGreaterEqual(frame.targets[joint], low)
                self.assertLessEqual(frame.targets[joint], high)

    def assert_untouched(self, plan, joint, measured):
        for frame in plan.keyframes:
            self.assertNotIn(joint, frame.active_joints)
            self.assertEqual(frame.targets[joint], measured)
        self.assert_bounded_active_targets(plan)

    def test_greeting_preserves_logged_elbow_feedback_and_still_moves_gripper(self):
        measured = -70.09936766034326
        positions = {**self.positions, ELBOW: measured}
        original_limits = dict(self.limits)
        plan = self.build("perform_greeting", {"count": 1}, positions=positions)
        self.assert_untouched(plan, ELBOW, measured)
        self.assertTrue(any(
            GRIPPER in frame.active_joints and frame.targets[GRIPPER] != positions[GRIPPER]
            for frame in plan.keyframes
        ))
        self.assertEqual(plan.keyframes[-1].targets[GRIPPER], positions[GRIPPER])
        self.assertEqual(positions[ELBOW], measured)
        self.assertEqual(self.limits, original_limits)

    def test_optional_elbow_does_not_block_independent_primary_actions_at_either_edge(self):
        primary_joints = {
            "perform_greeting": GRIPPER,
            "open_gripper": GRIPPER,
            "close_gripper": GRIPPER,
            "wave_wrist": WRIST_ROLL,
            "happy_wiggle": WRIST_ROLL,
        }
        for measured in (-70.189701897, 70.189701897):
            positions = {**self.positions, ELBOW: measured}
            for name, primary in primary_joints.items():
                with self.subTest(measured=measured, action=name):
                    plan = self.build(name, {"count": 1}, positions=positions)
                    self.assert_untouched(plan, ELBOW, measured)
                    self.assertTrue(any(
                        primary in frame.active_joints and frame.targets[primary] != positions[primary]
                        for frame in plan.keyframes
                    ))

    def test_exact_elbow_endpoints_remain_available_and_restore_exactly(self):
        for measured in (-70.0, 70.0):
            with self.subTest(measured=measured):
                plan = self.build("perform_greeting", positions={**self.positions, ELBOW: measured})
                self.assertTrue(any(ELBOW in frame.active_joints for frame in plan.keyframes))
                self.assertEqual(plan.keyframes[-1].targets[ELBOW], measured)
                self.assert_bounded_active_targets(plan)

    def test_response_categories_use_safe_axes_and_restore_only_those_axes(self):
        for measured in (-70.09936766034326, 70.09936766034326):
            positions = {**self.positions, ELBOW: measured}
            for category in ("neutral", "happy", "curious", "cute", "encouraging"):
                with self.subTest(measured=measured, category=category):
                    plan = self.build("perform_response_gesture", {"category": category}, positions=positions)
                    self.assert_untouched(plan, ELBOW, measured)
                    touched = {joint for frame in plan.keyframes for joint in frame.active_joints}
                    self.assertTrue(touched)
                    self.assertEqual(set(plan.keyframes[-1].active_joints), touched)
                    for joint in touched:
                        self.assertEqual(plan.keyframes[-1].targets[joint], positions[joint])

    def test_response_with_no_safe_axes_reports_failure(self):
        positions = {joint: high + 0.1 for joint, (_, high) in self.limits.items()}
        with self.assertRaisesRegex(ValueError, "No response gesture can safely return"):
            self.build("perform_response_gesture", positions=positions)

    def test_unsafe_required_head_axes_are_not_silently_omitted(self):
        for measured in (-70.1, 70.1):
            for name in ("nod", "enter_pet_pose", "leave_pet_pose"):
                with self.subTest(measured=measured, action=name):
                    with self.assertRaisesRegex(ValueError, ELBOW):
                        self.build(name, positions={**self.positions, ELBOW: measured})

    def test_unsafe_primary_wrist_is_not_silently_omitted(self):
        for measured in (-100.1, 100.1):
            for name in ("wave_wrist", "happy_wiggle"):
                with self.subTest(measured=measured, action=name):
                    with self.assertRaisesRegex(ValueError, WRIST_ROLL):
                        self.build(name, positions={**self.positions, WRIST_ROLL: measured})

    def test_greeting_cannot_manufacture_an_in_range_gripper_return(self):
        for measured in (-0.1, 100.1):
            with self.subTest(measured=measured):
                with self.assertRaisesRegex(ValueError, GRIPPER):
                    self.build("perform_greeting", positions={**self.positions, GRIPPER: measured})

    def test_explicit_targets_remain_strict_at_both_edges(self):
        for target in (-70.1, 70.1):
            with self.subTest(target=target):
                with self.assertRaisesRegex(ValueError, ELBOW):
                    self.build("move_joints", {"targets": {ELBOW: target}})

    def test_saved_action_targets_remain_strict(self):
        with self.assertRaisesRegex(ValueError, ELBOW):
            self.build("play_saved_action", {"frames": [{"targets": {ELBOW: -70.1}}]})

    def test_connection_return_remains_exact_and_rejects_unsafe_saved_pose(self):
        for target in (-70.1, 70.1):
            with self.subTest(target=target):
                home = {**self.positions, ELBOW: target}
                with self.assertRaisesRegex(ValueError, ELBOW):
                    self.build("return_to_neutral", home=home)

    def test_recorded_trajectory_targets_remain_strict(self):
        for target in (-70.1, 70.1):
            with self.subTest(target=target):
                samples = [
                    {"time_s": 0.0, "positions": dict(self.positions)},
                    {"time_s": 1.0, "positions": {**self.positions, ELBOW: target}},
                ]
                with self.assertRaisesRegex(ValueError, ELBOW):
                    self.build("play_follower_trajectory", {"samples": samples})


if __name__ == "__main__":
    unittest.main()
