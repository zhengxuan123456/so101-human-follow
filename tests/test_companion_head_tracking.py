import math
import unittest

from so101_human_follow.companion import CompanionMotion, head_target
from so101_human_follow.config import ALL_JOINTS
from so101_human_follow.pose_detector import PoseDetection
from so101_human_follow.pose_math import Landmark


YAW = "shoulder_pan.pos"
SHOULDER = "shoulder_lift.pos"
ELBOW = "elbow_flex.pos"
WRIST = "wrist_flex.pos"
GRIPPER = "gripper.pos"


class CompanionHeadTrackingTest(unittest.TestCase):
    def setUp(self):
        self.positions = dict.fromkeys(ALL_JOINTS, 0.0)
        self.positions[GRIPPER] = 50.0
        self.limits = {joint: (-70.0, 70.0) for joint in ALL_JOINTS}
        self.limits[WRIST] = (-100.0, 100.0)
        self.limits["wrist_roll.pos"] = (-100.0, 100.0)
        self.limits[GRIPPER] = (0.0, 100.0)

    def test_independent_head_target_needs_no_body_or_arm_features(self):
        detection = PoseDetection(
            features=None, landmarks=(), people_count=0, reason="no person",
            head_center=(0.23, 0.81), head_confidence=0.9,
        )
        self.assertEqual(head_target(detection), (0.23, 0.81))

    def test_independent_head_target_takes_priority_over_pose_face_points(self):
        detection = PoseDetection(
            features=None, landmarks=(Landmark(0.7, 0.3),) * 33, people_count=1, reason="partial pose",
            head_center=(0.25, 0.65), head_confidence=0.8,
        )
        self.assertEqual(head_target(detection), (0.25, 0.65))

    def test_invalid_independent_head_target_is_not_used(self):
        for center, confidence in (
            ((0.2, 0.3), 0.54), ((0.2, 0.3), math.nan),
            ((0.2, 0.3), math.inf), ((math.nan, 0.3), 0.9),
            ((0.2, math.inf), 0.9), ((-0.01, 0.5), 0.9),
            ((0.5, 1.01), 0.9), ((0.2, 0.3, 0.4), 0.9),
        ):
            with self.subTest(center=center, confidence=confidence):
                detection = PoseDetection(
                    features=None, landmarks=(), people_count=0, reason="no person",
                    head_center=center, head_confidence=confidence,
                )
                self.assertIsNone(head_target(detection))

    def test_legacy_pose_head_points_remain_a_valid_fallback(self):
        points = [Landmark(0.3, 0.2)] * 33
        points[2], points[5] = Landmark(0.28, 0.2), Landmark(0.32, 0.2)
        detection = PoseDetection(None, tuple(points), 1, "no reliable control channel")
        x, y = head_target(detection)
        self.assertAlmostEqual(x, 0.3)
        self.assertAlmostEqual(y, 0.2)

    def test_head_near_image_edge_quiets_body_and_prioritizes_camera_motion(self):
        motion = CompanionMotion(self.positions, self.limits)
        for _ in range(80):
            motion.step(0.05, face_target=(0.5, 0.5), max_step=0.8)
        for _ in range(120):
            targets = motion.step(0.05, face_target=(0.9, 0.9), max_step=0.8, gaze_delta=(0.4, 0.4))
        self.assertLess(abs(targets[SHOULDER]), 0.01)
        self.assertLess(abs(targets[ELBOW]), 0.01)
        self.assertGreater(targets[YAW], 10.0)
        self.assertGreater(targets[WRIST], 11.0)

    def test_centered_head_restores_visible_body_motion(self):
        motion = CompanionMotion(self.positions, self.limits)
        for _ in range(120):
            motion.step(0.05, face_target=(0.9, 0.9), max_step=0.8)
        shoulder = []
        elbow = []
        for _ in range(150):
            targets = motion.step(0.05, face_target=(0.5, 0.5), max_step=0.8)
            shoulder.append(targets[SHOULDER])
            elbow.append(targets[ELBOW])
        self.assertGreater(max(shoulder) - min(shoulder), 15.0)
        self.assertGreater(max(elbow) - min(elbow), 20.0)

    def test_pitch_correction_uses_existing_travel_at_either_image_edge(self):
        for target_y, direction in ((0.95, 1.0), (0.05, -1.0)):
            with self.subTest(target_y=target_y):
                motion = CompanionMotion(self.positions, self.limits)
                previous = 0.0
                for _ in range(120):
                    targets = motion.step(0.05, face_target=(0.5, target_y), max_step=0.8,
                                          gaze_delta=(0.0, direction * 0.4))
                    self.assertLessEqual(abs(targets[WRIST] - previous), 0.800001)
                    self.assertLessEqual(abs(targets[WRIST]), 24.0)
                    previous = targets[WRIST]
                self.assertGreater(direction * targets[WRIST], 11.0)

    def test_repeated_head_corrections_stay_inside_joint_limits_and_step_caps(self):
        self.positions[ELBOW] = -69.9
        self.positions[WRIST] = 98.9
        motion = CompanionMotion(self.positions, self.limits)
        previous = dict(self.positions)
        for index in range(600):
            center = (0.95, 0.95) if index % 180 < 90 else (0.05, 0.05)
            targets = motion.step(0.05, face_target=center, max_step=0.8)
            for joint, value in targets.items():
                self.assertTrue(self.limits[joint][0] <= value <= self.limits[joint][1])
                self.assertLessEqual(abs(value - previous[joint]), 0.800001)
                previous[joint] = value
        self.assertNotIn("wrist_roll.pos", targets)

    def test_target_loss_does_not_issue_fresh_body_goals(self):
        motion = CompanionMotion(self.positions, self.limits)
        for _ in range(30):
            motion.step(0.05, face_target=(0.8, 0.8), max_step=0.8)
        self.assertEqual(motion.step(0.05, face_target=None, max_step=0.8), {})

    def test_saturated_gaze_reports_its_limit_and_clears_when_head_is_centered(self):
        motion = CompanionMotion(self.positions, self.limits)
        motion.step(0.05, face_target=(0.95, 0.95), max_step=0.8, gaze_delta=(0.4, 0.4))
        self.assertEqual(motion.gaze_limit_reason, "")
        for _ in range(120):
            motion.step(0.05, face_target=(0.95, 0.95), max_step=0.8, gaze_delta=(0.4, 0.4))
        self.assertIn("底座", motion.gaze_limit_reason)
        self.assertIn("腕部", motion.gaze_limit_reason)
        self.assertIn("允许范围", motion.gaze_limit_reason)
        motion.step(0.05, face_target=(0.5, 0.5), max_step=0.8)
        self.assertEqual(motion.gaze_limit_reason, "")

    def test_no_camera_direction_is_guessed_without_calibrated_delta(self):
        motion = CompanionMotion(self.positions, self.limits)
        for _ in range(50):
            targets = motion.step(0.05, face_target=(0.95, 0.95), max_step=0.8)
        self.assertEqual(targets[YAW], self.positions[YAW])
        self.assertEqual(targets[WRIST], self.positions[WRIST])

    def test_coupled_camera_reports_yaw_limit_for_a_vertical_image_error(self):
        motion = CompanionMotion(self.positions, self.limits)
        for _ in range(60):
            motion.step(0.05, face_target=(0.5, 0.7), gaze_delta=(0.5, 0.0), max_step=0.8)
        self.assertIn("底座", motion.gaze_limit_reason)

    def test_collapsed_pose_points_do_not_bypass_head_validation(self):
        detection = PoseDetection(None, (Landmark(0.3, 0.2),) * 33, 1, "partial pose")
        self.assertIsNone(head_target(detection))

    def test_new_near_edge_rest_is_not_forced_to_old_continuous_motion_margin(self):
        self.positions[YAW] = self.positions[WRIST] = -99.5
        limits = {joint: ((0.0, 100.0) if joint == GRIPPER else (-100.0, 100.0)) for joint in ALL_JOINTS}
        motion = CompanionMotion(self.positions, limits)
        target = motion.step(0.05, face_target=(0.5, 0.5), amplitude=0.0, gaze_delta=(0.0, 0.0), max_step=0.2)
        self.assertAlmostEqual(target[YAW], -99.5)
        self.assertAlmostEqual(target[WRIST], -99.5)
        self.assertEqual(motion.gaze_bounds[YAW][0], -99.5)
        self.assertEqual(motion.gaze_bounds[WRIST][0], -99.5)

    def test_vertical_search_is_opt_in_and_never_moves_body_or_gripper(self):
        motion = CompanionMotion(self.positions, self.limits)
        for _ in range(30):
            targets = motion.step(0.05, face_target=None, search=True, hand_open=1.0)
            self.assertEqual(set(targets), {YAW})
        for _ in range(30):
            targets = motion.step(
                0.05, face_target=None, search=True, search_vertical=True,
                hand_open=1.0, max_step=0.8,
            )
            self.assertEqual(set(targets), {YAW, WRIST})
        targets = motion.step(0.05, face_target=None, search=True, search_vertical=False)
        self.assertEqual(set(targets), {YAW})
        self.assertFalse(motion.gripper_active)

    def test_vertical_search_reverses_inside_existing_wrist_travel_and_speed_caps(self):
        motion = CompanionMotion(self.positions, self.limits)
        previous = dict(self.positions)
        samples = []
        for _ in range(400):
            targets = motion.step(
                0.05, face_target=None, search=True, search_vertical=True, max_step=0.25,
            )
            for joint, value in targets.items():
                self.assertLessEqual(abs(value - previous[joint]), 0.250001)
                self.assertTrue(self.limits[joint][0] <= value <= self.limits[joint][1])
                previous[joint] = value
            self.assertLessEqual(abs(targets[WRIST]), 24.0)
            samples.append(targets[WRIST])
        self.assertIn(24.0, samples)
        self.assertIn(-24.0, samples)
        self.assertTrue(any(after > before for before, after in zip(samples, samples[1:])))
        self.assertTrue(any(after < before for before, after in zip(samples, samples[1:])))

        motion = CompanionMotion(self.positions, self.limits)
        first = motion.step(0.05, face_target=None, search=True, search_vertical=True)
        self.assertGreater(first[WRIST], 0.0)
        self.assertLessEqual(first[WRIST] / 0.05, 9.0)

    def test_vertical_search_starts_toward_the_larger_available_wrist_span(self):
        self.positions[WRIST] = 92.0
        motion = CompanionMotion(self.positions, self.limits)
        previous = 92.0
        for _ in range(300):
            targets = motion.step(
                0.05, face_target=None, search=True, search_vertical=True, max_step=0.25,
            )
            self.assertTrue(68.0 <= targets[WRIST] <= 94.0)
            self.assertLessEqual(abs(targets[WRIST] - previous), 0.250001)
            previous = targets[WRIST]
        self.assertNotEqual(previous, self.positions[WRIST])

    def test_explicit_vertical_search_can_work_with_an_out_of_range_base(self):
        self.positions[YAW] = -70.1
        motion = CompanionMotion(self.positions, self.limits)
        self.assertFalse(motion.tracks_yaw)
        self.assertTrue(motion.tracks_pitch)
        self.assertFalse(motion.can_search())
        self.assertTrue(motion.can_search(vertical=True))
        self.assertEqual(motion.step(0.05, face_target=None, search=True), {})
        targets = motion.step(0.05, face_target=None, search=True, search_vertical=True)
        self.assertEqual(set(targets), {WRIST})
        self.assertEqual(self.positions[YAW], -70.1)

    def test_vertical_search_omits_a_wrist_outside_its_envelope(self):
        self.positions[WRIST] = 100.1
        motion = CompanionMotion(self.positions, self.limits)
        self.assertFalse(motion.tracks_pitch)
        self.assertTrue(motion.can_search(vertical=True))
        self.assertEqual(set(motion.step(
            0.05, face_target=None, search=True, search_vertical=True,
        )), {YAW})
        self.positions[YAW] = -70.1
        motion = CompanionMotion(self.positions, self.limits)
        self.assertFalse(motion.can_search(vertical=True))
        self.assertEqual(motion.step(0.05, face_target=None, search=True, search_vertical=True), {})

    def test_search_to_head_tracking_keeps_the_actual_wrist_search_pose(self):
        motion = CompanionMotion(self.positions, self.limits)
        for _ in range(40):
            searched = motion.step(0.05, face_target=None, search=True, search_vertical=True, max_step=0.8)
        # The serial loop reads actual encoders when acquiring the head. Do not
        # use the old zero-pitch origin or a slightly unfinished search command.
        measured = {**self.positions, **searched, WRIST: searched[WRIST] - 0.15}
        motion.resume_from(measured)
        targets = motion.step(0.05, face_target=(0.5, 0.5), amplitude=0.0, max_step=0.8)
        self.assertAlmostEqual(targets[WRIST], measured[WRIST])
        self.assertAlmostEqual(targets[YAW], measured[YAW])
        self.assertGreater(targets[WRIST], 5.0)


if __name__ == "__main__":
    unittest.main()
