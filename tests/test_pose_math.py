import math
import unittest

from so101_human_follow.pose_math import (
    Landmark,
    PoseFeatures,
    extract_arm_feature_channels,
    extract_arm_features,
    extract_hand_feature_channels,
    extract_hand_features,
    median_features,
)


def pose_points() -> list[Landmark]:
    points = [Landmark(0.5, 0.5) for _ in range(33)]
    points[11] = Landmark(0.4, 0.4)  # left shoulder
    points[12] = Landmark(0.6, 0.4)  # right shoulder
    points[14] = Landmark(0.8, 0.4)  # right elbow
    points[16] = Landmark(1.0, 0.4)  # right wrist: straight arm
    return points


def open_hand_points() -> list[Landmark]:
    points = [Landmark(0.5, 0.5) for _ in range(21)]
    points[0] = Landmark(0.5, 0.8)  # wrist
    for mcp, pip, tip, x in (
        (5, 6, 8, 0.35),
        (9, 10, 12, 0.45),
        (13, 14, 16, 0.55),
        (17, 18, 20, 0.65),
    ):
        points[mcp] = Landmark(x, 0.6)
        points[pip] = Landmark(x, 0.42)
        points[tip] = Landmark(x, 0.2)
    return points


def rotate_points(points: list[Landmark], angle: float) -> list[Landmark]:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    result = []
    for point in points:
        dx = point.x - 0.5
        dy = point.y - 0.5
        result.append(
            Landmark(
                0.5 + dx * cosine - dy * sine,
                0.5 + dx * sine + dy * cosine,
                point.z,
                point.visibility,
                point.presence,
            )
        )
    return result


class PoseMathTest(unittest.TestCase):
    def test_straight_right_arm(self):
        features = extract_arm_features(pose_points(), side="right")
        self.assertIsNotNone(features)
        self.assertAlmostEqual(features.pan, 1.0)
        self.assertAlmostEqual(features.lift, 0.0)
        self.assertAlmostEqual(features.elbow, 0.0)

    def test_ninety_degree_elbow_is_half_flexed(self):
        points = pose_points()
        points[16] = Landmark(0.8, 0.2)
        features = extract_arm_features(points, side="right")
        self.assertIsNotNone(features)
        self.assertAlmostEqual(features.elbow, 0.5, places=6)

    def test_low_visibility_is_rejected(self):
        points = pose_points()
        points[14] = Landmark(0.8, 0.4, visibility=0.2)
        self.assertIsNone(extract_arm_features(points, side="right", min_visibility=0.7))

    def test_hidden_wrist_still_allows_lift_channel(self):
        points = pose_points()
        points[16] = Landmark(1.0, 0.4, visibility=0.1)
        _, valid = extract_arm_feature_channels(points, side="right", min_visibility=0.7)
        self.assertIn("lift", valid)
        self.assertNotIn("elbow", valid)

    def test_nan_is_rejected(self):
        points = pose_points()
        points[16] = Landmark(math.nan, 0.4)
        self.assertIsNone(extract_arm_features(points))

    def test_single_visible_shoulder_still_allows_arm_features(self):
        points = pose_points()
        points[11] = Landmark(0.6, 0.4)
        features = extract_arm_features(points)
        self.assertIsNotNone(features)
        self.assertAlmostEqual(features.elbow, 0.0)

    def test_invalid_aspect_ratio_is_rejected(self):
        self.assertIsNone(extract_arm_features(pose_points(), aspect_ratio=0.0))

    def test_four_by_three_aspect_preserves_right_angle(self):
        points = pose_points()
        points[16] = Landmark(0.8, 0.2)
        features = extract_arm_features(points, aspect_ratio=4 / 3)
        self.assertIsNotNone(features)
        self.assertAlmostEqual(features.elbow, 0.5, places=6)

    def test_median_neutral(self):
        first = extract_arm_features(pose_points())
        second_points = pose_points()
        second_points[14] = Landmark(0.7, 0.3)
        second = extract_arm_features(second_points)
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        result = median_features([first, second])
        self.assertTrue(math.isfinite(result.pan))
        self.assertTrue(math.isfinite(result.lift))
        self.assertTrue(math.isfinite(result.elbow))

    def test_pose_features_keep_backward_compatible_hand_defaults(self):
        features = PoseFeatures(pan=0.1, lift=0.2, elbow=0.3, center_x=0.4)
        self.assertEqual(features.hand_roll, 0.0)
        self.assertEqual(features.hand_open, 0.0)

    def test_open_hand_has_neutral_roll_and_high_openness(self):
        features = extract_hand_features(open_hand_points())
        self.assertIsNotNone(features)
        self.assertAlmostEqual(features.hand_roll, 0.0, places=6)
        self.assertGreater(features.hand_open, 0.95)

    def test_in_plane_hand_rotation_is_normalized(self):
        features = extract_hand_features(rotate_points(open_hand_points(), math.pi / 4))
        self.assertIsNotNone(features)
        self.assertAlmostEqual(features.hand_roll, -0.5, places=6)

    def test_curled_fingers_are_less_open(self):
        open_features = extract_hand_features(open_hand_points())
        curled = open_hand_points()
        for pip, tip in ((6, 8), (10, 12), (14, 16), (18, 20)):
            curled[tip] = Landmark(curled[pip].x + 0.12, curled[pip].y + 0.02)
        curled_features = extract_hand_features(curled)
        self.assertIsNotNone(open_features)
        self.assertIsNotNone(curled_features)
        self.assertLess(curled_features.hand_open, open_features.hand_open - 0.4)

    def test_hand_low_visibility_is_rejected(self):
        points = open_hand_points()
        points[8] = Landmark(points[8].x, points[8].y, visibility=0.1)
        self.assertIsNone(extract_hand_features(points, min_visibility=0.5))

    def test_hidden_fingertip_still_allows_hand_roll_channel(self):
        points = open_hand_points()
        points[8] = Landmark(points[8].x, points[8].y, visibility=0.1)
        _, valid = extract_hand_feature_channels(points, min_visibility=0.5)
        self.assertIn("hand_roll", valid)
        # The robust channel accepts the remaining fingers and fingertip-distance
        # evidence instead of discarding ID6 because one fingertip is occluded.
        self.assertIn("hand_open", valid)

    def test_hand_nan_is_rejected(self):
        points = open_hand_points()
        points[17] = Landmark(math.nan, points[17].y)
        self.assertIsNone(extract_hand_features(points))

    def test_degenerate_hand_is_rejected(self):
        points = [Landmark(0.5, 0.5) for _ in range(21)]
        self.assertIsNone(extract_hand_features(points))

    def test_median_neutral_includes_hand_features(self):
        first = PoseFeatures(0.0, 0.0, 0.0, 0.5, hand_roll=-0.4, hand_open=0.2)
        second = PoseFeatures(0.0, 0.0, 0.0, 0.5, hand_roll=0.6, hand_open=0.8)
        result = median_features([first, second])
        self.assertAlmostEqual(result.hand_roll, 0.1)
        self.assertAlmostEqual(result.hand_open, 0.5)


if __name__ == "__main__":
    unittest.main()

