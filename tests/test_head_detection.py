import sys
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from so101_human_follow.config import load_config
from so101_human_follow.pose_detector import (
    MediaPipePoseDetector, PoseDetection, _pose_head, _select_head, annotate_frame,
)
from so101_human_follow.pose_math import Landmark


def face(x=240, y=120, width=160, height=180, score=0.9):
    return SimpleNamespace(
        bounding_box=SimpleNamespace(origin_x=x, origin_y=y, width=width, height=height),
        categories=[SimpleNamespace(score=score)],
    )


def pose_head(x=0.65, y=0.3):
    points = [Landmark(0.5, 0.5, visibility=0.1, presence=0.1) for _ in range(33)]
    for index, dx, dy in ((0, 0, 0), (2, -0.03, -0.025), (5, 0.03, -0.025),
                          (7, -0.055, 0), (8, 0.055, 0), (9, -0.02, 0.04), (10, 0.02, 0.04)):
        points[index] = Landmark(x + dx, y + dy, visibility=0.95, presence=0.95)
    return points


class HeadSelectionTest(unittest.TestCase):
    def test_legacy_configuration_defaults_face_model_next_to_pose_model(self):
        raw = json.loads(Path("config.example.json").read_text(encoding="utf-8"))
        raw["pose"].pop("face_model_path", None)
        raw["pose"]["model_path"] = "relative_models/body.task"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            self.assertEqual(load_config(path).pose.face_model_path,
                             Path(directory) / "relative_models" / "blaze_face_short_range.tflite")
            raw["pose"]["face_model_path"] = "other_models/front.tflite"
            path.write_text(json.dumps(raw), encoding="utf-8")
            self.assertEqual(load_config(path).pose.face_model_path,
                             Path(directory) / "other_models" / "front.tflite")

    def test_single_face_works_without_pose_or_hand(self):
        head = _select_head(SimpleNamespace(detections=[face()]), (480, 640, 3), ())
        self.assertEqual(head.source, "face")
        self.assertEqual(head.center, (0.5, 0.4375))
        self.assertEqual(head.box, (0.375, 0.25, 0.625, 0.625))
        self.assertEqual(head.confidence, 0.9)

    def test_largest_obvious_face_beats_small_poster_faces(self):
        head = _select_head(SimpleNamespace(detections=[
            face(10, 10, 40, 50, 0.99), face(240, 100, 180, 200, 0.85),
        ]), (480, 640, 3), ())
        self.assertEqual(head.source, "face")
        self.assertEqual(head.confidence, 0.85)
        self.assertGreater(head.box[2] - head.box[0], 0.25)

    def test_similar_faces_without_pose_association_are_ambiguous(self):
        head = _select_head(SimpleNamespace(detections=[
            face(40, 100, 120, 160), face(350, 80, 118, 156),
        ]), (480, 640, 3), ())
        self.assertEqual(head.source, "ambiguous")
        self.assertIsNone(head.center)
        self.assertIsNone(head.box)

    def test_reliable_pose_head_associates_the_correct_face(self):
        head = _select_head(SimpleNamespace(detections=[
            face(40, 100, 120, 160), face(350, 80, 118, 156),
        ]), (480, 640, 3), pose_head())
        self.assertEqual(head.source, "face")
        self.assertGreater(head.center[0], 0.6)

    def test_tiny_pose_associated_poster_does_not_beat_a_nearby_face(self):
        head = _select_head(SimpleNamespace(detections=[
            face(10, 10, 40, 50), face(240, 100, 180, 200),
        ]), (480, 640, 3), pose_head(0.05, 0.07))
        self.assertGreater(head.center[0], 0.4)

    def test_invalid_boxes_and_scores_are_rejected(self):
        invalid = [face(width=-10), face(width=0), face(x=900), face(x=-400, width=405),
                   face(score=float("nan")), face(score=0.54), face(score=1.1), face(width=float("inf"))]
        for candidate in invalid:
            with self.subTest(candidate=candidate):
                head = _select_head(SimpleNamespace(detections=[candidate]), (480, 640, 3), ())
                self.assertIsNone(head.center)

    def test_partial_face_box_is_clipped_to_frame(self):
        head = _select_head(SimpleNamespace(detections=[face(-20, 200, 160, 200)]), (480, 640, 3), ())
        self.assertEqual(head.box[0], 0.0)
        self.assertTrue(all(0 <= coordinate <= 1 for coordinate in head.box))

    def test_pose_head_fallback_ignores_missing_arm_channels(self):
        head = _pose_head(pose_head())
        self.assertEqual(head.source, "pose")
        self.assertAlmostEqual(head.center[0], 0.65)
        self.assertGreaterEqual(head.confidence, 0.55)
        for index in (0, 2, 5, 7, 8, 9, 10):
            points = [Landmark(0.5, 0.5, visibility=0.1) for _ in range(33)]
            points[index] = Landmark(float("nan"), 0.4)
            self.assertIsNone(_pose_head(points).center)

    def test_head_annotation_does_not_draw_unreliable_body_skeleton(self):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        detection = PoseDetection(
            None, tuple(Landmark(0.5, 0.5, visibility=0.1) for _ in range(33)),
            1, "no reliable control channel", head_center=(0.65, 0.3),
            head_box=(0.55, 0.15, 0.75, 0.45), head_confidence=0.9, head_source="face",
        )
        output = annotate_frame(frame, detection, [])
        green = (output[:, :, 1] > output[:, :, 0] * 1.5) & (output[:, :, 1] > output[:, :, 2] * 1.5)
        self.assertFalse(green.any())
        self.assertTrue((output[:, :, 0] > 200).any())
        self.assertFalse(frame.any())


class HeadDetectorIntegrationTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.pose_path, self.hand_path = self.directory / "pose.task", self.directory / "hand.task"
        self.pose_path.touch()
        self.hand_path.touch()
        self.settings = replace(load_config("config.example.json").pose, model_path=self.pose_path,
                                hand_model_path=self.hand_path, face_model_path=None)
        self.pose = Mock()
        self.pose.detect_for_video.return_value = SimpleNamespace(pose_landmarks=[])
        self.hand = Mock()
        self.hand.detect_for_video.return_value = SimpleNamespace(hand_landmarks=[], handedness=[])
        self.face = Mock()
        self.face.detect_for_video.return_value = SimpleNamespace(detections=[face()])
        options = lambda **kwargs: SimpleNamespace(**kwargs)
        vision = SimpleNamespace(
            RunningMode=SimpleNamespace(VIDEO="VIDEO"),
            PoseLandmarkerOptions=options, HandLandmarkerOptions=options, FaceDetectorOptions=options,
            PoseLandmarker=SimpleNamespace(create_from_options=Mock(return_value=self.pose)),
            HandLandmarker=SimpleNamespace(create_from_options=Mock(return_value=self.hand)),
            FaceDetector=SimpleNamespace(create_from_options=Mock(return_value=self.face)),
        )
        self.mp = SimpleNamespace(
            tasks=SimpleNamespace(vision=vision, BaseOptions=options),
            Image=options, ImageFormat=SimpleNamespace(SRGB="SRGB"),
        )
        module = patch.dict(sys.modules, {"mediapipe": self.mp})
        module.start()
        self.addCleanup(module.stop)

    def make_detector(self, face_model=True):
        if face_model:
            self.pose_path.with_name("blaze_face_short_range.tflite").touch()
        detector = MediaPipePoseDetector(self.settings)
        self.addCleanup(detector.close)
        return detector

    def test_face_only_detection_populates_head_without_changing_pose_count(self):
        self.pose_path.with_name("face_detection_yunet_2026may.onnx").touch()
        verifier = SimpleNamespace(verify=Mock(return_value=True), close=Mock())
        with patch("so101_human_follow.pose_detector._YuNetFaceVerifier", return_value=verifier):
            detector = self.make_detector()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        with patch("so101_human_follow.pose_detector.time.monotonic", side_effect=[10.0, 10.06, 10.12]):
            for _ in range(2):
                pending = detector.detect(frame)
                self.assertIsNone(pending.head_center)
                self.assertFalse(pending.head_verified)
                self.assertEqual(pending.head_state, "confirming")
            detection = detector.detect(frame)
        self.assertEqual(detection.people_count, 0)
        self.assertIsNone(detection.features)
        self.assertEqual(detection.valid_features, frozenset())
        self.assertEqual(detection.head_center, (0.5, 0.4375))
        self.assertEqual(detection.head_source, "face")
        self.assertTrue(detection.head_verified)
        self.assertEqual(detection.head_state, "verified")
        self.assertEqual(detection.head_track_id, "face-1")
        self.assertEqual(self.pose.detect_for_video.call_args.args[1], self.face.detect_for_video.call_args.args[1])

    def test_missing_optional_model_logs_and_uses_pose_head(self):
        self.pose.detect_for_video.return_value = SimpleNamespace(pose_landmarks=[pose_head()])
        with self.assertLogs("so101_human_follow.pose_detector", level="WARNING") as captured:
            detector = self.make_detector(face_model=False)
        self.assertIn("Face model missing", " ".join(captured.output))
        detection = detector.detect(np.zeros((480, 640, 3), dtype=np.uint8))
        self.assertIsNone(detector.face_detector)
        self.assertIsNone(detection.features)
        self.assertEqual(detection.head_source, "pose")
        self.assertAlmostEqual(detection.head_center[0], 0.65)

    def test_optional_model_initialization_failure_keeps_pose_and_hand_alive(self):
        self.mp.tasks.vision.FaceDetector.create_from_options.side_effect = RuntimeError("invalid model")
        with self.assertLogs("so101_human_follow.pose_detector", level="WARNING"):
            detector = self.make_detector()
        self.assertIsNone(detector.face_detector)
        self.pose.close.assert_not_called()
        self.hand.close.assert_not_called()

    def test_face_inference_failure_disables_only_face_model(self):
        detector = self.make_detector()
        self.face.detect_for_video.side_effect = RuntimeError("inference failed")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        with self.assertLogs("so101_human_follow.pose_detector", level="WARNING"):
            detector.detect(frame)
        detector.detect(frame)
        self.assertIsNone(detector.face_detector)
        self.assertEqual(self.face.detect_for_video.call_count, 1)
        self.face.close.assert_called_once()
        self.assertEqual(self.pose.detect_for_video.call_count, 2)
        self.assertEqual(self.hand.detect_for_video.call_count, 2)

    def test_close_always_releases_pose_and_hand_even_when_face_close_fails(self):
        detector = self.make_detector()
        self.face.close.side_effect = RuntimeError("face cleanup failed")
        with self.assertRaisesRegex(RuntimeError, "cleanup failed"):
            detector.close()
        self.hand.close.assert_called_once()
        self.pose.close.assert_called_once()
        self.face.close.side_effect = None


if __name__ == "__main__":
    unittest.main()
