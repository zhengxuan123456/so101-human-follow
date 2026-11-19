import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from so101_human_follow.config import load_config
from so101_human_follow.pose_detector import (
    _HeadObservation, _VerifiedHeadTracker, _YuNetFaceVerifier,
)


FACE = _HeadObservation((0.5, 0.5), (0.4, 0.3, 0.6, 0.7), 0.6, "face")


class FaceVerificationTrackerTest(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.verifier = SimpleNamespace(verify=Mock(return_value=True), close=Mock())
        self.tracker = _VerifiedHeadTracker(self.verifier)

    def update(self, at, candidate=FACE):
        return self.tracker.update(self.frame, candidate, at)

    def confirmed(self):
        self.update(1.0)
        self.update(1.06)
        result = self.update(1.12)
        self.assertTrue(result.verified)
        return result

    def test_requires_three_fresh_frames_and_minimum_time_span(self):
        for at in (1.0, 1.02, 1.04):
            result = self.update(at)
            self.assertIsNone(result.center)
            self.assertIsNone(result.box)
            self.assertEqual(result.state, "confirming")
        result = self.update(1.12)
        self.assertEqual(result.center, FACE.center)
        self.assertTrue(result.verified)
        self.assertEqual(result.state, "verified")
        self.assertEqual(result.track_id, "face-1")

    def test_verification_cache_is_reused_briefly_then_rechecked(self):
        self.confirmed()
        self.update(1.20)
        self.assertEqual(self.verifier.verify.call_count, 1)
        self.update(1.26)
        self.assertEqual(self.verifier.verify.call_count, 2)

    def test_current_target_is_returned_instead_of_cached_center(self):
        result = self.confirmed()
        shifted = replace(FACE, center=(0.54, 0.5), box=(0.44, 0.3, 0.64, 0.7))
        fresh = self.update(1.18, shifted)
        self.assertTrue(fresh.verified)
        self.assertEqual(fresh.center, shifted.center)
        self.assertEqual(fresh.track_id, result.track_id)

    def test_missing_face_immediately_clears_output_and_changes_reacquired_id(self):
        original = self.confirmed()
        lost = self.update(1.18, _HeadObservation())
        self.assertIsNone(lost.center)
        self.assertIsNone(lost.track_id)
        self.assertFalse(lost.verified)
        self.assertEqual(lost.state, "")
        pending = self.update(1.24)
        self.assertNotEqual(pending.track_id, original.track_id)
        self.assertEqual(pending.state, "confirming")
        self.assertIsNone(pending.center)

    def test_target_jump_requires_a_new_confirmation(self):
        original = self.confirmed()
        jumped = replace(FACE, center=(0.8, 0.5), box=(0.7, 0.3, 0.9, 0.7))
        result = self.update(1.18, jumped)
        self.assertIsNone(result.center)
        self.assertFalse(result.verified)
        self.assertNotEqual(result.track_id, original.track_id)
        self.assertEqual(result.state, "confirming")
        self.assertEqual(self.verifier.verify.call_count, 2)

    def test_long_frame_gap_discards_the_old_track(self):
        old = self.confirmed()
        result = self.update(1.60)
        self.assertIsNone(result.center)
        self.assertEqual(result.state, "confirming")
        self.assertNotEqual(result.track_id, old.track_id)

    def test_duplicate_timestamp_does_not_count_as_a_new_frame(self):
        self.update(1.0)
        repeated = self.update(1.0)
        self.assertIsNone(repeated.center)
        self.assertFalse(repeated.verified)
        self.assertEqual(repeated.state, "unverified")
        self.assertIsNone(self.update(1.06).center)
        self.assertIsNone(self.update(1.12).center)

    def test_failed_reverification_discards_the_previously_confirmed_head(self):
        self.confirmed()
        self.verifier.verify.return_value = False
        rejected = self.update(1.26)
        self.assertIsNone(rejected.center)
        self.assertIsNone(rejected.box)
        self.assertIsNone(rejected.track_id)
        self.assertFalse(rejected.verified)
        self.assertEqual(rejected.state, "unverified")
        self.assertEqual(rejected.source, "face")

    def test_changed_face_size_requires_early_reverification(self):
        self.confirmed()
        self.verifier.verify.return_value = False
        changed = replace(FACE, box=(0.46, 0.38, 0.54, 0.62))
        result = self.update(1.18, changed)
        self.assertEqual(self.verifier.verify.call_count, 2)
        self.assertEqual(result.state, "unverified")
        self.assertIsNone(result.center)

    def test_unavailable_verifier_never_exposes_independent_face_coordinates(self):
        tracker = _VerifiedHeadTracker(None)
        for at in (1.0, 1.06, 1.12, 1.2):
            result = tracker.update(self.frame, FACE, at)
            self.assertIsNone(result.center)
            self.assertFalse(result.verified)
            self.assertEqual(result.state, "unavailable")

    def test_inference_failure_disables_the_verifier_and_clears_cached_head(self):
        self.confirmed()
        self.verifier.verify.side_effect = RuntimeError("inference failed")
        with self.assertLogs("so101_human_follow.pose_detector", level="WARNING"):
            rejected = self.update(1.26)
        self.assertEqual(rejected.state, "unavailable")
        self.assertIsNone(rejected.center)
        self.verifier.close.assert_called_once()
        self.update(1.32)
        self.assertEqual(self.verifier.verify.call_count, 2)

    def test_ambiguous_face_set_clears_an_existing_track(self):
        self.confirmed()
        result = self.update(1.18, _HeadObservation(source="ambiguous"))
        self.assertIsNone(result.center)
        self.assertFalse(result.verified)
        self.assertEqual(result.state, "ambiguous")
        self.assertEqual(result.source, "ambiguous")

    def test_fresh_pose_fallback_cannot_reuse_a_face_track_id(self):
        old = self.confirmed()
        pose = _HeadObservation((0.42, 0.48), (0.3, 0.3, 0.6, 0.65), 0.8,
                                "pose", verified=True, state="pose")
        for at in (1.18, 1.24, 1.3):
            result = self.update(at, pose)
            self.assertEqual(result.source, "pose")
            self.assertEqual(result.center, pose.center)
            self.assertTrue(result.verified)
            self.assertIsNone(result.track_id)
            self.assertNotEqual(result.track_id, old.track_id)

    def test_corrupt_candidate_never_reaches_the_verification_model(self):
        for bad in (replace(FACE, center=(float("nan"), .5)),
                    replace(FACE, box=(.6, .3, .4, .7)), replace(FACE, center=(1.2, .5))):
            result = self.update(1.0, bad)
            self.assertFalse(result.verified)
            self.assertIsNone(result.center)
        self.verifier.verify.assert_not_called()

    def test_close_releases_model_and_cached_tracking_state(self):
        self.confirmed()
        self.tracker.close()
        self.verifier.close.assert_called_once()
        self.assertIsNone(self.tracker._last)
        self.assertIsNone(self.tracker.verifier)
        self.assertEqual(self.update(1.2).state, "unavailable")


class YuNetROIVerificationTest(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.verifier = _YuNetFaceVerifier.__new__(_YuNetFaceVerifier)
        self.verifier.score_threshold = 0.5
        self.verifier.min_iou = 0.3
        self.verifier.detector = SimpleNamespace(detect=Mock())

    def results(self, *rows):
        self.verifier.detector.detect.return_value = (len(rows), np.asarray(rows, dtype=np.float32) if rows else None)

    @staticmethod
    def matching_row(score=0.6):
        return [53.333, 53.333, 213.333, 213.333, *([0.0] * 10), score]

    def test_matching_independent_face_accepts_a_low_confidence_blazeface_candidate(self):
        self.results(self.matching_row())
        self.assertTrue(self.verifier.verify(self.frame, FACE))
        roi = self.verifier.detector.detect.call_args.args[0]
        self.assertEqual(roi.shape, (320, 320, 3))

    def test_weak_or_nonoverlapping_verifier_boxes_cannot_confirm_a_candidate(self):
        for row in (self.matching_row(0.49), [0, 0, 20, 20, *([0.0] * 10), 0.99],
                    self.matching_row(float("nan"))):
            self.results(row)
            self.assertFalse(self.verifier.verify(self.frame, FACE))

    def test_no_verifier_detection_rejects_the_proposal(self):
        self.results()
        self.assertFalse(self.verifier.verify(self.frame, FACE))

    def test_closed_verifier_rejects_without_running_a_model(self):
        self.verifier.close()
        self.assertFalse(self.verifier.verify(self.frame, FACE))


class FaceVerifierConfigurationTest(unittest.TestCase):
    def test_optional_model_path_resolves_relative_to_config_and_thresholds_are_validated(self):
        raw = json.loads(Path("config.example.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            raw["pose"]["face_verifier_model_path"] = "verification/yunet.onnx"
            path.write_text(json.dumps(raw), encoding="utf-8")
            settings = load_config(path).pose
            self.assertEqual(settings.face_verifier_model_path, Path(directory) / "verification" / "yunet.onnx")
            self.assertEqual(settings.face_verifier_score_threshold, 0.5)
            self.assertEqual(settings.face_verifier_min_iou, 0.3)
            for name in ("face_verifier_score_threshold", "face_verifier_min_iou"):
                for invalid in (0, 1.1, float("nan")):
                    with self.subTest(name=name, invalid=invalid):
                        raw["pose"][name] = invalid
                        path.write_text(json.dumps(raw), encoding="utf-8")
                        with self.assertRaisesRegex(ValueError, name):
                            load_config(path)
                raw["pose"].pop(name)


if __name__ == "__main__":
    unittest.main()
