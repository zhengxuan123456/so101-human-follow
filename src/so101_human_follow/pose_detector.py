from __future__ import annotations

import math
import logging
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

import numpy as np

from .config import PoseSettings
from .pose_math import (
    Landmark,
    PoseFeatures,
    extract_arm_feature_channels,
    extract_hand_feature_channels,
)


LOGGER = logging.getLogger(__name__)
HEAD_MIN_CONFIDENCE = 0.55
_HEAD_POINTS = (0, 2, 5, 7, 8, 9, 10)
_FACE_VERIFY_INTERVAL_S = 0.25
_FACE_CONFIRM_FRAMES = 3
_FACE_CONFIRM_SPAN_S = 0.12
_FACE_FRAME_MAX_GAP_S = 0.35


@dataclass(frozen=True)
class PoseDetection:
    features: PoseFeatures | None
    landmarks: tuple[Landmark, ...]
    people_count: int
    reason: str
    hand_landmarks: tuple[Landmark, ...] = ()
    hand_label: str | None = None
    valid_features: frozenset[str] = frozenset()
    head_center: tuple[float, float] | None = None
    head_box: tuple[float, float, float, float] | None = None
    head_confidence: float = 0.0
    head_source: str | None = None
    head_track_id: str | None = None
    head_verified: bool = False
    head_state: str = ""


@dataclass(frozen=True)
class _HeadObservation:
    center: tuple[float, float] | None = None
    box: tuple[float, float, float, float] | None = None
    confidence: float = 0.0
    source: str | None = None
    track_id: str | None = None
    verified: bool = False
    state: str = ""


def _reliable_point(point: Landmark, min_confidence: float) -> bool:
    return (
        all(math.isfinite(value) for value in (point.x, point.y, point.visibility, point.presence))
        and min(point.visibility, point.presence) >= min_confidence
        and 0.0 <= point.x <= 1.0 and 0.0 <= point.y <= 1.0
    )


def _pose_head(landmarks: Sequence[Landmark]) -> _HeadObservation:
    """Recover a head target from reliable facial points even without arm channels."""
    points = [landmarks[index] for index in _HEAD_POINTS
              if index < len(landmarks) and _reliable_point(landmarks[index], HEAD_MIN_CONFIDENCE)]
    if len(points) < 2:
        return _HeadObservation()
    left, right = min(p.x for p in points), max(p.x for p in points)
    top, bottom = min(p.y for p in points), max(p.y for p in points)
    if max(right - left, bottom - top) < 0.015:
        return _HeadObservation()
    pad_x = max(0.015, (right - left) * 0.35)
    pad_y = max(0.015, (bottom - top) * 0.55, (right - left) * 0.45)
    return _HeadObservation(
        (sum(p.x for p in points) / len(points), sum(p.y for p in points) / len(points)),
        (max(0.0, left - pad_x), max(0.0, top - pad_y),
         min(1.0, right + pad_x), min(1.0, bottom + pad_y)),
        min(min(p.visibility, p.presence) for p in points), "pose", verified=True, state="pose",
    )


def _select_head(face_result, image_shape: Sequence[int], landmarks: Sequence[Landmark]) -> _HeadObservation:
    """Select a visible operator face without changing body-pose people_count."""
    fallback = _pose_head(landmarks)
    if len(image_shape) < 2 or min(image_shape[:2]) <= 0:
        return fallback
    height, width = image_shape[:2]
    candidates: list[tuple[float, _HeadObservation]] = []
    for detection in getattr(face_result, "detections", ()):
        box = getattr(detection, "bounding_box", None)
        try:
            score = max(float(category.score) for category in detection.categories)
            x, y = float(box.origin_x) / width, float(box.origin_y) / height
            w, h = float(box.width) / width, float(box.height) / height
        except (AttributeError, TypeError, ValueError, OverflowError):
            continue
        if (not all(math.isfinite(value) for value in (x, y, w, h, score))
                or not HEAD_MIN_CONFIDENCE <= score <= 1.0 or w <= 0.0 or h <= 0.0):
            continue
        left, top, right, bottom = max(0.0, x), max(0.0, y), min(1.0, x + w), min(1.0, y + h)
        visible_area = max(0.0, right - left) * max(0.0, bottom - top)
        if (min(right - left, bottom - top) < 0.035
                or visible_area < 0.0025 or visible_area < w * h * 0.5):
            continue
        candidates.append((visible_area, _HeadObservation(
            ((left + right) / 2.0, (top + bottom) / 2.0),
            (left, top, right, bottom), score, "face",
        )))
    if not candidates:
        return fallback
    candidates.sort(key=lambda item: item[0], reverse=True)
    # Associate with the body's reliable head only when that face is large enough
    # to be plausible. A tiny face printed on a poster must not beat a nearby user.
    if fallback.center is not None:
        x, y = fallback.center
        associated = []
        for area, candidate in candidates:
            left, top, right, bottom = candidate.box
            pad_x, pad_y = max(0.03, (right - left) * 0.15), max(0.03, (bottom - top) * 0.15)
            if (area >= candidates[0][0] * 0.3
                    and left - pad_x <= x <= right + pad_x and top - pad_y <= y <= bottom + pad_y):
                associated.append(candidate)
        if associated:
            return min(associated, key=lambda candidate: math.dist(candidate.center, fallback.center))
    if len(candidates) > 1 and candidates[0][0] < candidates[1][0] * 1.8:
        return _HeadObservation(source="ambiguous")
    return candidates[0][1]


def _head_box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    overlap = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - overlap
    return overlap / union if union > 0.0 else 0.0


class _YuNetFaceVerifier:
    """Check a BlazeFace proposal using an independent, local ROI detector."""

    def __init__(self, path: Path, score_threshold: float = 0.5, min_iou: float = 0.3):
        import cv2

        # Starting values were checked against two offline screenshots. They
        # are configurable, not a guarantee for every pose or installation.
        self.score_threshold = score_threshold
        self.min_iou = min_iou
        self.detector = cv2.FaceDetectorYN.create(str(path), "", (320, 320), score_threshold, 0.3, 5000)

    def verify(self, frame_bgr: np.ndarray, proposal: _HeadObservation) -> bool:
        import cv2

        if self.detector is None or proposal.box is None or frame_bgr.ndim != 3:
            return False
        height, width = frame_bgr.shape[:2]
        left, top, right, bottom = proposal.box
        if (min(height, width) <= 0 or not all(math.isfinite(value) for value in proposal.box)
                or not 0.0 <= left < right <= 1.0 or not 0.0 <= top < bottom <= 1.0):
            return False
        w, h = right - left, bottom - top
        x1, y1 = max(0, int((left - w * .25) * width)), max(0, int((top - h * .25) * height))
        x2, y2 = min(width, int((right + w * .25) * width)), min(height, int((bottom + h * .25) * height))
        if x2 <= x1 or y2 <= y1:
            return False
        roi = cv2.resize(frame_bgr[y1:y2, x1:x2], (320, 320))
        _, faces = self.detector.detect(roi)
        if faces is None:
            return False
        for row in faces:
            if (len(row) < 15 or not np.isfinite(row).all() or row[-1] < self.score_threshold
                    or row[2] <= 0.0 or row[3] <= 0.0):
                continue
            mapped = (
                (x1 + float(row[0]) * (x2 - x1) / 320) / width,
                (y1 + float(row[1]) * (y2 - y1) / 320) / height,
                (x1 + float(row[0] + row[2]) * (x2 - x1) / 320) / width,
                (y1 + float(row[1] + row[3]) * (y2 - y1) / 320) / height,
            )
            if _head_box_iou(proposal.box, mapped) >= self.min_iou:
                return True
        return False

    def close(self) -> None:
        self.detector = None


class _VerifiedHeadTracker:
    """Require independent verification and fresh consecutive observations."""

    def __init__(self, verifier: _YuNetFaceVerifier | None):
        self.verifier = verifier
        self._sequence = 0
        self.reset()

    def reset(self) -> None:
        self._last: _HeadObservation | None = None
        self._last_at: float | None = None
        self._first_at = 0.0
        self._frames = 0
        self._track_id: str | None = None
        self._verified_at: float | None = None
        self._verified_box: tuple[float, float, float, float] | None = None

    @staticmethod
    def _blocked(candidate: _HeadObservation, state: str) -> _HeadObservation:
        return _HeadObservation(confidence=candidate.confidence, source=candidate.source, state=state)

    def update(self, frame: np.ndarray, candidate: _HeadObservation, now: float) -> _HeadObservation:
        if candidate.source != "face":
            self.reset()
            return replace(candidate, state="ambiguous") if candidate.source == "ambiguous" else candidate
        if (candidate.center is None or candidate.box is None
                or len(candidate.center) != 2 or len(candidate.box) != 4
                or not all(math.isfinite(value) for value in (*candidate.center, *candidate.box))
                or not all(0.0 <= value <= 1.0 for value in candidate.center)
                or not 0.0 <= candidate.box[0] < candidate.box[2] <= 1.0
                or not 0.0 <= candidate.box[1] < candidate.box[3] <= 1.0):
            self.reset()
            return self._blocked(candidate, "unverified")
        if self.verifier is None:
            self.reset()
            return self._blocked(candidate, "unavailable")
        if not math.isfinite(now) or (self._last_at is not None and now <= self._last_at):
            self.reset()
            return self._blocked(candidate, "unverified")
        continuous = bool(
            self._last is not None and self._last_at is not None
            and now - self._last_at <= _FACE_FRAME_MAX_GAP_S
            and math.dist(candidate.center, self._last.center) <= 0.12
            and _head_box_iou(candidate.box, self._last.box) > 0.1
        )
        if not continuous:
            self.reset()
            self._sequence += 1
            self._track_id = f"face-{self._sequence}"
            self._first_at = now
        verify_due = bool(
            self._verified_at is None or now - self._verified_at >= _FACE_VERIFY_INTERVAL_S
            or self._verified_box is None or _head_box_iou(candidate.box, self._verified_box) < 0.4
        )
        if verify_due:
            try:
                accepted = self.verifier.verify(frame, candidate)
            except Exception as exc:
                LOGGER.warning("Face verification failed; independent face tracking disabled: %s", exc)
                self.verifier.close()
                self.verifier = None
                self.reset()
                return self._blocked(candidate, "unavailable")
            if not accepted:
                self.reset()
                return self._blocked(candidate, "unverified")
            self._verified_at, self._verified_box = now, candidate.box
        self._last, self._last_at = candidate, now
        self._frames += 1
        if self._frames < _FACE_CONFIRM_FRAMES or now - self._first_at + 1e-9 < _FACE_CONFIRM_SPAN_S:
            return _HeadObservation(confidence=candidate.confidence, source="face",
                                    track_id=self._track_id, state="confirming")
        return replace(candidate, track_id=self._track_id, verified=True, state="verified")

    def close(self) -> None:
        self.reset()
        if self.verifier is not None:
            self.verifier.close()
            self.verifier = None


class MediaPipePoseDetector:
    """Thin wrapper around the current MediaPipe Tasks Pose Landmarker API."""

    def __init__(self, settings: PoseSettings):
        self.settings = settings
        self.model_path = Path(settings.model_path)
        self.hand_model_path = Path(settings.hand_model_path)
        self.face_model_path = (
            Path(settings.face_model_path) if settings.face_model_path is not None
            else self.model_path.with_name("blaze_face_short_range.tflite")
        )
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Pose model not found: {self.model_path}\n"
                "Run: python scripts\\download_pose_model.py"
            )
        if not self.hand_model_path.is_file():
            raise FileNotFoundError(
                f"Hand model not found: {self.hand_model_path}\n"
                "Run: python scripts\\download_hand_model.py"
            )

        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError(
                "MediaPipe is not installed. Use an isolated clone of the lerobot environment, then run "
                "'python -m pip install -r requirements-vision.txt'."
            ) from exc

        self.mp = mp
        pose_options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(self.model_path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            # The arm-mounted camera often sees printed people/robots in the
            # background.  Asking MediaPipe for two poses made those posters a
            # hard "multiple people" fault and dropped every control channel.
            # This controller follows one operator, so keep only the strongest
            # pose candidate.
            num_poses=1,
            min_pose_detection_confidence=settings.min_detection_confidence,
            min_pose_presence_confidence=settings.min_presence_confidence,
            min_tracking_confidence=settings.min_tracking_confidence,
            output_segmentation_masks=False,
        )
        hand_options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(self.hand_model_path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            # Follow one dominant hand.  With two hands enabled, a hand-only
            # frame could be rejected as ambiguous even though one hand was
            # large and close to the camera.
            num_hands=1,
            min_hand_detection_confidence=settings.hand_min_detection_confidence,
            min_hand_presence_confidence=settings.hand_min_presence_confidence,
            min_tracking_confidence=settings.hand_min_tracking_confidence,
        )
        self.pose_landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(pose_options)
        try:
            self.hand_landmarker = mp.tasks.vision.HandLandmarker.create_from_options(hand_options)
        except Exception:
            self.pose_landmarker.close()
            raise
        self.face_detector = None
        if self.face_model_path.is_file():
            try:
                face_options = mp.tasks.vision.FaceDetectorOptions(
                    base_options=mp.tasks.BaseOptions(model_asset_path=str(self.face_model_path)),
                    running_mode=mp.tasks.vision.RunningMode.VIDEO,
                    min_detection_confidence=HEAD_MIN_CONFIDENCE,
                )
                self.face_detector = mp.tasks.vision.FaceDetector.create_from_options(face_options)
            except Exception as exc:
                LOGGER.warning("Face detector unavailable; using reliable pose head points: %s", exc)
        else:
            LOGGER.warning(
                "Face model missing: %s; using reliable pose head points. Run: python scripts/download_face_model.py",
                self.face_model_path,
            )
        verifier = None
        if self.face_detector is not None:
            verifier_path = (
                Path(settings.face_verifier_model_path) if settings.face_verifier_model_path is not None
                else self.model_path.with_name("face_detection_yunet_2026may.onnx")
            )
            if verifier_path.is_file():
                try:
                    verifier = _YuNetFaceVerifier(
                        verifier_path, settings.face_verifier_score_threshold, settings.face_verifier_min_iou,
                    )
                except Exception as exc:
                    LOGGER.warning("Face verifier unavailable; independent face tracking blocked: %s", exc)
            else:
                LOGGER.warning(
                    "Face verifier missing: %s; independent face tracking blocked. Run: python scripts/download_face_verifier.py",
                    verifier_path,
                )
        self._head_tracker = _VerifiedHeadTracker(verifier)
        self.last_timestamp_ms = -1

    def detect(self, frame_bgr: np.ndarray) -> PoseDetection:
        import cv2

        timestamp_ms = int(time.monotonic() * 1000)
        timestamp_ms = max(timestamp_ms, self.last_timestamp_ms + 1)
        self.last_timestamp_ms = timestamp_ms

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb = np.ascontiguousarray(frame_rgb)
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=frame_rgb)
        pose_result = self.pose_landmarker.detect_for_video(image, timestamp_ms)
        hand_result = self.hand_landmarker.detect_for_video(image, timestamp_ms)
        face_result = None
        if self.face_detector is not None:
            try:
                face_result = self.face_detector.detect_for_video(image, timestamp_ms)
            except Exception as exc:
                LOGGER.warning("Face inference failed; using reliable pose head points: %s", exc)
                try:
                    self.face_detector.close()
                except Exception:
                    LOGGER.warning("Face detector cleanup failed", exc_info=True)
                self.face_detector = None
        count = len(pose_result.pose_landmarks)
        if count > 1:
            head = self._head_tracker.update(
                frame_bgr, _select_head(face_result, frame_bgr.shape, ()), timestamp_ms / 1000.0,
            )
            return PoseDetection(None, (), count, "multiple people", head_center=head.center,
                                 head_box=head.box, head_confidence=head.confidence, head_source=head.source,
                                 head_track_id=head.track_id, head_verified=head.verified, head_state=head.state)

        landmarks: tuple[Landmark, ...] = ()
        pose_features = PoseFeatures(pan=0.0, lift=0.0, elbow=0.0, center_x=0.5)
        pose_valid: frozenset[str] = frozenset()
        if count == 1:
            landmarks = tuple(
                Landmark(
                    x=float(point.x),
                    y=float(point.y),
                    z=float(point.z or 0.0),
                    visibility=float(point.visibility if point.visibility is not None else 1.0),
                    presence=float(point.presence if point.presence is not None else 1.0),
                )
                for point in pose_result.pose_landmarks[0]
            )
            pose_features, pose_valid = extract_arm_feature_channels(
                landmarks,
                side=self.settings.side,
                min_visibility=self.settings.min_landmark_visibility,
                aspect_ratio=frame_bgr.shape[1] / frame_bgr.shape[0],
            )

        pose_for_hand = landmarks if landmarks else None
        if pose_for_hand is not None:
            wrist_index = 16 if self.settings.side == "right" else 15
            wrist = pose_for_hand[wrist_index]
            if (
                not all(math.isfinite(value) for value in (wrist.x, wrist.y, wrist.z))
                or min(wrist.visibility, wrist.presence) < self.settings.min_landmark_visibility
            ):
                pose_for_hand = None
        selected = self._select_hand(
            hand_result,
            pose_for_hand,
            frame_bgr.shape[1] / frame_bgr.shape[0],
        )
        if selected is None:
            hand_landmarks: tuple[Landmark, ...] = ()
            hand_label = None
            hand_features = None
            hand_valid = frozenset()
        else:
            hand_landmarks, hand_label = selected
            hand_features, hand_valid = extract_hand_feature_channels(
                hand_landmarks,
                min_visibility=0.0,
                aspect_ratio=frame_bgr.shape[1] / frame_bgr.shape[0],
            )

        features = pose_features
        combined_valid = set(pose_valid)
        if hand_features is not None:
            features = replace(
                features,
                hand_roll=hand_features.hand_roll,
                hand_open=hand_features.hand_open,
            )
            combined_valid.update(hand_valid)

        valid = frozenset(combined_valid)
        control_valid = valid & {"lift", "elbow", "hand_roll", "hand_open"}
        if not control_valid:
            reason = "no person" if count == 0 else "no reliable control channel"
            features_or_none = None
        else:
            pose_channels = control_valid & {"lift", "elbow"}
            hand_channels = control_valid & {"hand_roll", "hand_open"}
            if pose_channels and hand_channels:
                reason = "pose + hand channels active"
            elif pose_channels:
                reason = "pose channels active; hand held"
            else:
                reason = "hand channels active; arm held"
            features_or_none = features
        head = self._head_tracker.update(
            frame_bgr, _select_head(face_result, frame_bgr.shape, landmarks), timestamp_ms / 1000.0,
        )
        return PoseDetection(
            features_or_none,
            landmarks,
            count,
            reason,
            hand_landmarks,
            hand_label,
            valid,
            head.center,
            head.box,
            head.confidence,
            head.source,
            head.track_id,
            head.verified,
            head.state,
        )

    def _select_hand(
        self,
        result,
        pose_landmarks: tuple[Landmark, ...] | None,
        aspect_ratio: float,
    ) -> tuple[tuple[Landmark, ...], str | None] | None:
        candidates: list[tuple[float, tuple[Landmark, ...], str | None]] = []
        for index, raw_hand in enumerate(result.hand_landmarks):
            hand = tuple(
                Landmark(
                    x=float(point.x),
                    y=float(point.y),
                    z=float(point.z or 0.0),
                    visibility=float(
                        point.visibility if getattr(point, "visibility", None) is not None else 1.0
                    ),
                    presence=float(
                        point.presence if getattr(point, "presence", None) is not None else 1.0
                    ),
                )
                for point in raw_hand
            )
            if not hand:
                continue
            if pose_landmarks:
                wrist_index = 16 if self.settings.side == "right" else 15
                pose_wrist = pose_landmarks[wrist_index]
                distance = (
                    ((hand[0].x - pose_wrist.x) * aspect_ratio) ** 2
                    + (hand[0].y - pose_wrist.y) ** 2
                ) ** 0.5
            else:
                distance = 0.0
            label = None
            if index < len(result.handedness) and result.handedness[index]:
                category = result.handedness[index][0]
                label = category.category_name or category.display_name or None
            candidates.append((distance, hand, label))
        if not candidates:
            return None
        if pose_landmarks is None and len(candidates) != 1:
            return None
        distance, hand, label = min(candidates, key=lambda candidate: candidate[0])
        # With one hand candidate, accept it directly.  On an arm-mounted camera
        # the Pose wrist can drift far from the much more accurate Hand wrist;
        # rejecting that single hand disabled the gripper completely.
        if (
            pose_landmarks is not None
            and len(candidates) > 1
            and distance > self.settings.max_hand_wrist_distance
        ):
            return None
        return hand, label

    def close(self) -> None:
        try:
            try:
                self._head_tracker.close()
            finally:
                if self.face_detector is not None:
                    self.face_detector.close()
        finally:
            try:
                self.hand_landmarker.close()
            finally:
                self.pose_landmarker.close()


def orient_camera_frame(
    frame_bgr: np.ndarray,
    *,
    rotation_degrees: float = 0.0,
    flip_horizontal: bool = False,
    flip_vertical: bool = False,
) -> np.ndarray:
    """Apply configured image-plane mount correction before visual inference."""

    import cv2

    output = frame_bgr
    if flip_horizontal and flip_vertical:
        output = cv2.flip(output, -1)
    elif flip_horizontal:
        output = cv2.flip(output, 1)
    elif flip_vertical:
        output = cv2.flip(output, 0)
    angle = float(rotation_degrees) % 360.0
    if min(angle, 360.0 - angle) < 1e-6:
        return output
    height, width = output.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.0)
    return cv2.warpAffine(
        output,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def annotate_frame(
    frame_bgr: np.ndarray,
    detection: PoseDetection,
    status_lines: Sequence[str],
    *,
    face_target: tuple[float, float] | None = None,
) -> np.ndarray:
    import cv2

    output = frame_bgr.copy()
    height, width = output.shape[:2]
    if detection.landmarks:
        connections = ((11, 12), (11, 13), (13, 15), (12, 14), (14, 16))
        for start, end in connections:
            if (max(start, end) >= len(detection.landmarks)
                    or not _reliable_point(detection.landmarks[start], 0.5)
                    or not _reliable_point(detection.landmarks[end], 0.5)):
                continue
            a = detection.landmarks[start]
            b = detection.landmarks[end]
            cv2.line(
                output,
                (int(a.x * width), int(a.y * height)),
                (int(b.x * width), int(b.y * height)),
                (40, 220, 40),
                3,
                cv2.LINE_AA,
            )
        for index in {11, 12, 13, 14, 15, 16}:
            if index >= len(detection.landmarks) or not _reliable_point(detection.landmarks[index], 0.5):
                continue
            point = detection.landmarks[index]
            cv2.circle(
                output,
                (int(point.x * width), int(point.y * height)),
                5,
                (0, 180, 255),
                -1,
                cv2.LINE_AA,
            )

    if detection.hand_landmarks:
        hand_connections = (
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12),
            (9, 13), (13, 14), (14, 15), (15, 16),
            (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
        )
        for start, end in hand_connections:
            if (max(start, end) >= len(detection.hand_landmarks)
                    or not _reliable_point(detection.hand_landmarks[start], 0.0)
                    or not _reliable_point(detection.hand_landmarks[end], 0.0)):
                continue
            a = detection.hand_landmarks[start]
            b = detection.hand_landmarks[end]
            cv2.line(
                output,
                (int(a.x * width), int(a.y * height)),
                (int(b.x * width), int(b.y * height)),
                (255, 150, 30),
                2,
                cv2.LINE_AA,
            )
        for point in detection.hand_landmarks:
            if not _reliable_point(point, 0.0):
                continue
            cv2.circle(
                output,
                (int(point.x * width), int(point.y * height)),
                3,
                (255, 220, 40),
                -1,
                cv2.LINE_AA,
            )

    if (detection.head_box is not None and len(detection.head_box) == 4
            and all(math.isfinite(value) for value in detection.head_box)
            and detection.head_confidence >= HEAD_MIN_CONFIDENCE):
        left, top, right, bottom = detection.head_box
        if 0.0 <= left < right <= 1.0 and 0.0 <= top < bottom <= 1.0:
            cv2.rectangle(output, (int(left * width), int(top * height)),
                          (int(right * width), int(bottom * height)), (255, 210, 40), 2, cv2.LINE_AA)
            cv2.putText(output, f"head {detection.head_source or ''} {detection.head_confidence:.2f}",
                        (int(left * width), max(16, int(top * height) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 210, 40), 1, cv2.LINE_AA)

    target = face_target if face_target is not None else detection.head_center
    if (target is not None and len(target) == 2
            and all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in target)):
        x, y = target
        cv2.drawMarker(
            output, (width // 2, height // 2), (160, 160, 160),
            cv2.MARKER_CROSS, markerSize=16, thickness=1, line_type=cv2.LINE_AA,
        )
        cv2.drawMarker(
            output, (int(x * width), int(y * height)), (255, 235, 40),
            cv2.MARKER_CROSS, markerSize=22, thickness=2, line_type=cv2.LINE_AA,
        )

    y = 26
    for line in status_lines:
        cv2.putText(
            output,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            output,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (30, 30, 30),
            1,
            cv2.LINE_AA,
        )
        y += 24
    return output

