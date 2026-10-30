from __future__ import annotations

import math
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


@dataclass(frozen=True)
class PoseDetection:
    features: PoseFeatures | None
    landmarks: tuple[Landmark, ...]
    people_count: int
    reason: str
    hand_landmarks: tuple[Landmark, ...] = ()
    hand_label: str | None = None
    valid_features: frozenset[str] = frozenset()


class MediaPipePoseDetector:
    """Thin wrapper around the current MediaPipe Tasks Pose Landmarker API."""

    def __init__(self, settings: PoseSettings):
        self.settings = settings
        self.model_path = Path(settings.model_path)
        self.hand_model_path = Path(settings.hand_model_path)
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
        count = len(pose_result.pose_landmarks)
        if count > 1:
            return PoseDetection(None, (), count, "multiple people")

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
        return PoseDetection(
            features_or_none,
            landmarks,
            count,
            reason,
            hand_landmarks,
            hand_label,
            valid,
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
) -> np.ndarray:
    import cv2

    output = frame_bgr.copy()
    height, width = output.shape[:2]
    if detection.landmarks:
        connections = ((11, 12), (11, 13), (13, 15), (12, 14), (14, 16))
        for start, end in connections:
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
            cv2.circle(
                output,
                (int(point.x * width), int(point.y * height)),
                3,
                (255, 220, 40),
                -1,
                cv2.LINE_AA,
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

