from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class Landmark:
    x: float
    y: float
    z: float = 0.0
    visibility: float = 1.0
    presence: float = 1.0


@dataclass(frozen=True)
class PoseFeatures:
    pan: float
    lift: float
    elbow: float
    center_x: float
    hand_roll: float = 0.0
    hand_open: float = 0.0


@dataclass(frozen=True)
class HandFeatures:
    hand_roll: float
    hand_open: float


POSE_INDEX = {
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
}


HAND_INDEX = {
    "wrist": 0,
    "index_mcp": 5,
    "index_pip": 6,
    "index_tip": 8,
    "middle_mcp": 9,
    "middle_pip": 10,
    "middle_tip": 12,
    "ring_mcp": 13,
    "ring_pip": 14,
    "ring_tip": 16,
    "pinky_mcp": 17,
    "pinky_pip": 18,
    "pinky_tip": 20,
}


def _finite_landmark(point: Landmark) -> bool:
    return all(
        math.isfinite(value)
        for value in (point.x, point.y, point.z, point.visibility, point.presence)
    )


def _distance(a: Landmark, b: Landmark, aspect_ratio: float) -> float:
    return math.hypot((a.x - b.x) * aspect_ratio, a.y - b.y)


def _elbow_flexion(
    shoulder: Landmark,
    elbow: Landmark,
    wrist: Landmark,
    aspect_ratio: float,
) -> float | None:
    first = ((shoulder.x - elbow.x) * aspect_ratio, shoulder.y - elbow.y)
    second = ((wrist.x - elbow.x) * aspect_ratio, wrist.y - elbow.y)
    first_norm = math.hypot(*first)
    second_norm = math.hypot(*second)
    if first_norm < 1e-6 or second_norm < 1e-6:
        return None
    cosine = (first[0] * second[0] + first[1] * second[1]) / (first_norm * second_norm)
    angle = math.acos(max(-1.0, min(1.0, cosine)))
    return max(0.0, min(1.0, (math.pi - angle) / math.pi))


def _finger_extension(
    mcp: Landmark,
    pip: Landmark,
    tip: Landmark,
    aspect_ratio: float,
) -> float | None:
    """Return a 0..1 extension score from a finger's PIP joint angle."""

    first = ((mcp.x - pip.x) * aspect_ratio, mcp.y - pip.y)
    second = ((tip.x - pip.x) * aspect_ratio, tip.y - pip.y)
    first_norm = math.hypot(*first)
    second_norm = math.hypot(*second)
    if first_norm < 1e-6 or second_norm < 1e-6:
        return None
    cosine = (first[0] * second[0] + first[1] * second[1]) / (first_norm * second_norm)
    angle = math.acos(max(-1.0, min(1.0, cosine)))

    # A straight finger is close to pi radians. A tightly curled finger is
    # normally below about 0.75 radians. Clamping makes the output useful as a
    # conservative gripper command rather than an anatomical angle estimate.
    return max(0.0, min(1.0, (angle - 0.75) / (math.pi - 0.75)))


def extract_hand_features(
    landmarks: Sequence[Landmark],
    *,
    min_visibility: float = 0.5,
    aspect_ratio: float = 1.0,
) -> HandFeatures | None:
    """Extract in-plane palm roll and four-finger openness from 21 landmarks.

    ``hand_roll`` is normalized to [-1, 1] over a conservative +/-90 degree
    in-plane range. The transverse palm axis is folded so the signal is stable
    when MediaPipe sees the palm or the back of the same hand. ``hand_open`` is
    the mean extension of the index, middle, ring and pinky fingers.
    """

    if (
        len(landmarks) < 21
        or not math.isfinite(aspect_ratio)
        or aspect_ratio <= 0
        or not math.isfinite(min_visibility)
        or not 0.0 <= min_visibility <= 1.0
    ):
        return None

    required_indices = set(HAND_INDEX.values())
    required = tuple(landmarks[index] for index in required_indices)
    if any(not _finite_landmark(point) for point in required):
        return None
    if any(min(point.visibility, point.presence) < min_visibility for point in required):
        return None

    wrist = landmarks[HAND_INDEX["wrist"]]
    index_mcp = landmarks[HAND_INDEX["index_mcp"]]
    middle_mcp = landmarks[HAND_INDEX["middle_mcp"]]
    pinky_mcp = landmarks[HAND_INDEX["pinky_mcp"]]
    palm_width = _distance(index_mcp, pinky_mcp, aspect_ratio)
    palm_length = _distance(wrist, middle_mcp, aspect_ratio)
    if palm_width < 0.02 or palm_length < 0.02:
        return None

    across_x = (pinky_mcp.x - index_mcp.x) * aspect_ratio
    across_y = pinky_mcp.y - index_mcp.y
    # Fold the palm axis into the right half-plane. This avoids the +/-pi
    # discontinuity when the detector alternates between palm and back views.
    if across_x < 0.0:
        across_x = -across_x
        across_y = -across_y
    roll_angle = math.atan2(-across_y, across_x)
    hand_roll = max(-1.0, min(1.0, roll_angle / (math.pi / 2.0)))

    fingers = (
        ("index_mcp", "index_pip", "index_tip"),
        ("middle_mcp", "middle_pip", "middle_tip"),
        ("ring_mcp", "ring_pip", "ring_tip"),
        ("pinky_mcp", "pinky_pip", "pinky_tip"),
    )
    extensions: list[float] = []
    for mcp_name, pip_name, tip_name in fingers:
        extension = _finger_extension(
            landmarks[HAND_INDEX[mcp_name]],
            landmarks[HAND_INDEX[pip_name]],
            landmarks[HAND_INDEX[tip_name]],
            aspect_ratio,
        )
        if extension is None:
            return None
        extensions.append(extension)

    hand_open = statistics.fmean(extensions)
    if not math.isfinite(hand_roll) or not math.isfinite(hand_open):
        return None
    return HandFeatures(hand_roll=hand_roll, hand_open=max(0.0, min(1.0, hand_open)))


def extract_hand_feature_channels(
    landmarks: Sequence[Landmark],
    *,
    min_visibility: float = 0.5,
    aspect_ratio: float = 1.0,
) -> tuple[HandFeatures, frozenset[str]]:
    """Extract hand controls independently so one bad channel does not block the other."""

    fallback = HandFeatures(hand_roll=0.0, hand_open=0.0)
    if (
        len(landmarks) < 21
        or not math.isfinite(aspect_ratio)
        or aspect_ratio <= 0
        or not math.isfinite(min_visibility)
        or not 0.0 <= min_visibility <= 1.0
    ):
        return fallback, frozenset()

    def usable(*indices: int) -> bool:
        points = (landmarks[index] for index in indices)
        return all(
            _finite_landmark(point)
            and min(point.visibility, point.presence) >= min_visibility
            for point in points
        )

    valid: set[str] = set()
    hand_roll = 0.0
    palm_indices = (
        HAND_INDEX["wrist"],
        HAND_INDEX["index_mcp"],
        HAND_INDEX["middle_mcp"],
        HAND_INDEX["pinky_mcp"],
    )
    if usable(*palm_indices):
        wrist = landmarks[HAND_INDEX["wrist"]]
        index_mcp = landmarks[HAND_INDEX["index_mcp"]]
        middle_mcp = landmarks[HAND_INDEX["middle_mcp"]]
        pinky_mcp = landmarks[HAND_INDEX["pinky_mcp"]]
        palm_width = _distance(index_mcp, pinky_mcp, aspect_ratio)
        palm_length = _distance(wrist, middle_mcp, aspect_ratio)
        if palm_width >= 0.02 and palm_length >= 0.02:
            across_x = (pinky_mcp.x - index_mcp.x) * aspect_ratio
            across_y = pinky_mcp.y - index_mcp.y
            if across_x < 0.0:
                across_x = -across_x
                across_y = -across_y
            hand_roll = max(
                -1.0,
                min(1.0, math.atan2(-across_y, across_x) / (math.pi / 2.0)),
            )
            valid.add("hand_roll")

    extensions: list[float] = []
    fingers = (
        ("index_mcp", "index_pip", "index_tip"),
        ("middle_mcp", "middle_pip", "middle_tip"),
        ("ring_mcp", "ring_pip", "ring_tip"),
        ("pinky_mcp", "pinky_pip", "pinky_tip"),
    )
    for mcp_name, pip_name, tip_name in fingers:
        indices = (
            HAND_INDEX[mcp_name],
            HAND_INDEX[pip_name],
            HAND_INDEX[tip_name],
        )
        if not usable(*indices):
            continue
        extension = _finger_extension(
            landmarks[indices[0]],
            landmarks[indices[1]],
            landmarks[indices[2]],
            aspect_ratio,
        )
        if extension is not None:
            extensions.append(extension)

    # A fist also shortens the wrist-to-fingertip distances.  This measure is
    # robust when one or two PIP joints are temporarily occluded, which is
    # common with a webcam and was previously dropping the entire ID6 channel.
    distance_scores: list[float] = []
    wrist_index = HAND_INDEX["wrist"]
    middle_mcp_index = HAND_INDEX["middle_mcp"]
    tip_indices = (
        HAND_INDEX["index_tip"],
        HAND_INDEX["middle_tip"],
        HAND_INDEX["ring_tip"],
        HAND_INDEX["pinky_tip"],
    )
    if usable(wrist_index, middle_mcp_index):
        palm_length = _distance(
            landmarks[wrist_index], landmarks[middle_mcp_index], aspect_ratio
        )
        if palm_length >= 0.02:
            for tip_index in tip_indices:
                if not usable(tip_index):
                    continue
                ratio = _distance(
                    landmarks[wrist_index], landmarks[tip_index], aspect_ratio
                ) / palm_length
                distance_scores.append(max(0.0, min(1.0, (ratio - 1.0) / 1.4)))

    hand_open = 0.0
    openness_parts: list[float] = []
    if len(extensions) >= 2:
        openness_parts.append(statistics.fmean(extensions))
    if len(distance_scores) >= 2:
        openness_parts.append(statistics.fmean(distance_scores))
    if openness_parts:
        hand_open = max(0.0, min(1.0, statistics.fmean(openness_parts)))
        valid.add("hand_open")
    return HandFeatures(hand_roll=hand_roll, hand_open=hand_open), frozenset(valid)


def extract_arm_feature_channels(
    landmarks: Sequence[Landmark],
    *,
    side: str = "right",
    min_visibility: float = 0.7,
    aspect_ratio: float = 1.0,
) -> tuple[PoseFeatures, frozenset[str]]:
    """Extract lift and elbow channels independently from one pose."""

    fallback = PoseFeatures(pan=0.0, lift=0.0, elbow=0.0, center_x=0.5)
    if (
        len(landmarks) < 17
        or side not in {"left", "right"}
        or not math.isfinite(aspect_ratio)
        or aspect_ratio <= 0
    ):
        return fallback, frozenset()

    def usable(point: Landmark) -> bool:
        return _finite_landmark(point) and min(point.visibility, point.presence) >= min_visibility

    left_shoulder = landmarks[POSE_INDEX["left_shoulder"]]
    right_shoulder = landmarks[POSE_INDEX["right_shoulder"]]
    shoulder = landmarks[POSE_INDEX[f"{side}_shoulder"]]
    elbow = landmarks[POSE_INDEX[f"{side}_elbow"]]
    wrist = landmarks[POSE_INDEX[f"{side}_wrist"]]
    # The arm-mounted camera frequently sees only the operator's selected arm.
    # Requiring both shoulders made ID3/ID4 disappear even while the hand
    # channels remained healthy.  One reliable shoulder-elbow pair is enough
    # for lift; the wrist is only required for elbow flexion.
    if not usable(shoulder) or not usable(elbow):
        return fallback, frozenset()

    both_shoulders = usable(left_shoulder) and usable(right_shoulder)
    shoulder_width = (
        _distance(left_shoulder, right_shoulder, aspect_ratio)
        if both_shoulders
        else 0.0
    )
    upper_arm_length = _distance(shoulder, elbow, aspect_ratio)
    if shoulder_width >= 0.05:
        scale = shoulder_width
        center_x = max(0.0, min(1.0, (left_shoulder.x + right_shoulder.x) / 2.0))
    elif upper_arm_length >= 0.04:
        scale = upper_arm_length
        center_x = max(0.0, min(1.0, shoulder.x))
    else:
        return fallback, frozenset()

    valid = {"center_x"}
    pan = 0.0
    lift = 0.0
    elbow_flex = 0.0
    if usable(shoulder) and usable(elbow):
        pan = max(-1.5, min(1.5, ((elbow.x - shoulder.x) * aspect_ratio) / scale))
        lift = max(-1.5, min(1.5, -(elbow.y - shoulder.y) / scale))
        valid.update({"pan", "lift"})
        if usable(wrist):
            flexion = _elbow_flexion(shoulder, elbow, wrist, aspect_ratio)
            if flexion is not None:
                elbow_flex = max(0.0, min(1.0, flexion))
                valid.add("elbow")

    return (
        PoseFeatures(pan=pan, lift=lift, elbow=elbow_flex, center_x=center_x),
        frozenset(valid),
    )


def extract_arm_features(
    landmarks: Sequence[Landmark],
    *,
    side: str = "right",
    min_visibility: float = 0.7,
    aspect_ratio: float = 1.0,
) -> PoseFeatures | None:
    features, valid = extract_arm_feature_channels(
        landmarks,
        side=side,
        min_visibility=min_visibility,
        aspect_ratio=aspect_ratio,
    )
    required = {"pan", "lift", "elbow", "center_x"}
    return features if required.issubset(valid) else None


def median_features(samples: Sequence[PoseFeatures]) -> PoseFeatures:
    if not samples:
        raise ValueError("At least one pose sample is required.")
    return PoseFeatures(
        pan=statistics.median(sample.pan for sample in samples),
        lift=statistics.median(sample.lift for sample in samples),
        elbow=statistics.median(sample.elbow for sample in samples),
        center_x=statistics.median(sample.center_x for sample in samples),
        hand_roll=statistics.median(sample.hand_roll for sample in samples),
        hand_open=statistics.median(sample.hand_open for sample in samples),
    )

