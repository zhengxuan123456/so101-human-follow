"""Validate and sample timestamped six-joint recordings."""
from __future__ import annotations

import math
from bisect import bisect_right, insort
from dataclasses import dataclass
from itertools import pairwise
from typing import Mapping, Sequence

from .config import ALL_JOINTS


MAX_RECORDING_SECONDS = 119
MAX_RECORDING_SAMPLES = 3600
MAX_RECORDING_BYTES = 4 * 1024 * 1024
KEY_POSE_TOLERANCE = 2.0
HOLD_TOLERANCE = 1.0


@dataclass(frozen=True)
class FollowerTrajectory:
    samples: tuple[dict, ...]
    times: tuple[float, ...]

    @property
    def duration_s(self) -> float:
        return self.times[-1]

    def position_at(self, elapsed: float) -> dict[str, float]:
        return recording_position(self.samples, self.times, elapsed)

    def limited_to(self, speeds: Mapping[str, float]) -> FollowerTrajectory:
        times = [0.0]
        for index, (start, end) in enumerate(pairwise(self.samples)):
            recorded_dt = self.times[index + 1] - self.times[index]
            minimum_dt = max(
                abs(end["positions"][joint] - start["positions"][joint]) / speeds[joint]
                for joint in ALL_JOINTS
            )
            times.append(times[-1] + max(recorded_dt, minimum_dt))
        return FollowerTrajectory(self.samples, tuple(times))


def follower_trajectory(
    samples: object, limits: Mapping[str, tuple[float, float]],
) -> FollowerTrajectory:
    """Check recorded targets against the current six-axis travel envelope."""
    if not isinstance(samples, list) or not 2 <= len(samples) <= MAX_RECORDING_SAMPLES:
        raise ValueError("从臂轨迹至少需要两帧有效的位置。")
    validated = []
    times: list[float] = []
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict) or set(sample) != {"time_s", "positions"}:
            raise ValueError(f"从臂轨迹第 {index + 1} 帧格式错误。")
        timestamp = sample["time_s"]
        if (type(timestamp) not in (int, float) or not math.isfinite(timestamp)
                or not 0 <= timestamp <= MAX_RECORDING_SECONDS
                or (times and timestamp <= times[-1]) or (not times and timestamp != 0)):
            raise ValueError(f"从臂轨迹第 {index + 1} 帧时间无效。")
        positions = sample["positions"]
        if not isinstance(positions, dict) or positions.keys() != set(ALL_JOINTS):
            raise ValueError(f"从臂轨迹第 {index + 1} 帧缺少六轴位置。")
        for joint, value in positions.items():
            low, high = limits[joint]
            if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f"从臂轨迹第 {index + 1} 帧的 {joint} 超出安全范围 {low:.2f}～{high:.2f}。")
        times.append(float(timestamp))
        validated.append({"time_s": float(timestamp), "positions": dict(positions)})
    return FollowerTrajectory(tuple(validated), tuple(times))


def action_metadata(name: object, description: object = "") -> dict[str, str]:
    if not isinstance(name, str) or not 1 <= len(name.strip()) <= 40 or any(ord(c) < 32 for c in name):
        raise ValueError("动作名称需要 1～40 个可见字符。")
    if not isinstance(description, str):
        raise ValueError("动作介绍须为文字。")
    description = description.replace("\r\n", "\n").replace("\r", "\n").strip()
    if len(description) > 240 or any(ord(c) < 32 and c != "\n" for c in description):
        raise ValueError("动作介绍最多 240 个字符。")
    return {"name": name.strip(), "description": description}


def recording_position(
    samples: Sequence[dict], times: Sequence[float], elapsed: float,
) -> dict[str, float]:
    right = bisect_right(times, elapsed)
    if right == 0:
        return samples[0]["positions"]
    if right == len(times):
        return samples[-1]["positions"]
    start = samples[right - 1]["positions"]
    end = samples[right]["positions"]
    fraction = (elapsed - times[right - 1]) / (times[right] - times[right - 1])
    return {joint: start[joint] + (end[joint] - start[joint]) * fraction for joint in ALL_JOINTS}


def recording_extrema(samples: list[dict]) -> dict[str, list[float]]:
    extrema = {}
    for joint in ALL_JOINTS:
        values = [sample["positions"][joint] for sample in samples]
        extrema[joint] = [min(values), max(values)]
    return extrema


def editable_steps(samples: list[dict], limits: dict[str, tuple[float, float]]) -> list[dict]:
    times = [sample["time_s"] for sample in samples]
    duration = times[-1]

    # Keep the anchors less than 10 s apart, including across missing samples.
    partitions = max(1, math.ceil(duration / 9.9))
    boundaries = [duration * index / partitions for index in range(partitions)] + [duration]
    positions = {sample["time_s"]: sample["positions"] for sample in samples}
    for boundary in boundaries:
        if boundary not in positions:
            positions[boundary] = recording_position(samples, times, boundary)

    times = sorted(positions)
    bounded = [
        {joint: max(low, min(high, positions[time_s][joint])) for joint, (low, high) in limits.items()}
        for time_s in times
    ]
    indices = {time_s: index for index, time_s in enumerate(times)}
    anchors = [indices[time_s] for time_s in boundaries]

    # Prefer key poses with the largest deviation from the current approximation.
    # Adjacent steps must still meet the editor's 0.1 s movement minimum.
    while len(anchors) <= 32:
        largest_error = KEY_POSE_TOLERANCE
        split = None
        for start, end in pairwise(anchors):
            span = times[end] - times[start]
            for index in range(start + 1, end):
                before = times[index] - times[start]
                if before < 0.100001 or span - before < 0.100001:
                    continue
                fraction = before / span
                error = max(
                    abs(bounded[index][joint] - (
                        bounded[start][joint] + (bounded[end][joint] - bounded[start][joint]) * fraction
                    ))
                    for joint in ALL_JOINTS
                )
                if error > largest_error:
                    largest_error = error
                    split = index
        if split is None:
            break
        insort(anchors, split)

    frames = []
    for start, end in pairwise(anchors):
        interval = times[end] - times[start]
        previous = frames[-1]["targets"] if frames else bounded[start]
        stable = max(abs(bounded[end][joint] - previous[joint]) for joint in ALL_JOINTS) <= HOLD_TOLERANCE
        if stable and frames and frames[-1]["hold_s"] + interval <= 10:
            frames[-1]["hold_s"] += interval
            continue
        move = 0.1 if stable else max(0.1, interval)
        frames.append({
            "targets": bounded[end],
            "duration_s": move,
            "hold_s": max(0, interval - move),
            "mode": "parallel",
            "joint_order": list(ALL_JOINTS),
        })
    frames[0]["start_positions"] = bounded[0]
    return frames
