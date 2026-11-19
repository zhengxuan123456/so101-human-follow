"""Local MuJoCo preview and bounded action files.  No serial or robot imports.

The renderer and mjData belong to one worker thread.  Browser requests submit
commands to that thread rather than moving an OpenGL context between Flask
request threads.  Simulation positions use LeRobot normalized units; the
piecewise mapping to MJCF radians is an editing approximation, not a verified
hardware zero/direction calibration.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import queue
import re
import threading
import time
import uuid
from concurrent.futures import Future, TimeoutError as FutureTimeout
from itertools import islice
from pathlib import Path
from typing import Any, Callable, Mapping

from flask import Blueprint, Response, abort, jsonify, request

from .config import ALL_JOINTS
from .motion_profile import cosine_progress
from .teach_trajectory import (
    MAX_RECORDING_BYTES,
    MAX_RECORDING_SAMPLES,
    MAX_RECORDING_SECONDS,
    action_metadata,
    editable_steps,
    recording_extrema,
    recording_position,
)

LOGGER = logging.getLogger("so101_human_follow.simulation")
LABELS = ("底座左右", "肩部抬落", "肘部弯曲", "腕部俯仰", "腕部旋转", "夹爪开合")
ACTION_ID = re.compile(r"^[a-f0-9]{32}$")
START_APPROACH_S = 0.5
MAX_ACTION_BYTES = 65536
MAX_COMPOSITION_ITEMS = 12
MAX_COMPOSITION_REPEATS = 8
MAX_COMPOSITION_SECONDS = 120.0
MAX_COMPOSITION_STEPS = 512
FOLLOWER_APPROACH_S = 1.0
MAPPING_NOTE = "仿真使用现有 SO-101 MJCF 模型；归一化位置与模型角度暂为近似映射，零点和方向仍需真机核对。仿真预览不会驱动真机。"


def _number(value: Any, low: float, high: float, label: str) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or (isinstance(value, float) and not math.isfinite(value))):
        raise ValueError(f"{label}必须是有限数值。")
    if not low <= value <= high:
        raise ValueError(f"{label}必须在 {low:g}～{high:g} 之间。")
    return float(value)


def _validate_recording_samples(samples_raw: Any) -> list[dict[str, Any]]:
    if not isinstance(samples_raw, list) or not 2 <= len(samples_raw) <= MAX_RECORDING_SAMPLES:
        raise ValueError(f"示教轨迹需要 2～{MAX_RECORDING_SAMPLES} 个采样点。")
    samples = []
    previous = -1.0
    for sample in samples_raw:
        if not isinstance(sample, dict):
            raise ValueError("示教采样格式不正确。")
        stamp = _number(sample.get("time_s"), 0, MAX_RECORDING_SECONDS, "示教采样时间")
        if (not samples and stamp != 0) or stamp <= previous:
            raise ValueError("示教时间须从 0 开始并严格递增。")
        positions = sample.get("positions")
        if not isinstance(positions, dict) or set(positions) != set(ALL_JOINTS):
            raise ValueError("每个示教采样点须包含六个关节的位置。")
        samples.append({"time_s": stamp, "positions": {
            joint: _number(positions[joint], 0 if joint == "gripper.pos" else -100, 100, joint)
            for joint in ALL_JOINTS
        }})
        previous = stamp
    return samples


def validate_recording(raw: Any) -> dict[str, Any]:
    samples = _validate_recording_samples(raw.get("samples") if isinstance(raw, dict) else None)
    recording = {"duration_s": samples[-1]["time_s"], "samples": samples, "ranges": recording_extrema(samples)}
    if "follower_samples" in raw:
        recording["follower_samples"] = _validate_recording_samples(raw["follower_samples"])
        fingerprint = raw.get("follower_calibration_fingerprint")
        if fingerprint is not None:
            if not isinstance(fingerprint, str) or not re.fullmatch(r"[a-f0-9]{16,64}", fingerprint):
                raise ValueError("从臂录制校准指纹无效。")
            recording["follower_calibration_fingerprint"] = fingerprint
    if "calibration_ranges" in raw:
        ranges = raw["calibration_ranges"]
        if not isinstance(ranges, dict) or set(ranges) != set(ALL_JOINTS):
            raise ValueError("示教主臂校准范围须包含六个关节。")
        calibration = {}
        for joint, pair in ranges.items():
            if (not isinstance(pair, (list, tuple)) or len(pair) != 2
                    or any(type(value) is not int for value in pair) or not 0 <= pair[0] < pair[1] <= 32767):
                raise ValueError(f"{joint} 的主臂编码器校准范围无效。")
            calibration[joint] = list(pair)
        recording["calibration_ranges"] = calibration
    return recording


def _frame_schedule(
    frame: Mapping[str, Any],
) -> tuple[str, list[str], float, float, dict[str, dict[str, float]] | None]:
    """Validate canonical group scheduling independently of hardware limits."""
    targets = frame.get("targets")
    if not isinstance(targets, dict) or not targets or set(targets) - set(ALL_JOINTS):
        raise ValueError("步骤需要至少一个已知关节目标。")
    mode = frame.get("mode", "parallel")
    if not isinstance(mode, str) or mode not in {"parallel", "sequential"}:
        raise ValueError("执行方式只能是同时运动或依次运动。")
    order = frame.get("joint_order", list(targets))
    if (not isinstance(order, list) or not all(isinstance(joint, str) for joint in order)
            or len(order) != len(targets) or len(set(order)) != len(order)
            or set(order) != set(targets)):
        raise ValueError("关节顺序必须恰好包含本步骤的每个活动关节一次。")
    if "joint_timing" in frame:
        raw_timing = frame["joint_timing"]
        if not isinstance(raw_timing, dict) or set(raw_timing) != set(targets):
            raise ValueError("请为本步骤的每个活动关节填写运动时间和停留时间。")
        timing = {}
        for joint in order:
            values = raw_timing[joint]
            if not isinstance(values, dict) or set(values) != {"duration_s", "hold_s"}:
                raise ValueError(f"{joint} 的时间设置格式不正确。")
            timing[joint] = {
                "duration_s": _number(values["duration_s"], 0.1, 10, f"{joint} 运动时间"),
                "hold_s": _number(values["hold_s"], 0, 10, f"{joint} 停留时间"),
            }
        if mode == "sequential":
            move = math.fsum(values["duration_s"] for values in timing.values())
            hold = math.fsum(values["hold_s"] for values in timing.values())
        else:
            move = max(values["duration_s"] for values in timing.values())
            hold = max(values["hold_s"] - (move - values["duration_s"]) for values in timing.values())
        return mode, list(order), move, hold, timing
    move = _number(frame.get("duration_s", 0.8), 0.1, 10, "本步骤总运动时间")
    hold = _number(frame.get("hold_s", 0), 0, 10, "停留时间")
    minimum = 0.1 * len(order) if mode == "sequential" else 0.1
    if move + 1e-9 < minimum:
        raise ValueError(f"依次运动 {len(order)} 个关节时，总运动时间至少需要 {minimum:.1f} 秒。")
    return mode, list(order), move, hold, None


def _start_positions(frame: Mapping[str, Any], targets: Mapping[str, Any]) -> dict[str, Any]:
    starts = frame.get("start_positions", {})
    if not isinstance(starts, dict) or set(starts) - set(targets):
        raise ValueError("起始位置只能设置在本步骤参与运动的关节上。")
    return starts


def expand_action_frames(frames: Any) -> list[dict[str, Any]]:
    """Compile group scheduling once for both simulation and robot execution.

    An explicit start position adds an approach before the matching motion.
    Parallel axes share a start time. Sequential axes each finish their own
    move and hold before the next axis starts. Legacy frames keep their group timing.
    """
    if not isinstance(frames, list) or not 1 <= len(frames) <= 32:
        raise ValueError("每个动作需要 1～32 个步骤。")
    expanded = []
    for frame in frames:
        if not isinstance(frame, dict):
            raise ValueError("步骤必须是 JSON 对象。")
        mode, order, move, hold, timing = _frame_schedule(frame)
        targets = {joint: _number(frame["targets"][joint], -1000, 1000, joint) for joint in order}
        starts = {joint: _number(value, -1000, 1000, f"{joint} 起点")
                  for joint, value in _start_positions(frame, targets).items()}
        if mode == "parallel":
            if starts:
                expanded.append({"targets": starts, "duration_s": START_APPROACH_S, "hold_s": 0.0})
            segment: dict[str, Any] = {"targets": targets, "duration_s": move, "hold_s": hold}
            if timing is not None:
                segment["joint_timing"] = timing
            expanded.append(segment)
        else:
            for index, joint in enumerate(order):
                if joint in starts:
                    expanded.append({"targets": {joint: starts[joint]}, "duration_s": START_APPROACH_S, "hold_s": 0.0})
                joint_time = timing[joint] if timing is not None else {
                    "duration_s": max(0.1, move / len(order)),
                    "hold_s": hold if index == len(order) - 1 else 0.0,
                }
                expanded.append({"targets": {joint: targets[joint]}, **joint_time})
    if math.fsum(step["duration_s"] + step["hold_s"] for step in expanded) > 120 + 1e-9:
        raise ValueError("含移到起点的动作总时长不能超过 120 秒。")
    return expanded


class DeletedActionError(ValueError):
    """A stale request refers to an action that was removed."""


class ActionLibrary:
    """Strict local action documents; every read revalidates the current envelope."""

    def __init__(self, directory: str | Path, limits_provider: Callable[[], Mapping]):
        self.directory = Path(directory)
        self.limits_provider = limits_provider
        self._lock = threading.RLock()

    def envelope(self) -> tuple[dict[str, tuple[float, float]], str]:
        raw = self.limits_provider()
        source = raw.get("limits", raw)
        limits = {}
        for joint in ALL_JOINTS:
            pair = source.get(joint)
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError(f"缺少 {joint} 的安全范围。")
            low = _number(pair[0], -1000, 1000, "最小位置")
            high = _number(pair[1], -1000, 1000, "最大位置")
            if low >= high:
                raise ValueError(f"{joint} 的安全范围无效。")
            limits[joint] = (low, high)
        identity = {"limits": limits, "calibration": raw.get("calibration_fingerprint", "")}
        fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
        return limits, fingerprint

    def validate(self, payload: Any, *, check_fingerprint: bool = False, for_review: bool = False) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("动作必须是 JSON 对象。")
        metadata = action_metadata(payload.get("name"), payload.get("description", ""))
        limits, fingerprint = self.envelope()
        kind = payload.get("kind")
        if check_fingerprint and kind != "composition" and payload.get("limits_fingerprint") != fingerprint:
            raise ValueError("保存后关节范围或校准已变化，请重新检查并保存此动作。")
        return_to_start = payload.get("return_to_start", True)
        if not isinstance(return_to_start, bool):
            raise ValueError("回位选项必须是布尔值。")
        identifier = payload.get("id") or uuid.uuid4().hex
        if not isinstance(identifier, str) or not ACTION_ID.fullmatch(identifier):
            raise ValueError("动作编号不正确。")
        action = {
            "schema_version": 1, "id": identifier, **metadata,
            "units": "lerobot_normalized_positions", "limits_fingerprint": fingerprint,
            "return_to_start": return_to_start,
        }
        if kind == "composition":
            return {**action, "kind": kind, "items": self._composition_items(payload.get("items"), identifier), "frames": []}
        if kind is not None:
            raise ValueError("动作类型不正确。")
        frames_raw = payload.get("frames")
        recording_only = for_review and frames_raw == [] and payload.get("recording") is not None
        if not recording_only and (not isinstance(frames_raw, list) or not 1 <= len(frames_raw) <= 32):
            raise ValueError("每个动作需要 1～32 个姿势。")
        frames = []
        for index, frame in enumerate(frames_raw):
            if not isinstance(frame, dict) or set(frame) - {
                "targets", "duration_s", "hold_s", "label", "mode", "joint_order", "start_positions", "joint_timing",
            }:
                raise ValueError(f"第 {index + 1} 个姿势格式不正确。")
            targets = self.validate_targets(frame.get("targets"), limits)
            mode, order, move, hold, timing = _frame_schedule(frame)
            starts = {joint: _number(value, *limits[joint], f"{joint} 起点")
                      for joint, value in _start_positions(frame, targets).items()}
            validated = {"targets": targets, "duration_s": move, "hold_s": hold,
                         "mode": mode, "joint_order": order}
            if starts:
                validated["start_positions"] = starts
            if timing is not None:
                validated["joint_timing"] = timing
            frames.append(validated)
        if frames:
            expanded = expand_action_frames(frames)
            duration = math.fsum(step["duration_s"] + step["hold_s"] for step in expanded)
            if return_to_start and duration > 119 + 1e-9:
                raise ValueError("含移到起点和回位的动作总时长不能超过 120 秒。")
        action["frames"] = frames
        if payload.get("recording") is not None:
            recording = validate_recording(payload["recording"])
            action.update(recording=recording, observed_limits=recording["ranges"])
        return action

    @staticmethod
    def _composition_items(raw: Any, identifier: str) -> list[dict[str, Any]]:
        if not isinstance(raw, list) or not 1 <= len(raw) <= MAX_COMPOSITION_ITEMS:
            raise ValueError(f"组合需要 1～{MAX_COMPOSITION_ITEMS} 个动作。")
        items = []
        for item in raw:
            if not isinstance(item, dict) or set(item) - {"action_id", "playback", "repeat", "hold_s", "revision"}:
                raise ValueError("组合动作条目格式不正确。")
            source = item.get("action_id")
            if not isinstance(source, str) or not ACTION_ID.fullmatch(source) or source == identifier:
                raise ValueError("请选择一个有效的已保存动作，组合不能引用自己。")
            playback = item.get("playback", "keyframes")
            if playback not in ("keyframes", "follower_trajectory"):
                raise ValueError("组合动作播放方式不正确。")
            repeat = item.get("repeat", 1)
            if type(repeat) is not int or not 1 <= repeat <= MAX_COMPOSITION_REPEATS:
                raise ValueError(f"每项重复次数必须为 1～{MAX_COMPOSITION_REPEATS} 的整数。")
            entry = {"action_id": source, "playback": playback, "repeat": repeat,
                     "hold_s": _number(item.get("hold_s", 0), 0, 10, "每次播放后的停留时间")}
            if "revision" in item:
                revision = item["revision"]
                if not isinstance(revision, str) or not re.fullmatch(r"[a-f0-9]{24}", revision):
                    raise ValueError("组合引用版本无效。")
                entry["revision"] = revision
            items.append(entry)
        return items

    def _leaf_block(self, raw: dict[str, Any], playback: str, *, physical: bool, return_to_start: bool):
        if raw.get("kind") == "composition":
            raise ValueError("组合不能嵌套其他组合。")
        if playback == "keyframes":
            action = self.validate(raw, check_fingerprint=physical)
            frames = expand_action_frames(action["frames"])
            duration = math.fsum(frame["duration_s"] + frame["hold_s"] for frame in frames)
            return {"name": "play_saved_action", "arguments": {
                "frames": frames, "return_to_start": return_to_start,
            }}, duration, len(frames)
        if playback != "follower_trajectory":
            raise ValueError("动作播放方式不正确。")
        recording = self._recording_from_action(raw)["recording"]
        samples = recording.get("follower_samples")
        if not samples:
            raise ValueError("该动作没有从臂实测轨迹。")
        arguments = {"samples": samples, "return_to_start": return_to_start}
        if physical:
            stamp = recording.get("follower_calibration_fingerprint")
            if not stamp:
                raise ValueError("此录制没有从臂校准记录，请重新录制后再试真机连续回放。")
            if stamp != self.limits_provider().get("calibration_fingerprint"):
                raise ValueError("从臂校准与录制时不同，请确认连接的是原从臂，或重新录制。")
            limits, _ = self.envelope()
            for joint, (low, high) in recording_extrema(samples).items():
                if low < limits[joint][0] or high > limits[joint][1]:
                    raise ValueError(f"从臂录制中的 {joint} 超出当前允许范围。")
            arguments["calibration_fingerprint"] = stamp
        return {"name": "play_follower_trajectory", "arguments": arguments}, samples[-1]["time_s"] + FOLLOWER_APPROACH_S, 2

    def _composition_blocks(
        self, action: dict[str, Any], *, physical: bool, bind_revisions: bool = False, check_revisions: bool = False,
    ) -> tuple[list[dict], float]:
        blocks = []
        duration = 1.0 if action["return_to_start"] else 0.0
        step_count = int(action["return_to_start"])
        leaves = {}
        for index, item in enumerate(action["items"], 1):
            identifier = item["action_id"]
            key = (identifier, item["playback"])
            try:
                if key not in leaves:
                    source = self._read_action(identifier)
                    block, length, steps = self._leaf_block(source, item["playback"], physical=physical, return_to_start=False)
                    revision_args = dict(block["arguments"])
                    revision_args.pop("calibration_fingerprint", None)
                    revision = hashlib.sha256(json.dumps(revision_args, sort_keys=True).encode()).hexdigest()[:24]
                    leaves[key] = block, length, steps, revision
                block, length, steps, revision = leaves[key]
                if check_revisions and item.get("revision") != revision:
                    raise ValueError("引用动作已修改，请预览并重新保存组合。")
            except (KeyError, DeletedActionError) as error:
                raise ValueError(f"第 {index} 项缺少已保存动作或对应轨迹。") from error
            except ValueError as error:
                raise ValueError(f"第 {index} 项无法播放。{error}") from error
            if bind_revisions:
                item["revision"] = revision
            for _ in range(item["repeat"]):
                blocks.append(block)
                if item["hold_s"]:
                    blocks.append({"name": "wait", "arguments": {"seconds": item["hold_s"]}})
            duration += item["repeat"] * (length + item["hold_s"])
            step_count += item["repeat"] * (steps + int(item["hold_s"] > 0))
            if duration > MAX_COMPOSITION_SECONDS + 1e-9 or step_count > MAX_COMPOSITION_STEPS:
                raise ValueError(f"组合总时长最多 {MAX_COMPOSITION_SECONDS:g} 秒，展开后最多 {MAX_COMPOSITION_STEPS} 步。")
        if action["return_to_start"]:
            blocks.append({"name": "return_to_plan_start", "arguments": {}})
        return blocks, duration

    def execution_blocks(self, identifier: str, playback: str = "keyframes") -> list[dict[str, Any]]:
        with self._lock:
            raw = self._read_action(identifier)
            if raw.get("kind") == "composition":
                if playback != "keyframes":
                    raise ValueError("组合按编排中选择的方式播放。")
                action = self.validate(raw, check_fingerprint=True)
                return self._composition_blocks(action, physical=True, check_revisions=True)[0]
            return_to_start = raw.get("return_to_start", True)
            if not isinstance(return_to_start, bool):
                raise ValueError("回位选项必须是布尔值。")
            block, _, _ = self._leaf_block(raw, playback, physical=True, return_to_start=return_to_start)
            return [block]

    def preview_blocks(self, action: dict[str, Any]) -> list[dict]:
        with self._lock:
            return self._composition_blocks(action, physical=False)[0]

    @staticmethod
    def validate_targets(raw: Any, limits: Mapping) -> dict[str, float]:
        if not isinstance(raw, dict) or not raw or set(raw) - set(ALL_JOINTS):
            raise ValueError("姿势需要至少一个已知关节目标。")
        return {joint: _number(value, *limits[joint], joint) for joint, value in raw.items()}

    def recording_to_draft(self, raw: Any) -> dict[str, Any]:
        recording = validate_recording(raw)
        limits, _ = self.envelope()
        follower_samples = recording.get("follower_samples")
        samples = follower_samples or recording["samples"]
        observed = recording_extrema(follower_samples) if follower_samples else recording["ranges"]
        limited = [
            joint for joint, (low, high) in limits.items()
            if observed[joint][0] < low or observed[joint][1] > high
        ]
        frames = editable_steps(samples, limits)
        expand_action_frames(frames)
        LOGGER.info("recording draft generated | source=%s samples=%d frames=%d limited_joints=%s",
                    "follower" if follower_samples else "leader", len(samples), len(frames), limited)
        return {
            "recording": recording, "frames": frames, "limits": limits, "observed_limits": observed,
            "limited_joints": limited, "return_to_start": False,
            "source": "follower" if follower_samples else "leader",
        }

    def save(self, payload: Any) -> dict[str, Any]:
        with self._lock:
            if isinstance(payload, dict) and payload.get("id"):
                try:
                    original = self._read_action(payload["id"])
                except KeyError:
                    pass
                else:
                    if original.get("kind") != payload.get("kind"):
                        raise ValueError("已保存动作的编排类型不能变更，请新建动作。")
                    payload = {"description": original.get("description", ""), **payload}
                    if original.get("recording") is not None:
                        payload["recording"] = original["recording"]
            action = self.validate(payload)
            if action.get("kind") == "composition":
                self._composition_blocks(action, physical=False, bind_revisions=True)
            self._write_action(action)
        LOGGER.info("custom action saved | id=%s frames=%d items=%d return_to_start=%s",
                    action["id"], len(action["frames"]), len(action.get("items", [])), action["return_to_start"])
        return action

    def save_recording(self, raw: dict[str, Any]) -> dict[str, Any]:
        identifier = raw.get("id")
        with self._lock:
            try:
                original = self._read_action(identifier)
            except KeyError:
                metadata = action_metadata(raw.get("name"), raw.get("description", ""))
                recording = validate_recording(raw)
                saved = {"id": identifier, **metadata, "recording": recording}
                self._write_action({
                    **saved, "schema_version": 1, "units": "lerobot_normalized_positions",
                    "limits_fingerprint": "", "frames": [], "return_to_start": False,
                    "observed_limits": recording["ranges"],
                })
                LOGGER.info("original recording saved | id=%s samples=%d", identifier, len(recording["samples"]))
                return saved
            return self._recording_from_action(original)

    def _write_action(self, action: dict[str, Any]) -> None:
        encoded = json.dumps(action, ensure_ascii=False, indent=2)
        maximum = MAX_RECORDING_BYTES if "recording" in action else MAX_ACTION_BYTES
        if len(encoded.encode("utf-8")) + 1 > maximum:
            raise ValueError("动作文件过大。")
        with self._lock:
            self.directory.mkdir(parents=True, exist_ok=True)
            path = self.directory / f"{action['id']}.json"
            backup = path.with_suffix(".backup.json")
            # This workspace permits writes but not atomic rename/delete.
            previous = encoded
            if path.is_file():
                try:
                    prior_text = path.read_text(encoding="utf-8")
                    self.validate(json.loads(prior_text), for_review=True)
                    previous = prior_text
                except (ValueError, OSError):
                    # Do not overwrite the last readable backup with a broken
                    # primary file left by an interrupted previous write.
                    previous = None if backup.is_file() else encoded
            writes = ((backup, previous), (path, encoded)) if previous is not None else ((path, encoded),)
            for target, text in writes:
                with target.open("w", encoding="utf-8", newline="\n") as handle:
                    handle.write(text + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())

    def _action_paths(self, identifier: str) -> tuple[Path, Path, Path]:
        if not isinstance(identifier, str) or not ACTION_ID.fullmatch(identifier):
            raise ValueError("动作编号不正确。")
        primary = self.directory / f"{identifier}.json"
        return primary, primary.with_suffix(".backup.json"), primary.with_suffix(".deleted")

    def delete(self, identifier: str) -> None:
        primary, backup, deleted = self._action_paths(identifier)
        with self._lock:
            already_deleted = deleted.is_file()
            if not already_deleted and not any(path.is_file() or path.is_symlink() for path in (primary, backup)):
                raise KeyError("未找到保存的动作。")
            # Retrying a saved recording or an old edit must not recreate the deleted action.
            if not already_deleted:
                deleted.touch(exist_ok=False)
            try:
                backup.unlink(missing_ok=True)
                primary.unlink(missing_ok=True)
            except OSError:
                if not already_deleted:
                    deleted.unlink(missing_ok=True)
                raise
        LOGGER.info("custom action deleted | id=%s", identifier)

    def _read_action(self, identifier: str) -> dict[str, Any]:
        primary, backup, deleted = self._action_paths(identifier)
        with self._lock:
            if deleted.is_file():
                raise DeletedActionError("这个动作已删除。")
            for path in (primary, backup):
                if not path.is_file():
                    continue
                if path.stat().st_size > MAX_RECORDING_BYTES:
                    raise ValueError("动作文件过大。")
                try:
                    raw = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
                if not isinstance(raw, dict) or raw.get("id") != identifier or raw.get("schema_version") != 1:
                    raise ValueError("动作文件版本或编号不正确。")
                return raw
        raise KeyError("未找到保存的动作。")

    def get(self, identifier: str, *, for_review: bool = False) -> dict[str, Any]:
        with self._lock:
            raw = self._read_action(identifier)
            action = self.validate(raw, check_fingerprint=not for_review, for_review=for_review)
            if not for_review:
                return action
            changed_limits = raw.get("limits_fingerprint") != action["limits_fingerprint"]
            if action.get("kind") == "composition":
                action["requires_review"] = False
                try:
                    _, action["duration_s"] = self._composition_blocks(action, physical=False, check_revisions=True)
                except ValueError as error:
                    action["requires_review"] = True
                    action["review_error"] = str(error)
                if not action["requires_review"] and self.limits_provider().get("calibration_fingerprint"):
                    try:
                        self._composition_blocks(action, physical=True, check_revisions=True)
                    except ValueError as error:
                        action["requires_review"] = True
                        action["review_error"] = str(error)
            else:
                action["requires_review"] = not action["frames"] or changed_limits
                action["duration_s"] = math.fsum(
                    frame["duration_s"] + frame["hold_s"] for frame in expand_action_frames(action["frames"])
                ) if action["frames"] else 0.0
            return action

    def get_recording(self, identifier: str) -> dict[str, Any]:
        raw = self._read_action(identifier)
        return self._recording_from_action(raw)

    @staticmethod
    def _recording_from_action(raw: dict[str, Any]) -> dict[str, Any]:
        if raw.get("recording") is None:
            raise KeyError("这个动作没有原始录制。")
        return {"id": raw["id"], **action_metadata(raw.get("name"), raw.get("description", "")),
                "recording": validate_recording(raw["recording"])}

    def _identifiers(self):
        return islice((path.stem for path in sorted(self.directory.glob("*.json"))
                       if ACTION_ID.fullmatch(path.stem)), 200)

    def list(self) -> list[dict[str, Any]]:
        result = []
        for identifier in self._identifiers():
            try:
                action = self.get(identifier, for_review=True)
                result.append({
                    "id": identifier, "name": action["name"], "description": action["description"],
                    "frame_count": len(action["frames"]), "compatible": not action["requires_review"],
                    "kind": action.get("kind"),
                    "item_count": len(action.get("items", [])),
                    "duration_s": action.get("duration_s"),
                    "review_error": action.get("review_error"),
                    "recording_only": not action["frames"] and action.get("kind") != "composition",
                    "follower_duration_s": self._follower_duration(action.get("recording")),
                })
            except DeletedActionError:
                continue
            except (ValueError, KeyError, OSError):
                result.append({"id": identifier, "name": "需重新检查的动作", "description": "", "frame_count": 0, "compatible": False})
        return result

    @staticmethod
    def _follower_duration(recording: dict | None) -> float | None:
        samples = recording.get("follower_samples") if recording is not None else None
        return samples[-1]["time_s"] if samples else None

    def list_recordings(self) -> list[dict[str, Any]]:
        result = []
        for identifier in self._identifiers():
            try:
                saved = self.get_recording(identifier)
            except (ValueError, KeyError, OSError):
                continue
            recording = saved.pop("recording")
            result.append({**saved, "duration_s": recording["duration_s"],
                           "follower_duration_s": self._follower_duration(recording),
                           "has_follower_calibration": bool(recording.get("follower_calibration_fingerprint"))})
        return result


class SimulationService:
    def __init__(self, model_path: str | Path, library: ActionLibrary):
        self.model_path = Path(model_path)
        self.library = library
        self._queue: queue.Queue = queue.Queue(maxsize=16)
        self._start_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._shutdown = threading.Event()
        self._model = self._data = self._renderer = None
        self._playing = False
        self._physics_paused = False
        self._elapsed = 0.0
        self._segments: list[dict] = []
        self._recording: tuple[list[dict], list[float]] | None = None
        self._targets: dict[str, float] = {}
        self._last_tick = 0.0
        self._active_at = 0.0
        self._last_error: str | None = None
        self._limits: dict = {}
        self._limits_fingerprint = ""

    def call(self, command: str, payload: Any = None, timeout: float = 20.0):
        with self._start_lock:
            if self._shutdown.is_set():
                raise RuntimeError("仿真服务已关闭。")
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name="so101-mujoco-preview", daemon=True)
                self._thread.start()
        future: Future = Future()
        try:
            self._queue.put_nowait((command, payload, future))
            return future.result(timeout=timeout)
        except queue.Full as error:
            raise RuntimeError("仿真请求过多，请稍后重试。") from error
        except FutureTimeout as error:
            future.cancel()
            raise RuntimeError("仿真初始化或渲染超时，请查看后端日志。") from error

    def shutdown(self):
        self._shutdown.set()
        if self._thread is not None:
            self._thread.join(timeout=3)

    def _run(self):
        try:
            while not self._shutdown.is_set():
                try:
                    command, payload, future = self._queue.get(timeout=0.01)
                except queue.Empty:
                    command = None
                if command is not None:
                    try:
                        if future.set_running_or_notify_cancel():
                            self._active_at = time.monotonic()
                            future.set_result(self._execute(command, payload))
                    except Exception as error:
                        self._last_error = str(error)
                        LOGGER.exception("simulation command failed | command=%s", command)
                        future.set_exception(error)
                    finally:
                        self._queue.task_done()
                if self._model is not None:
                    try:
                        self._advance()
                    except Exception as error:
                        self._last_error = str(error)
                        self._playing = False
                        self._physics_paused = True
                        LOGGER.exception("simulation stepping paused after fault")
        finally:
            if self._renderer is not None:
                self._renderer.close()

    def _open(self):
        if self._model is not None:
            return
        try:
            import mujoco
        except ImportError as error:
            raise RuntimeError("未安装 MuJoCo，请在项目环境安装 mujoco 或项目的 simulation 可选依赖。") from error
        if not self.model_path.is_file():
            raise ValueError("未找到 MuJoCo 模型；请配置 SO101_MUJOCO_MODEL 或提供同级 mujoco_sim/scene.xml。")
        self._mj = mujoco
        self._limits, self._limits_fingerprint = self.library.envelope()
        model = mujoco.MjModel.from_xml_path(str(self.model_path))
        if model.nu != 6:
            raise ValueError("仿真模型必须包含六个关节执行器。")
        self._indices = {}
        for joint in ALL_JOINTS:
            name = joint.removesuffix(".pos")
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if jid < 0 or aid < 0:
                raise ValueError(f"仿真模型缺少关节或执行器：{name}")
            self._indices[joint] = (int(model.jnt_qposadr[jid]), aid, tuple(model.jnt_range[jid]))
            if name == "wrist_roll":
                self._wrist_camera_body = int(model.jnt_bodyid[jid])
        model.vis.global_.offwidth = max(720, model.vis.global_.offwidth)
        model.vis.global_.offheight = max(540, model.vis.global_.offheight)
        self._data = mujoco.MjData(model)
        self._model = model
        self._camera = mujoco.MjvCamera()
        self._camera.lookat[:] = (0.03, 0, 0.18)
        self._camera.distance = 0.85
        self._camera.azimuth = 140
        self._camera.elevation = -22
        self._reset()
        self._last_error = None
        LOGGER.info("MuJoCo preview opened | joints=6 mapping=approximate physics=true")

    def _to_radians(self, joint: str, value: float) -> float:
        low, high = self._indices[joint][2]
        if joint == "gripper.pos":
            return low + max(0, min(100, value)) / 100 * (high - low)
        return high * min(1, value / 100) if value >= 0 else -low * max(-1, value / 100)

    def _to_normalized(self, joint: str, value: float) -> float:
        low, high = self._indices[joint][2]
        if joint == "gripper.pos":
            return (value - low) / (high - low) * 100
        return value / (high if value >= 0 else -low) * 100

    def _reset(self):
        self._mj.mj_resetData(self._model, self._data)
        self._targets = {joint: max(self._limits[joint][0], min(self._limits[joint][1], 50 if joint == "gripper.pos" else 0)) for joint in ALL_JOINTS}
        for joint, target in self._targets.items():
            qid, aid, _ = self._indices[joint]
            angle = self._to_radians(joint, target)
            self._data.qpos[qid] = angle
            self._data.ctrl[aid] = angle
        self._mj.mj_forward(self._model, self._data)
        self._playing = False
        self._physics_paused = False
        self._elapsed = 0.0
        self._segments = []
        self._recording = None
        self._last_tick = time.monotonic()

    def _show_recorded_pose(self, positions: dict[str, float]):
        self._targets = dict(positions)
        for joint, position in positions.items():
            qid, aid, _ = self._indices[joint]
            angle = self._to_radians(joint, position)
            self._data.qpos[qid] = angle
            self._data.ctrl[aid] = angle
        self._data.qvel[:] = 0
        self._mj.mj_forward(self._model, self._data)

    def _advance(self):
        now = time.monotonic()
        delta = max(0, now - self._last_tick)
        self._last_tick = now
        if now - self._active_at > 60:
            self._playing = False
            return
        if self._physics_paused:
            return
        if self._recording is not None:
            if self._playing:
                samples, times = self._recording
                self._elapsed = min(times[-1], self._elapsed + delta)
                self._show_recorded_pose(recording_position(samples, times, self._elapsed))
                self._playing = self._elapsed < times[-1]
            return
        delta = min(0.05, delta)
        if self._playing:
            self._elapsed += delta
            total = 0.0
            for segment in self._segments:
                end = total + segment["duration_s"] + segment["hold_s"]
                if self._elapsed <= end:
                    elapsed = self._elapsed - total
                    if "samples" in segment:
                        self._show_recorded_pose(recording_position(segment["samples"], segment["times"], elapsed))
                        return
                    timing = segment.get("joint_timing", {})
                    targets = {}
                    for joint in ALL_JOINTS:
                        duration = timing[joint]["duration_s"] if joint in timing else segment["duration_s"]
                        progress = cosine_progress(min(1, max(0, elapsed / duration)))
                        start = segment["start"][joint]
                        targets[joint] = start + (segment["targets"][joint] - start) * progress
                    self._targets = targets
                    break
                total = end
            else:
                self._elapsed = total
                self._targets = dict(self._segments[-1]["targets"])
                self._playing = False
        for joint, target in self._targets.items():
            self._data.ctrl[self._indices[joint][1]] = self._to_radians(joint, target)
        steps = max(1, min(30, round(delta / self._model.opt.timestep)))
        self._mj.mj_step(self._model, self._data, nstep=steps)
        if not all(math.isfinite(float(value)) for value in self._data.qpos):
            self._reset()
            self._last_error = "仿真状态异常，已重置；未向真机发送任何指令。"

    def _state(self):
        _, fingerprint = self.library.envelope()
        joints = []
        for index, joint in enumerate(ALL_JOINTS):
            qid, _, limits = self._indices[joint]
            angle = float(self._data.qpos[qid])
            joints.append({"id": index + 1, "name": joint, "label": LABELS[index],
                           "min": self._limits[joint][0], "max": self._limits[joint][1],
                           "target": self._targets[joint], "position": self._to_normalized(joint, angle),
                           "angle_rad": angle, "angle_deg": math.degrees(angle),
                           "model_min_deg": math.degrees(limits[0]), "model_max_deg": math.degrees(limits[1])})
        duration = (self._recording[1][-1] if self._recording is not None
                    else sum(s["duration_s"] + s["hold_s"] for s in self._segments))
        return {"ready": True, "playing": self._playing, "elapsed_s": round(self._elapsed, 3),
                "duration_s": duration,
                "contacts": int(self._data.ncon), "joints": joints, "mapping_note": MAPPING_NOTE,
                "limits_fingerprint": fingerprint, "error": self._last_error}

    def _preview_segments(self, blocks: list[dict]) -> list[dict]:
        # Physics can settle just beyond a soft joint stop; bound preview starts.
        start = {
            joint: max(low, min(high, self._to_normalized(joint, float(self._data.qpos[self._indices[joint][0]]))))
            for joint, (low, high) in self._limits.items()
        }
        segments = []
        current = start

        def add(frame):
            nonlocal current
            targets = {**current, **frame["targets"]}
            segments.append({**frame, "start": current, "targets": targets})
            current = targets

        for block in blocks:
            name, arguments = block["name"], block["arguments"]
            if name == "play_saved_action":
                for frame in arguments["frames"]:
                    add(frame)
            elif name == "play_follower_trajectory":
                samples = arguments["samples"]
                add({"targets": samples[0]["positions"], "duration_s": FOLLOWER_APPROACH_S, "hold_s": 0.0})
                add({"targets": samples[-1]["positions"], "duration_s": samples[-1]["time_s"], "hold_s": 0.0,
                     "samples": samples, "times": [sample["time_s"] for sample in samples]})
            elif name == "wait":
                add({"targets": {}, "duration_s": arguments["seconds"], "hold_s": 0.0})
            elif name == "return_to_plan_start":
                add({"targets": start, "duration_s": 1.0, "hold_s": 0.0})
        return segments

    def _execute(self, command: str, payload: Any):
        self._open()
        limits, fingerprint = self.library.envelope()
        if fingerprint != self._limits_fingerprint:
            self._limits, self._limits_fingerprint = limits, fingerprint
            self._reset()
            LOGGER.info("simulation envelope changed; preview reset")
        if command == "pose":
            updates = self.library.validate_targets(payload, limits)
            if self._recording is not None:
                self._targets = {joint: max(low, min(high, self._targets[joint]))
                                 for joint, (low, high) in limits.items()}
            self._targets.update(updates)
            self._playing = False
            self._physics_paused = False
            self._elapsed = 0.0
            self._segments = []
            self._recording = None
        elif command == "play":
            action = self.library.validate(payload)
            if action.get("kind") == "composition":
                blocks = self.library.preview_blocks(action)
            else:
                blocks = [{"name": "play_saved_action", "arguments": {"frames": expand_action_frames(action["frames"])}}]
                if action["return_to_start"]:
                    blocks.append({"name": "return_to_plan_start", "arguments": {}})
            self._segments = self._preview_segments(blocks)
            self._recording = None
            self._elapsed = 0.0
            self._last_tick = time.monotonic()
            self._playing = True
            self._physics_paused = False
        elif command == "play_recording":
            recording = validate_recording(payload.get("recording"))
            track = payload.get("track", "leader")
            if track not in ("leader", "follower"):
                raise ValueError("录制回放请选择主臂或从臂轨迹。")
            samples = recording.get("follower_samples") if track == "follower" else recording["samples"]
            if not samples:
                raise ValueError("该录制没有从臂实测轨迹。")
            self._recording = (samples, [sample["time_s"] for sample in samples])
            self._segments = []
            self._elapsed = 0.0
            self._show_recorded_pose(samples[0]["positions"])
            self._last_tick = time.monotonic()
            self._playing = True
            self._physics_paused = False
            self._last_error = None
            LOGGER.info("recording preview started | track=%s samples=%d duration_s=%.3f",
                        track, len(samples), samples[-1]["time_s"])
        elif command == "pause":
            self._playing = False
            self._physics_paused = True
            self._targets = {j: self._to_normalized(j, float(self._data.qpos[self._indices[j][0]])) for j in ALL_JOINTS}
        elif command == "reset":
            self._reset()
        elif command == "camera":
            self._camera.azimuth = _number(payload.get("azimuth"), -360, 360, "视角")
            self._camera.elevation = _number(payload.get("elevation"), -85, 0, "俯视角")
            self._camera.distance = _number(payload.get("distance"), 0.35, 2, "观察距离")
        elif command == "frame":
            if self._renderer is None:
                self._renderer = self._mj.Renderer(self._model, height=540, width=720)
            wrist_view = payload == "wrist"
            self._renderer.update_scene(self._data, camera=-1 if wrist_view else self._camera)
            if wrist_view:
                # SO-101 gripper model local +X is up and -Z points past the jaws.
                # The virtual lens is offset above the gripper without hardware calibration.
                body = self._wrist_camera_body
                rotation = self._data.xmat[body].reshape(3, 3)
                up, forward = rotation[:, 0], -rotation[:, 2]
                position = self._data.xpos[body] + 0.09 * up + 0.055 * forward
                for camera in self._renderer.scene.camera:
                    camera.pos[:] = position
                    camera.forward[:] = forward
                    camera.up[:] = up
            rgb = self._renderer.render()
            import cv2
            ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                raise RuntimeError("仿真画面编码失败。")
            return encoded.tobytes()
        elif command not in {"open", "status"}:
            raise ValueError("未知的仿真指令。")
        self._advance()
        return self._state()


def create_simulation_blueprint(
    *, require_operator: Callable, limits_provider: Callable, library: ActionLibrary,
    model_path: str | Path | None = None,
) -> tuple[Blueprint, SimulationService]:
    path = model_path or os.environ.get("SO101_MUJOCO_MODEL") or Path(__file__).resolve().parents[3] / "mujoco_sim" / "scene.xml"
    service = SimulationService(path, library)
    blueprint = Blueprint("so101_simulation", __name__, url_prefix="/api/simulation")

    @blueprint.route("/actions", methods=["GET", "POST"])
    def actions():
        require_operator()
        try:
            if request.method == "POST":
                request.max_content_length = MAX_RECORDING_BYTES
                payload = request.get_json(silent=True)
                if not isinstance(payload, dict) or payload.get("recording") is None:
                    if len(request.get_data()) > MAX_ACTION_BYTES:
                        abort(413, "动作请求过大。")
                return jsonify(ok=True, action=library.save(payload))
            return jsonify(ok=True, actions=library.list())
        except (ValueError, KeyError) as error:
            abort(400, str(error))
        except OSError as error:
            LOGGER.exception("custom action persistence failed")
            abort(503, "动作文件暂时无法读写，请检查项目目录权限。")

    @blueprint.get("/actions/<identifier>")
    def get_action(identifier):
        require_operator()
        try:
            action = library.get(identifier, for_review=True)
            requires_review = action.pop("requires_review")
            return jsonify(ok=True, action=action, requires_review=requires_review)
        except (ValueError, KeyError) as error:
            abort(400, str(error))

    @blueprint.post("/actions/<identifier>/delete")
    def delete_action(identifier):
        require_operator()
        try:
            library.delete(identifier)
        except ValueError as error:
            abort(400, str(error))
        except KeyError:
            abort(404, "未找到保存的动作。")
        except OSError:
            LOGGER.exception("custom action deletion failed | id=%s", identifier)
            abort(503, "暂时无法删除动作，请检查项目目录权限。")
        return jsonify(ok=True)

    @blueprint.get("/recordings")
    def recordings():
        require_operator()
        return jsonify(ok=True, recordings=library.list_recordings())

    @blueprint.post("/recording/draft")
    def recording_draft():
        require_operator()
        request.max_content_length = MAX_RECORDING_BYTES
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            abort(400, "示教编排请求必须是 JSON 对象。")
        try:
            if "recording_id" in payload:
                saved = library.get_recording(payload["recording_id"])
                draft = library.recording_to_draft(saved.pop("recording"))
                draft.update(saved)
            else:
                draft = library.recording_to_draft(payload.get("recording"))
            return jsonify(ok=True, draft=draft)
        except (ValueError, KeyError) as error:
            abort(400, str(error))

    @blueprint.route("/<command>", methods=["GET", "POST"])
    def simulation_command(command):
        require_operator()
        if (request.method == "GET" and command not in {"frame", "status"}) or (
            request.method == "POST" and command not in {"open", "pose", "play", "play_recording", "pause", "reset", "camera"}
        ):
            abort(404)
        if command == "play_recording":
            request.max_content_length = MAX_RECORDING_BYTES
        payload = request.get_json(silent=True) if request.method == "POST" else None
        if request.method == "POST" and not isinstance(payload, dict):
            abort(400, "仿真请求必须是 JSON 对象。")
        try:
            if command == "pose":
                payload = (payload or {}).get("targets")
            elif command == "frame":
                payload = request.args.get("view", "external")
                if payload not in {"external", "wrist"}:
                    raise ValueError("仿真视角请选择外部或腕部视角。")
            elif command == "play_recording" and "recording_id" in payload:
                saved = library.get_recording(payload["recording_id"])
                payload = {"recording": saved["recording"], "track": payload.get("track", "leader")}
            value = service.call(command, payload)
            return Response(value, mimetype="image/jpeg") if command == "frame" else jsonify(ok=True, simulation=value)
        except (ValueError, KeyError, TypeError) as error:
            abort(400, str(error))
        except (RuntimeError, OSError) as error:
            abort(503, str(error))

    return blueprint, service
