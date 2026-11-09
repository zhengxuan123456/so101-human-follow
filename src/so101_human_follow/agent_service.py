from __future__ import annotations

import queue
import json
import logging
import math
import os
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .agent_client import AgentClientSettings, OpenAICompatibleClient, validate_base_url
from .agent_schema import (
    InteractionPlan,
    apply_fixed_intent_rules,
    build_system_prompt,
    plan_to_dict,
)
from .speech_normalizer import SpeechNormalization, normalize_speech


PlanCallback = Callable[[InteractionPlan, Any], Any]
StopCallback = Callable[[str], Any]

TERMINAL_STATES = frozenset({"COMPLETED", "FAILED", "CANCELLED"})
LOGGER = logging.getLogger("so101_human_follow.agent")


@dataclass(frozen=True)
class AgentSettings:
    agent_name: str
    base_url: str
    model: str
    api_key: str = field(repr=False)
    voice: str = "taiwan_female"

    def validated(self) -> "AgentSettings":
        name = self.agent_name.strip()
        if not name or len(name) > 24 or any(ord(char) < 32 for char in name):
            raise ValueError("Agent name must contain 1 to 24 visible characters.")
        if any(char in "{}[]<>`" for char in name):
            raise ValueError("Agent name contains unsupported punctuation.")
        model = self.model.strip()
        if not model or len(model) > 120 or any(ord(char) < 32 for char in model):
            raise ValueError("Model name is empty or invalid.")
        api_key = self.api_key.strip()
        if len(api_key) > 4096 or any(char in "\r\n" for char in api_key):
            raise ValueError("API key is invalid.")
        voice = self.voice.strip()
        if not voice or len(voice) > 200 or any(ord(char) < 32 for char in voice):
            raise ValueError("Voice selection is invalid.")
        return AgentSettings(name, validate_base_url(self.base_url), model, api_key, voice)


@dataclass
class AgentTask:
    task_id: str
    original_text: str
    normalized_text: str
    corrections: tuple[str, ...]
    state: str = "QUEUED"
    reply: str | None = None
    plan: InteractionPlan | None = None
    result: Any = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    cancelled: bool = False
    context: Any = field(default=None, repr=False)
    generation: int = field(default=0, repr=False)

    def public(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "state": self.state,
            "original_text": self.original_text,
            "normalized_text": self.normalized_text,
            "corrections": list(self.corrections),
            "reply": self.reply,
            "plan": None if self.plan is None else plan_to_dict(self.plan),
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class AgentService:
    """Single-worker cloud agent that never owns the robot or serial bus."""

    def __init__(
        self,
        *,
        plan_callback: PlanCallback,
        stop_callback: StopCallback,
        queue_size: int = 8,
        history_limit: int = 50,
        preferences_path: Path | None = None,
    ):
        self._plan_callback = plan_callback
        self._stop_callback = stop_callback
        self._queue: queue.Queue[str | None] = queue.Queue(maxsize=queue_size)
        self._history_limit = history_limit
        self._lock = threading.RLock()
        # STOP/cancel must be totally ordered with the final local dispatch.  The
        # cloud request itself cannot be interrupted, so generation invalidates
        # its eventual result while this gate closes the last check/callback race.
        self._dispatch_gate = threading.RLock()
        self._generation = 0
        self._tasks: OrderedDict[str, AgentTask] = OrderedDict()
        self._settings: AgentSettings | None = None
        self._thread: threading.Thread | None = None
        self._shutdown = threading.Event()
        self._preferences_path = None if preferences_path is None else Path(preferences_path)
        self._load_preferences()

    def _load_preferences(self) -> None:
        path = self._preferences_path
        if path is None:
            return
        for source in (path, path.with_name(f"{path.stem}.backup.json")):
            if not source.is_file():
                continue
            try:
                self._settings = self._read_preferences(source)
            except (OSError, ValueError, TypeError):
                LOGGER.warning("agent preferences could not be loaded | backup=%s values_not_logged=true", source != path)
            else:
                LOGGER.info("agent non-secret preferences restored | backup=%s api_key_restored=false", source != path)
                return

    @staticmethod
    def _read_preferences(path: Path) -> AgentSettings:
        if path.stat().st_size > 8192:
            raise ValueError("Preferences file is too large.")
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Preferences must be an object.")
        # Explicit allowlist: even a manually edited file cannot restore a Key.
        values = {name: data.get(name, "taiwan_female" if name == "voice" else "")
                  for name in ("agent_name", "base_url", "model", "voice")}
        if not all(isinstance(value, str) for value in values.values()):
            raise ValueError("Preferences contain invalid field types.")
        return AgentSettings(api_key="", **values).validated()

    @staticmethod
    def _write_preferences(path: Path, settings: AgentSettings) -> None:
        values = {"agent_name": settings.agent_name, "base_url": settings.base_url,
                  "model": settings.model, "voice": settings.voice}
        with path.open("w", encoding="utf-8") as stream:
            json.dump(values, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())

    def _save_preferences(self, settings: AgentSettings) -> None:
        path = self._preferences_path
        if path is None:
            return
        # Some Windows project directories grant Write but not Delete/Rename.
        # Save in place under the service lock, retaining a validated previous
        # copy for recovery if a write is interrupted.  Neither file has a Key.
        stage = "create_directory"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            stage = "read_previous"
            try:
                previous = self._read_preferences(path) if path.is_file() else None
            except (OSError, ValueError, TypeError):
                previous = None  # Keep the existing backup if the primary is bad.
            backup = path.with_name(f"{path.stem}.backup.json")
            stage = "write_backup"
            if previous is not None:
                self._write_preferences(backup, previous)
            else:
                try:
                    self._read_preferences(backup)
                except (OSError, ValueError, TypeError):
                    # Even the first save has a valid recovery copy before the
                    # primary file is opened/truncated.
                    self._write_preferences(backup, settings)
            stage = "write_primary"
            self._write_preferences(path, settings)
        except OSError as exc:
            LOGGER.warning(
                "agent preferences write failed | stage=%s error_type=%s errno=%s winerror=%s values_not_logged=true",
                stage, type(exc).__name__, exc.errno, getattr(exc, "winerror", None),
            )
            raise RuntimeError("无法保存 Agent 偏好设置，请检查本地目录的写入权限。") from exc

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._shutdown.clear()
            self._thread = threading.Thread(
                target=self._run, name="so101-agent", daemon=True
            )
            self._thread.start()

    def shutdown(self, timeout_seconds: float = 5.0) -> None:
        self._shutdown.set()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        thread = self._thread
        if thread is not None:
            thread.join(timeout_seconds)
        with self._lock:
            self._settings = None

    def configure(
        self,
        *,
        agent_name: str,
        base_url: str,
        model: str,
        api_key: str | None = None,
        voice: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            previous_key = "" if self._settings is None else self._settings.api_key
            previous_voice = "taiwan_female" if self._settings is None else self._settings.voice
        settings = AgentSettings(
            agent_name,
            base_url,
            model,
            previous_key if api_key is None else api_key,
            previous_voice if voice is None else voice,
        ).validated()
        if not settings.api_key:
            raise ValueError("API key is required.")
        with self._lock:
            self._save_preferences(settings)
            self._settings = settings
        self.start()
        LOGGER.info(
            "agent settings saved | name=%s base_url=%s model=%s key_present=true",
            settings.agent_name,
            settings.base_url,
            settings.model,
        )
        return self.public_settings()

    def test_connection(
        self,
        *,
        agent_name: str,
        base_url: str,
        model: str,
        api_key: str | None = None,
    ) -> dict[str, Any]:
        """Validate an OpenAI-compatible endpoint without dispatching any motion."""

        with self._lock:
            previous_key = "" if self._settings is None else self._settings.api_key
        settings = AgentSettings(
            agent_name,
            base_url,
            model,
            previous_key if api_key is None else api_key,
        ).validated()
        if not settings.api_key:
            raise ValueError("API key is required for the connection test.")
        LOGGER.info(
            "agent connection test started | base_url=%s model=%s",
            settings.base_url,
            settings.model,
        )
        client = OpenAICompatibleClient(
            AgentClientSettings(
                base_url=settings.base_url,
                model=settings.model,
                api_key=settings.api_key,
            )
        )
        plan = client.request_plan(
            system_prompt=build_system_prompt(settings.agent_name),
            original_text="连接测试：请只回复连接成功，不执行任何动作。",
            normalized_text="连接测试：只回复，不执行动作。",
        )
        LOGGER.info(
            "agent connection test completed | actions=%d confirmation=%s",
            len(plan.actions),
            plan.requires_confirmation,
        )
        return {"reply": plan.reply, "plan": plan_to_dict(plan)}

    def clear_api_key(self) -> None:
        with self._lock:
            if self._settings is None:
                return
            current = self._settings
            self._settings = AgentSettings(
                current.agent_name, current.base_url, current.model, "", current.voice
            )

    def public_settings(self) -> dict[str, Any]:
        with self._lock:
            settings = self._settings
            running = bool(self._thread is not None and self._thread.is_alive())
        if settings is None:
            return {
                "configured": False,
                "agent_name": "",
                "base_url": "",
                "model": "",
                "voice": "taiwan_female",
                "api_key_configured": False,
                "running": running,
            }
        return {
            "configured": bool(settings.api_key),
            "agent_name": settings.agent_name,
            "base_url": settings.base_url,
            "model": settings.model,
            "voice": settings.voice,
            "api_key_configured": bool(settings.api_key),
            "running": running,
        }

    def submit_text(self, text: str, *, context: Any = None) -> dict[str, Any]:
        with self._lock:
            settings = self._settings
        normalized = normalize_speech(text, agent_name=settings.agent_name if settings else "")
        if (settings is None or not settings.api_key) and not normalized.emergency_stop:
            raise RuntimeError("Agent settings have not been configured.")
        task = self._new_task(normalized, context=context)
        LOGGER.info(
            "agent utterance accepted | task=%s source_length=%d normalized=%s corrections=%s emergency=%s",
            task.task_id,
            len(normalized.original_text),
            normalized.normalized_text,
            list(normalized.corrections),
            normalized.emergency_stop,
        )

        if normalized.emergency_stop:
            self.cancel_all(
                "voice emergency stop",
                exclude_task_id=task.task_id,
            )
            try:
                result = self._stop_callback("voice emergency stop")
            except Exception as exc:
                self._finish_failed(task.task_id, f"Emergency stop callback failed: {exc}")
            else:
                with self._lock:
                    stored = self._tasks[task.task_id]
                    stored.state = "COMPLETED"
                    stored.reply = "好的，已经停止。"
                    stored.result = result
                    stored.updated_at = time.time()
            return self.get_task(task.task_id)

        self.start()
        try:
            self._queue.put_nowait(task.task_id)
        except queue.Full as exc:
            self._finish_failed(task.task_id, "Agent request queue is busy.")
            raise RuntimeError("Agent request queue is busy.") from exc
        return self.get_task(task.task_id)

    def confirm(self, task_id: str) -> dict[str, Any]:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                raise KeyError("Unknown agent task.")
            if task.state != "WAITING_CONFIRMATION" or task.plan is None:
                raise RuntimeError("Agent task is not waiting for confirmation.")
            task.state = "DISPATCHING"
            task.updated_at = time.time()
            plan = task.plan
        self._dispatch(task_id, plan)
        return self.get_task(task_id)

    def start_ready(
        self, task_id: str, *, client_id: str, speech_duration_s: float | None = None
    ) -> dict[str, Any]:
        """Release one prepared plan when the browser's audio actually starts."""
        if speech_duration_s is not None:
            if isinstance(speech_duration_s, bool) or not isinstance(speech_duration_s, (int, float)):
                raise ValueError("Speech duration must be a finite number.")
            if not math.isfinite(speech_duration_s) or not 0.1 <= speech_duration_s <= 120.0:
                raise ValueError("Speech duration must be between 0.1 and 120 seconds.")
        with self._dispatch_gate:
            with self._lock:
                task = self._tasks.get(task_id)
                if task is None:
                    raise KeyError("Unknown agent task.")
                if task.cancelled or task.generation != self._generation:
                    raise RuntimeError("This prepared action was cancelled.")
                if not isinstance(task.context, dict) or task.context.get("client_id") != client_id:
                    raise RuntimeError("This prepared action belongs to another browser session.")
                if task.state != "READY" or task.plan is None:
                    raise RuntimeError("This action is no longer waiting for audio playback.")
                if time.time() - task.updated_at > 45.0:
                    task.cancelled = True
                    task.state = "CANCELLED"
                    task.error = "语音准备等待超时，请重新发送指令。"
                    raise RuntimeError(task.error)
                task.context["speech_duration_s"] = speech_duration_s
                task.state = "DISPATCHING"
                task.updated_at = time.time()
                plan = task.plan
            LOGGER.info(
                "agent audio started | task=%s speech_duration_s=%s actions=%s",
                task_id, speech_duration_s, [action.name for action in plan.actions],
            )
            self._dispatch(task_id, plan)
        return self.get_task(task_id)

    def cancel(self, task_id: str, *, stop_motion: bool = False) -> dict[str, Any]:
        with self._dispatch_gate:
            with self._lock:
                task = self._tasks.get(task_id)
                if task is None:
                    raise KeyError("Unknown agent task.")
                if task.state in TERMINAL_STATES:
                    return task.public()
                task.cancelled = True
                task.state = "CANCELLED"
                task.error = "Agent task cancelled."
                task.updated_at = time.time()
        if stop_motion:
            self._stop_callback("agent task cancelled")
        return self.get_task(task_id)

    def cancel_all(
        self,
        reason: str = "Agent tasks cancelled.",
        *,
        exclude_task_id: str | None = None,
    ) -> list[str]:
        """Invalidate every old cloud result before STOP/disconnect returns.

        Lock order is always dispatch_gate -> service lock.  If a dispatch already
        owns the gate, cancellation waits for that callback and the caller then
        stops the robot.  If cancellation wins, the callback can never start.
        """

        safe_reason = " ".join(str(reason).split())[:500] or "Agent tasks cancelled."
        cancelled_ids: list[str] = []
        with self._dispatch_gate:
            with self._lock:
                self._generation += 1
                for task_id, task in self._tasks.items():
                    if task_id == exclude_task_id or task.state in TERMINAL_STATES:
                        continue
                    task.cancelled = True
                    task.state = "CANCELLED"
                    task.error = safe_reason
                    task.updated_at = time.time()
                    cancelled_ids.append(task_id)
                if exclude_task_id is not None:
                    excluded = self._tasks.get(exclude_task_id)
                    if excluded is not None:
                        excluded.generation = self._generation
        if cancelled_ids:
            LOGGER.warning(
                "agent tasks invalidated | reason=%s tasks=%s generation=%d",
                safe_reason,
                cancelled_ids,
                self._generation,
            )
        return cancelled_ids

    def get_task(self, task_id: str) -> dict[str, Any]:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                raise KeyError("Unknown agent task.")
            return task.public()

    def history(self) -> list[dict[str, Any]]:
        with self._lock:
            return [task.public() for task in reversed(self._tasks.values())]

    def _new_task(self, normalized: SpeechNormalization, *, context: Any = None) -> AgentTask:
        with self._lock:
            task = AgentTask(
                task_id=uuid.uuid4().hex,
                original_text=normalized.original_text,
                normalized_text=normalized.normalized_text,
                corrections=normalized.corrections,
                context=context,
                generation=self._generation,
            )
            self._tasks[task.task_id] = task
            while len(self._tasks) > self._history_limit:
                removable = next(
                    (key for key, value in self._tasks.items() if value.state in TERMINAL_STATES),
                    None,
                )
                if removable is None:
                    break
                del self._tasks[removable]
        return task

    def _settings_snapshot(self) -> AgentSettings:
        with self._lock:
            if self._settings is None:
                raise RuntimeError("Agent settings have not been configured.")
            return self._settings

    def _run(self) -> None:
        while not self._shutdown.is_set():
            try:
                task_id = self._queue.get(timeout=0.25)
            except queue.Empty:
                continue
            if task_id is None:
                self._queue.task_done()
                break
            try:
                self._process_task(task_id)
            finally:
                self._queue.task_done()

    def _process_task(self, task_id: str) -> None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None or task.cancelled:
                return
            generation = task.generation
            task.state = "THINKING"
            task.updated_at = time.time()
            original_text = task.original_text
            normalized_text = task.normalized_text
        LOGGER.info("agent model request started | task=%s", task_id)
        try:
            settings = self._settings_snapshot()
            client = OpenAICompatibleClient(
                AgentClientSettings(
                    base_url=settings.base_url,
                    model=settings.model,
                    api_key=settings.api_key,
                )
            )
            plan = client.request_plan(
                system_prompt=build_system_prompt(settings.agent_name),
                original_text=original_text,
                normalized_text=normalized_text,
            )
            plan = apply_fixed_intent_rules(normalized_text, plan)
        except Exception as exc:
            self._finish_failed(task_id, str(exc))
            return

        with self._lock:
            task = self._tasks.get(task_id)
            if (
                task is None
                or task.cancelled
                or task.generation != generation
                or generation != self._generation
            ):
                return
            task.plan = plan
            task.reply = plan.reply
            task.updated_at = time.time()
            if plan.requires_confirmation:
                task.state = "WAITING_CONFIRMATION"
                LOGGER.info(
                    "agent plan waiting confirmation | task=%s actions=%s",
                    task_id,
                    [action.name for action in plan.actions],
                )
                return
            if plan.actions and isinstance(task.context, dict) and task.context.get("defer_motion"):
                task.state = "READY"
                LOGGER.info(
                    "agent plan ready; waiting for audio | task=%s actions=%s",
                    task_id, [action.name for action in plan.actions],
                )
                return
            task.state = "DISPATCHING"
        LOGGER.info(
            "agent plan validated | task=%s actions=%s",
            task_id,
            [action.name for action in plan.actions],
        )
        self._dispatch(task_id, plan)

    def _dispatch(self, task_id: str, plan: InteractionPlan) -> None:
        with self._dispatch_gate:
            with self._lock:
                task = self._tasks.get(task_id)
                if (
                    task is None
                    or task.cancelled
                    or task.generation != self._generation
                ):
                    return
                context = task.context
                generation = task.generation
            try:
                result = self._plan_callback(plan, context) if plan.actions else None
            except Exception as exc:
                self._finish_failed(task_id, f"Local plan dispatch failed: {exc}")
                return
            with self._lock:
                task = self._tasks.get(task_id)
                if (
                    task is None
                    or task.cancelled
                    or task.generation != generation
                    or generation != self._generation
                ):
                    return
                task.state = "COMPLETED"
                task.result = result
                task.updated_at = time.time()
        LOGGER.info(
            "agent task completed | task=%s actions=%s",
            task_id,
            [action.name for action in plan.actions],
        )

    def _finish_failed(self, task_id: str, error: str) -> None:
        safe_error = " ".join(str(error).split())[:500]
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None or task.cancelled:
                return
            task.state = "FAILED"
            task.error = safe_error
            task.updated_at = time.time()
        LOGGER.warning("agent task failed | task=%s error=%s", task_id, safe_error)
