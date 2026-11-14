from __future__ import annotations

import queue
import json
import logging
import math
import os
import threading
import time
import uuid
import copy
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .agent_client import AgentClientSettings, OpenAICompatibleClient, validate_base_url
from .agent_memory import AgentMemory
from .agent_schema import (
    InteractionPlan,
    build_fixed_intent_plan,
    build_system_prompt,
    plan_to_dict,
    parse_interaction_plan,
)
from .speech_normalizer import SpeechNormalization, normalize_language, normalize_speech


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
    language: str = "zh"

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
        return AgentSettings(
            name, validate_base_url(self.base_url), model, api_key, voice, normalize_language(self.language),
        )


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
    language: str = field(default="zh", repr=False)
    emergency_stop: bool = field(default=False, repr=False)
    queued_at: float = field(default_factory=time.monotonic, repr=False)
    plan_source: str | None = None

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
            "plan_source": self.plan_source,
        }


class AgentService:
    """Bounded cancellable cloud workers, with one current dispatch generation."""

    def __init__(
        self,
        *,
        plan_callback: PlanCallback,
        stop_callback: StopCallback,
        queue_size: int = 8,
        history_limit: int = 50,
        preferences_path: Path | None = None,
        memory_path: Path | None = None,
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
        self._threads: list[threading.Thread] = []
        self._request_cancellations: dict[str, Callable[[], None]] = {}
        self._client_sequences: dict[str, int] = {}
        self._shutdown = threading.Event()
        self._preferences_path = None if preferences_path is None else Path(preferences_path)
        if memory_path is None and self._preferences_path is not None:
            memory_path = self._preferences_path.with_name("agent_memory.json")
        self._memory = None if memory_path is None else AgentMemory(memory_path)
        self._memory_turns: queue.Queue[tuple[str, str, str]] = queue.Queue()
        self._memory_wakeup = threading.Event()
        self._memory_thread: threading.Thread | None = None
        self._memory_writer: threading.Thread | None = None
        self._memory_client: OpenAICompatibleClient | None = None
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
        defaults = {"agent_name": "", "base_url": "", "model": "", "voice": "taiwan_female", "language": "zh"}
        values = {name: data.get(name, default) for name, default in defaults.items()}
        if not all(isinstance(value, str) for value in values.values()):
            raise ValueError("Preferences contain invalid field types.")
        return AgentSettings(api_key="", **values).validated()

    @staticmethod
    def _write_preferences(path: Path, settings: AgentSettings) -> None:
        values = {"agent_name": settings.agent_name, "base_url": settings.base_url,
                  "model": settings.model, "voice": settings.voice, "language": settings.language}
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
            self._threads = [thread for thread in self._threads if thread.is_alive()]
            if self._shutdown.is_set():
                while True:
                    try:
                        self._queue.get_nowait()
                    except queue.Empty:
                        break
                    else:
                        self._queue.task_done()
            self._shutdown.clear()
            # The old socket is shut down on barge-in; spare workers also cover
            # a DNS/connect wait that the operating system cannot cancel yet.
            while len(self._threads) < 3:
                thread = threading.Thread(
                    target=self._run, name=f"so101-agent-{len(self._threads) + 1}", daemon=True,
                )
                self._threads.append(thread)
                thread.start()
            self._thread = self._threads[0]
            if self._memory is not None and (
                self._memory_writer is None or not self._memory_writer.is_alive()
            ):
                self._memory_writer = threading.Thread(
                    target=self._write_memory_turns, name="so101-agent-memory-writer", daemon=True,
                )
                self._memory_writer.start()
            if self._memory is not None and (
                self._memory_thread is None or not self._memory_thread.is_alive()
            ):
                self._memory_thread = threading.Thread(
                    target=self._run_memory, name="so101-agent-memory", daemon=True,
                )
                self._memory_wakeup.set()
                self._memory_thread.start()

    def shutdown(self, timeout_seconds: float = 5.0) -> None:
        self.cancel_all("Agent service shutdown")
        self._shutdown.set()
        self._memory_wakeup.set()
        with self._lock:
            memory_client = self._memory_client
        if memory_client is not None:
            memory_client.cancel()
        for _thread in self._threads:
            try:
                self._queue.put_nowait(None)
            except queue.Full:
                break
        deadline = time.monotonic() + max(0.0, timeout_seconds)
        for thread in self._threads:
            thread.join(max(0.0, deadline - time.monotonic()))
        if self._memory_thread is not None:
            self._memory_thread.join(max(0.0, deadline - time.monotonic()))
        if self._memory_writer is not None:
            self._memory_writer.join(max(0.0, deadline - time.monotonic()))
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
        language: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            previous_key = "" if self._settings is None else self._settings.api_key
            previous_voice = "taiwan_female" if self._settings is None else self._settings.voice
            previous_language = "zh" if self._settings is None else self._settings.language
        settings = AgentSettings(
            agent_name,
            base_url,
            model,
            previous_key if api_key is None else api_key,
            previous_voice if voice is None else voice,
            previous_language if language is None else language,
        ).validated()
        if not settings.api_key:
            raise ValueError("API key is required.")
        with self._lock:
            self._save_preferences(settings)
            self._settings = settings
            memory_client = self._memory_client
        if memory_client is not None:
            memory_client.cancel()
        self.start()
        self._memory_wakeup.set()
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
        language: str | None = None,
    ) -> dict[str, Any]:
        """Validate an OpenAI-compatible endpoint without dispatching any motion."""

        with self._lock:
            previous_key = "" if self._settings is None else self._settings.api_key
            previous_language = "zh" if self._settings is None else self._settings.language
        settings = AgentSettings(
            agent_name,
            base_url,
            model,
            previous_key if api_key is None else api_key,
            language=previous_language if language is None else language,
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
        test_text = (
            "Connection test. Confirm the connection without taking any action."
            if settings.language == "en" else "连接测试：请只回复连接成功，不执行任何动作。"
        )
        plan = client.request_plan(
            system_prompt=build_system_prompt(settings.agent_name, settings.language),
            original_text=test_text,
            normalized_text=test_text,
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
                current.agent_name, current.base_url, current.model, "", current.voice, current.language,
            )
            memory_client = self._memory_client
        if memory_client is not None:
            memory_client.cancel()

    def public_settings(self) -> dict[str, Any]:
        with self._lock:
            settings = self._settings
            running = any(thread.is_alive() for thread in self._threads)
        if settings is None:
            return {
                "configured": False,
                "agent_name": "",
                "base_url": "",
                "model": "",
                "voice": "taiwan_female",
                "language": "zh",
                "api_key_configured": False,
                "running": running,
            }
        return {
            "configured": bool(settings.api_key),
            "agent_name": settings.agent_name,
            "base_url": settings.base_url,
            "model": settings.model,
            "voice": settings.voice,
            "language": settings.language,
            "api_key_configured": bool(settings.api_key),
            "running": running,
        }

    def submit_text(self, text: str, *, context: Any = None) -> dict[str, Any]:
        # Serialize generation/sequence assignment with dispatch and other
        # submissions so an older concurrent HTTP call cannot become current.
        with self._dispatch_gate:
            return self._submit_text(text, context=context)

    def _submit_text(self, text: str, *, context: Any = None) -> dict[str, Any]:
        with self._lock:
            settings = self._settings
        normalized = normalize_speech(text, agent_name=settings.agent_name if settings else "")
        requested_language = context.get("language") if isinstance(context, dict) else None
        language = normalize_language(
            requested_language if requested_language is not None else (settings.language if settings else "zh")
        )
        if isinstance(context, dict) and not normalized.emergency_stop:
            client_id, sequence = context.get("client_id"), context.get("client_sequence")
            if sequence is not None:
                if not isinstance(client_id, str) or isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
                    raise ValueError("Invalid interaction sequence.")
                with self._lock:
                    if sequence <= self._client_sequences.get(client_id, -1):
                        raise RuntimeError("This utterance was superseded by a newer message.")
                    self._client_sequences[client_id] = sequence
                    if len(self._client_sequences) > 128:
                        del self._client_sequences[next(iter(self._client_sequences))]
            if context.get("interrupt") and not context.get("interrupt_handled"):
                self.cancel_all("new utterance superseded the previous interaction")
            if context.get("interrupt"):
                while True:
                    try:
                        self._queue.get_nowait()
                    except queue.Empty:
                        break
                    else:
                        self._queue.task_done()
        task = self._new_task(normalized, context=context, language=language)
        LOGGER.info(
            "agent utterance accepted | task=%s source_length=%d normalized_length=%d corrections=%s emergency=%s",
            task.task_id,
            len(normalized.original_text),
            len(normalized.normalized_text),
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
                    stored.reply = "Okay, I've stopped." if language == "en" else "好的，已经停止。"
                    stored.result = result
                    stored.plan_source = "local_stop"
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
        self, task_id: str, *, client_id: str, speech_duration_s: float | None = None,
        media_duration_s: float | None = None, playback_rate: float = 1.0,
    ) -> dict[str, Any]:
        """Release one prepared plan when the browser's audio actually starts."""
        if speech_duration_s is not None:
            if isinstance(speech_duration_s, bool) or not isinstance(speech_duration_s, (int, float)):
                raise ValueError("Speech duration must be a finite number.")
            if not math.isfinite(speech_duration_s) or not 0.1 <= speech_duration_s <= 120.0:
                raise ValueError("Speech duration must be between 0.1 and 120 seconds.")
        if isinstance(playback_rate, bool) or not isinstance(playback_rate, (int, float)) or not math.isfinite(playback_rate) or not 0.5 <= playback_rate <= 2:
            raise ValueError("Playback rate must be between 0.5 and 2.")
        if media_duration_s is not None and (
            isinstance(media_duration_s, bool) or not isinstance(media_duration_s, (int, float))
            or not math.isfinite(media_duration_s) or not 0.1 <= media_duration_s <= 120
        ):
            raise ValueError("Media duration must be between 0.1 and 120 seconds.")
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
                task.context["media_duration_s"] = media_duration_s
                task.context["playback_rate"] = float(playback_rate)
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
        cancel_request = None
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
                cancel_request = self._request_cancellations.get(task_id)
        if cancel_request is not None:
            cancel_request()
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
        cancel_requests = []
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
                    cancel_request = self._request_cancellations.get(task_id)
                    if cancel_request is not None:
                        cancel_requests.append(cancel_request)
                if exclude_task_id is not None:
                    excluded = self._tasks.get(exclude_task_id)
                    if excluded is not None:
                        excluded.generation = self._generation
        for cancel_request in cancel_requests:
            cancel_request()
        if cancelled_ids:
            LOGGER.warning(
                "agent tasks invalidated | reason=%s tasks=%s generation=%d",
                safe_reason,
                cancelled_ids,
                self._generation,
            )
        return cancelled_ids

    def interruption_context(self, client_id: str, execution_snapshot: dict[str, Any] | None = None) -> dict[str, Any] | None:
        """Keep prior intent and measured semantic progress, never raw angles."""
        with self._lock:
            latest = next((task for task in reversed(self._tasks.values())
                           if isinstance(task.context, dict) and task.context.get("client_id") == client_id), None)
            if latest is None or latest.emergency_stop:
                return None
            prior = latest.context.get("interruption") or {}
            utterances = list(prior.get("utterances") or [])
            utterances.append(latest.original_text)
            previous_actions = [] if latest.plan is None else plan_to_dict(latest.plan)["actions"]
            snapshot = execution_snapshot or {}
            result_motion = (latest.result or {}).get("agent_motion") if isinstance(latest.result, dict) else None
            matching_motion = bool(result_motion and snapshot.get("motion_id") is not None
                                   and snapshot.get("motion_id") == result_motion.get("motion_id"))
            if matching_motion:
                pending = snapshot.get("pending_actions", [])
                completed = snapshot.get("completed_actions", [])
            elif latest.state in {"QUEUED", "THINKING", "READY", "WAITING_CONFIRMATION", "DISPATCHING"}:
                pending = previous_actions if latest.plan is not None else prior.get("pending_actions", [])
                completed = prior.get("completed_actions", [])
            else:
                # A previously dispatched/completed turn with no active matching
                # motion must never be replayed merely because the user adds text.
                pending, completed = [], []
            def semantic_actions(value):
                if not isinstance(value, (list, tuple)):
                    return []
                result = parse_interaction_plan({
                    "version": "1.0", "reply": "进度", "actions": list(value)[:12],
                    "requires_confirmation": False,
                })
                return plan_to_dict(result)["actions"]
            safe_pending = semantic_actions(pending)
            safe_completed = semantic_actions(completed)
            current_action = snapshot.get("current_action") if matching_motion else None
            safe_current = semantic_actions([current_action])[0] if isinstance(current_action, dict) else None
            return {
                "utterances": [str(value)[:500] for value in utterances[-8:]],
                "previous_text": latest.original_text,
                "previous_reply": latest.reply or "",
                "previous_turn_pending": latest.state not in TERMINAL_STATES,
                "pending_actions": safe_pending,
                "completed_actions": safe_completed,
                "current_action": safe_current,
                "progress_note": str(snapshot.get("progress_note", "") if matching_motion else prior.get("progress_note", ""))[:600],
            }

    def interruption_resume(self, client_id: str) -> dict[str, Any] | None:
        """Carry private local frames through repeated interruptions; never send them to a model."""
        with self._lock:
            latest = next((task for task in reversed(self._tasks.values())
                           if isinstance(task.context, dict) and task.context.get("client_id") == client_id), None)
            if latest is None or latest.state in TERMINAL_STATES:
                return None
            value = latest.context.get("resume_motion")
            return copy.deepcopy(value) if isinstance(value, dict) else None

    def get_task(self, task_id: str) -> dict[str, Any]:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                raise KeyError("Unknown agent task.")
            return task.public()

    def history(self) -> list[dict[str, Any]]:
        with self._lock:
            return [task.public() for task in reversed(self._tasks.values())]

    def _new_task(self, normalized: SpeechNormalization, *, context: Any = None, language: str = "zh") -> AgentTask:
        with self._lock:
            task = AgentTask(
                task_id=uuid.uuid4().hex,
                original_text=normalized.original_text,
                normalized_text=normalized.normalized_text,
                corrections=normalized.corrections,
                context=context,
                generation=self._generation,
                language=language,
                emergency_stop=normalized.emergency_stop,
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

    def _remember_turn(self, user_text: str, reply: str) -> None:
        if self._memory is None:
            return
        with self._lock:
            api_key = self._settings.api_key if self._settings is not None else ""
        self._memory_turns.put_nowait((user_text, reply, api_key))

    def _write_memory_turns(self) -> None:
        memory = self._memory
        if memory is None:
            return
        while True:
            try:
                user_text, reply, api_key = self._memory_turns.get(timeout=0.25)
            except queue.Empty:
                if self._shutdown.is_set():
                    return
                continue
            try:
                if memory.append_turn(user_text, reply, api_key=api_key):
                    self._memory_wakeup.set()
            except (OSError, ValueError, RuntimeError) as exc:
                LOGGER.warning("agent memory could not save the completed turn | type=%s", type(exc).__name__)
            finally:
                self._memory_turns.task_done()

    def _memory_context(self) -> dict[str, Any]:
        if self._memory is None:
            return {}
        try:
            context = self._memory.context()
            return context if any(context.values()) else {}
        except (OSError, ValueError, RuntimeError) as exc:
            LOGGER.warning("agent memory could not be read | type=%s", type(exc).__name__)
            return {}

    def _run_memory(self) -> None:
        """Summarize completed dialogue without using a chat worker or the dispatch gate."""
        memory = self._memory
        if memory is None:
            return
        retry_seconds = 15
        while not self._shutdown.is_set():
            self._memory_wakeup.wait()
            self._memory_wakeup.clear()
            if self._shutdown.is_set():
                return
            with self._lock:
                settings = self._settings
            if settings is None or not settings.api_key:
                continue
            client = None
            try:
                batch = memory.next_batch()
                if batch is None:
                    continue
                client = OpenAICompatibleClient(AgentClientSettings(
                    base_url=settings.base_url, model=settings.model, api_key=settings.api_key,
                ))
                with self._lock:
                    if self._shutdown.is_set() or self._settings is not settings:
                        continue
                    self._memory_client = client
                summary = client.request_memory_summary(memory=batch.memory, messages=batch.messages)
                with self._lock:
                    still_current = self._settings is settings and not self._shutdown.is_set()
                if still_current and memory.commit(batch, summary, api_key=settings.api_key):
                    LOGGER.info("agent memory summarized | messages=%d", len(batch.messages))
                retry_seconds = 15
            except (OSError, ValueError, RuntimeError) as exc:
                with self._lock:
                    settings_still_current = self._settings is settings
                if settings_still_current and not self._shutdown.is_set():
                    LOGGER.warning("agent memory summary pending retry | type=%s", type(exc).__name__)
                    if self._shutdown.wait(retry_seconds):
                        return
                    retry_seconds = min(60, retry_seconds * 2)
            finally:
                with self._lock:
                    if self._memory_client is client:
                        self._memory_client = None
            self._memory_wakeup.set()

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
        started = time.monotonic()
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None or task.cancelled:
                return
            generation = task.generation
            task.state = "THINKING"
            task.updated_at = time.time()
            normalized_text = task.normalized_text
            language = task.language
            agent_name = self._settings.agent_name if self._settings else ""
            interruption = task.context.get("interruption") if isinstance(task.context, dict) else None
            available_actions = task.context.get("available_actions") if isinstance(task.context, dict) else None
        try:
            pending_context = bool(interruption and (
                interruption.get("pending_actions") or interruption.get("previous_turn_pending")
            ))
            plan = build_fixed_intent_plan(
                normalized_text, language=language, agent_name=agent_name,
            )
            if pending_context and not (
                plan and len(plan.actions) == 1 and plan.actions[0].name == "return_to_neutral"
            ):
                plan = None
            source = "local" if plan is not None else "cloud"
            if plan is None:
                plan = self._request_cloud_plan(task, interruption, available_actions)
        except Exception as exc:
            self._finish_failed(task_id, str(exc))
            return
        if plan is None:
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
            task.plan_source = source
            task.updated_at = time.time()
            LOGGER.info(
                "agent plan timing | task=%s source=%s queue_ms=%.0f plan_ms=%.0f",
                task_id, source, (started - task.queued_at) * 1000, (time.monotonic() - started) * 1000,
            )
            if plan.requires_confirmation:
                task.state = "WAITING_CONFIRMATION"
                LOGGER.info(
                    "agent plan waiting confirmation | task=%s actions=%s",
                    task_id,
                    [action.name for action in plan.actions],
                )
                return
            if (plan.actions and isinstance(task.context, dict) and task.context.get("defer_motion")
                    and (source == "cloud" or len(plan.actions) > 1)):
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

    def _request_cloud_plan(
        self, task: AgentTask, interruption: Any, available_actions: Any,
    ) -> InteractionPlan | None:
        with self._lock:
            settings = self._settings
            if settings is None or not settings.api_key:
                raise RuntimeError("这句话需要云端模型，请先在 Agent 设置中填写 API Key。")
            if task.cancelled or task.generation != self._generation:
                return None
            client = OpenAICompatibleClient(AgentClientSettings(
                base_url=settings.base_url, model=settings.model, api_key=settings.api_key,
            ))
            self._request_cancellations[task.task_id] = getattr(client, "cancel", lambda: None)
        LOGGER.info("agent model request started | task=%s", task.task_id)
        request_started = time.monotonic()
        try:
            memory_context = self._memory_context()
            catalog = available_actions() if callable(available_actions) else available_actions
            return client.request_plan(
                system_prompt=build_system_prompt(settings.agent_name, task.language),
                original_text=task.original_text,
                normalized_text=task.normalized_text,
                **({"conversation_context": interruption} if interruption else {}),
                **({"available_actions": catalog} if catalog else {}),
                **({"memory_context": memory_context} if memory_context else {}),
            )
        finally:
            with self._lock:
                self._request_cancellations.pop(task.task_id, None)
            LOGGER.info(
                "agent model request released | task=%s elapsed_s=%.3f cancelled=%s merged_context=%s",
                task.task_id, time.monotonic() - request_started, task.cancelled, bool(interruption),
            )

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
            dispatched_at = time.monotonic()
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
            self._remember_turn(task.original_text, plan.reply)
        LOGGER.info(
            "agent task completed | task=%s actions=%s dispatch_ms=%.0f",
            task_id,
            [action.name for action in plan.actions],
            (time.monotonic() - dispatched_at) * 1000,
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
