from __future__ import annotations

import json
import os
import re
import tempfile
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SUMMARY_BATCH_SIZE = 20
RECENT_MESSAGE_COUNT = 6
MAX_MESSAGE_CHARS = 500
_SUMMARY_LIMITS = {"persona": 320, "recent_summary": 1000}
_MAX_USER_FACTS = 20
_MAX_FACT_CHARS = 240
_STATE_VERSION = 1

_UNSAFE_CHARACTERS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\ud800-\udfff]")
_KEY_PATTERN = re.compile(
    r"\b(?:sk-|rk-|gh[pousr]_|github_pat_|xox[baprs]-|AIza)[\w-]{12,}", re.ASCII
)
_LABELLED_SECRET = re.compile(
    r"(?i)(\bBearer\s+|"
    r"(?:api[\s_-]*key|access[\s_-]*token|secret[\s_-]*key|password|密钥|密码)"
    r"[\"']?\s*[:：=]\s*[\"']?)([A-Za-z0-9._/+~=\-]{4,})"
)


def sanitize_memory_text(text: str, api_key: str = "") -> str:
    """Remove configured keys, common credentials and invalid JSON text characters."""

    text = _UNSAFE_CHARACTERS.sub("", text).strip()
    if api_key:
        text = text.replace(api_key, "[已隐藏]")
    text = _KEY_PATTERN.sub("[已隐藏]", text)
    return _LABELLED_SECRET.sub(r"\1[已隐藏]", text)


def _bounded_text(value: Any, limit: int, api_key: str) -> str:
    if not isinstance(value, str):
        raise ValueError("Memory values must be strings.")
    value = sanitize_memory_text(value, api_key)
    if len(value) > limit:
        raise ValueError("A memory value is longer than its allowed size.")
    return value


def validate_memory_summary(summary: Any, *, api_key: str = "") -> dict[str, Any]:
    """Validate model output and return only the supported memory fields."""

    if not isinstance(summary, Mapping) or set(summary) != {"persona", "user_facts", "recent_summary"}:
        raise ValueError("Memory summaries must include persona, user_facts and recent_summary.")
    facts = summary["user_facts"]
    if not isinstance(facts, list) or len(facts) > _MAX_USER_FACTS:
        raise ValueError("Memory user facts must be a bounded list.")
    validated = {
        field: _bounded_text(summary[field], limit, api_key)
        for field, limit in _SUMMARY_LIMITS.items()
    }
    validated["user_facts"] = list(dict.fromkeys(
        fact for raw in facts if (fact := _bounded_text(raw, _MAX_FACT_CHARS, api_key))
    ))
    return validated


@dataclass(frozen=True)
class MemoryBatch:
    through_id: int
    messages: tuple[dict[str, str], ...]
    memory: dict[str, Any]


class AgentMemory:
    """Store completed turns and their summary in one local JSON file."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._lock_path = self.path.with_name(f".{self.path.name}.lock")
        self._lock = threading.Lock()

    @contextmanager
    def _locked_file(self) -> Iterator[None]:
        # A separate lock stays valid across atomic replacement of the JSON.
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock_path.open("a+b") as stream:
                if os.name == "nt":
                    import msvcrt

                    stream.seek(0)
                    msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
                    try:
                        yield
                    finally:
                        stream.seek(0)
                        msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
                    try:
                        yield
                    finally:
                        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    def _read(self) -> dict[str, Any]:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {
                "schema_version": _STATE_VERSION,
                "summarized_through": 0,
                "memory": {"persona": "亲切、简短，带一点俏皮。", "user_facts": [], "recent_summary": ""},
                "messages": [],
            }
        if not isinstance(raw, dict) or type(raw.get("schema_version")) is not int or raw["schema_version"] != _STATE_VERSION:
            raise ValueError("Unsupported Agent memory file version.")
        through = raw.get("summarized_through")
        if type(through) is not int or through < 0 or through % SUMMARY_BATCH_SIZE:
            raise ValueError("Invalid Agent memory summary position.")
        messages = raw.get("messages")
        if not isinstance(messages, list):
            raise ValueError("Invalid Agent memory message list.")
        first_id = max(1, through - RECENT_MESSAGE_COUNT + 1)
        last_id = first_id + len(messages) - 1
        if last_id < through or last_id % 2:
            raise ValueError("Agent memory must contain complete turns.")
        checked_messages = []
        for message_id, message in enumerate(messages, first_id):
            role = "user" if message_id % 2 else "assistant"
            if (not isinstance(message, dict) or type(message.get("id")) is not int
                    or message["id"] != message_id or message.get("role") != role):
                raise ValueError("Invalid Agent memory message position or role.")
            content = _bounded_text(message.get("content"), MAX_MESSAGE_CHARS, "")
            if not content:
                raise ValueError("Agent memory messages must be nonempty.")
            checked_messages.append({"id": message_id, "role": role, "content": content})
        return {
            "schema_version": _STATE_VERSION,
            "summarized_through": through,
            "memory": validate_memory_summary(raw.get("memory")),
            "messages": checked_messages,
        }

    def _write(self, state: dict[str, Any]) -> None:
        descriptor, filename = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent
        )
        temporary = Path(filename)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
                json.dump(state, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _public_message(message: Mapping[str, Any]) -> dict[str, str]:
        return {"role": message["role"], "content": message["content"]}

    @staticmethod
    def _batch(state: dict[str, Any]) -> MemoryBatch | None:
        through = state["summarized_through"]
        pending = [message for message in state["messages"] if message["id"] > through]
        if len(pending) < SUMMARY_BATCH_SIZE:
            return None
        return MemoryBatch(
            through + SUMMARY_BATCH_SIZE,
            tuple(AgentMemory._public_message(message) for message in pending[:SUMMARY_BATCH_SIZE]),
            state["memory"],
        )

    def context(self) -> dict[str, Any]:
        with self._locked_file():
            state = self._read()
            return {
                **state["memory"],
                "recent_messages": [
                    self._public_message(message) for message in state["messages"][-RECENT_MESSAGE_COUNT:]
                ],
            }

    def append_turn(self, user_text: str, assistant_text: str, *, api_key: str = "") -> bool:
        user_text = sanitize_memory_text(user_text, api_key)[:MAX_MESSAGE_CHARS]
        assistant_text = sanitize_memory_text(assistant_text, api_key)[:MAX_MESSAGE_CHARS]
        if not user_text or not assistant_text:
            raise ValueError("Agent memory requires a completed turn.")
        with self._locked_file():
            state = self._read()
            messages = state["messages"]
            last_id = messages[-1]["id"] if messages else 0
            messages.extend((
                {"id": last_id + 1, "role": "user", "content": user_text},
                {"id": last_id + 2, "role": "assistant", "content": assistant_text},
            ))
            self._write(state)
            return last_id + 2 - state["summarized_through"] >= SUMMARY_BATCH_SIZE

    def next_batch(self) -> MemoryBatch | None:
        with self._locked_file():
            return self._batch(self._read())

    def commit(self, batch: MemoryBatch, summary: Any, *, api_key: str = "") -> bool:
        validated = validate_memory_summary(summary, api_key=api_key)
        with self._locked_file():
            state = self._read()
            if self._batch(state) != batch:
                return False
            state["memory"] = validated
            state["summarized_through"] = batch.through_id
            state["messages"] = [
                message for message in state["messages"]
                if message["id"] > batch.through_id - RECENT_MESSAGE_COUNT
            ]
            self._write(state)
            return True
