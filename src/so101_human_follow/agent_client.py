from __future__ import annotations

import json
import ipaddress
import asyncio
import threading
import re
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from .agent_schema import (
    MEMORY_SUMMARY_PROMPT,
    TOOL_NAME,
    InteractionPlan,
    interaction_tool_schema,
    load_json_object,
    parse_interaction_plan,
)
from .agent_memory import (
    MAX_MESSAGE_CHARS,
    RECENT_MESSAGE_COUNT,
    SUMMARY_BATCH_SIZE,
    sanitize_memory_text,
    validate_memory_summary,
)


MAX_RESPONSE_BYTES = 1_000_000


class AgentClientError(RuntimeError):
    pass


def validate_base_url(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("API base URL must be a string.")
    value = value.strip().rstrip("/")
    parsed = urlsplit(value)
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("API base URL cannot contain credentials, query text or fragments.")
    if not parsed.hostname or parsed.scheme not in {"http", "https"}:
        raise ValueError("API base URL must be an absolute HTTP(S) URL.")
    if parsed.scheme == "http":
        hostname = parsed.hostname.strip("[]").lower()
        loopback = hostname == "localhost"
        if not loopback:
            try:
                loopback = ipaddress.ip_address(hostname).is_loopback
            except ValueError:
                loopback = False
        if not loopback:
            raise ValueError("Remote model APIs must use HTTPS; HTTP is allowed only on loopback.")
    path = parsed.path.rstrip("/")
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


@dataclass(frozen=True)
class AgentClientSettings:
    base_url: str
    model: str
    api_key: str = field(repr=False)
    timeout_seconds: float = 30.0

    def validated(self) -> "AgentClientSettings":
        base_url = validate_base_url(self.base_url)
        model = self.model.strip()
        if not model or len(model) > 120 or any(ord(char) < 32 for char in model):
            raise ValueError("Model name is empty or invalid.")
        api_key = self.api_key.strip()
        if len(api_key) > 4096 or any(char in "\r\n" for char in api_key):
            raise ValueError("API key is invalid.")
        timeout = float(self.timeout_seconds)
        if not 3.0 <= timeout <= 120.0:
            raise ValueError("API timeout must be between 3 and 120 seconds.")
        return AgentClientSettings(base_url, model, api_key, timeout)


class OpenAICompatibleClient:
    def __init__(self, settings: AgentClientSettings):
        self.settings = settings.validated()
        self._cancelled = threading.Event()
        self._transport_lock = threading.Lock()
        self._event_loop = None
        self._request_task = None

    def cancel(self) -> None:
        """Interrupt the old response wait without waiting for its API timeout."""
        self._cancelled.set()
        with self._transport_lock:
            loop, task = self._event_loop, self._request_task
        if loop is not None and task is not None:
            try:
                loop.call_soon_threadsafe(task.cancel)
            except RuntimeError:
                pass

    async def _request(self, encoded: bytes, headers: dict[str, str]) -> bytes:
        try:
            import httpx
        except ImportError as exc:
            raise AgentClientError("缺少 httpx 依赖，请安装项目的 web 依赖后重试。") from exc
        with self._transport_lock:
            self._event_loop = asyncio.get_running_loop()
            self._request_task = asyncio.current_task()
        if self._cancelled.is_set():
            raise asyncio.CancelledError()
        endpoint = self._endpoint()
        parsed = urlsplit(endpoint)
        # Preserve urllib's Windows/system proxy discovery and bypass rules,
        # while the async transport makes cancellation effective on Windows.
        bypass = urllib.request.proxy_bypass(parsed.hostname or "")
        proxy = None if bypass else urllib.request.getproxies().get(parsed.scheme)
        timeout = httpx.Timeout(self.settings.timeout_seconds, connect=min(5.0, self.settings.timeout_seconds))
        try:
            async with httpx.AsyncClient(proxy=proxy, trust_env=not bypass, timeout=timeout) as client:
                async with client.stream("POST", endpoint, content=encoded, headers=headers) as response:
                    body = bytearray()
                    limit = 4096 if response.is_error else MAX_RESPONSE_BYTES
                    async for part in response.aiter_bytes():
                        body.extend(part)
                        if len(body) > limit:
                            if response.is_error:
                                del body[limit:]
                                break
                            raise AgentClientError("Model response is too large.")
                    if response.is_error:
                        detail = self._redact(body.decode("utf-8", errors="replace"))
                        raise AgentClientError(f"Model API returned HTTP {response.status_code}: {detail}")
                    return bytes(body)
        except httpx.TimeoutException as exc:
            raise AgentClientError("Model API request timed out.") from exc
        except httpx.HTTPError as exc:
            raise AgentClientError(f"Could not reach model API: {self._redact(str(exc))}") from exc
        finally:
            with self._transport_lock:
                self._event_loop = None
                self._request_task = None

    def _endpoint(self) -> str:
        suffix = "/chat/completions"
        if self.settings.base_url.endswith(suffix):
            return self.settings.base_url
        return self.settings.base_url + suffix

    def _redact(self, value: str) -> str:
        value = value.replace(self.settings.api_key, "[redacted]") if self.settings.api_key else value
        return " ".join(value.split())[:500]

    def _send_completion(self, payload: dict[str, Any]) -> bytes:
        if self._cancelled.is_set():
            raise AgentClientError("Model request cancelled.")
        payload["model"] = self.settings.model
        # The existing DeepSeek integration uses non-thinking mode for tool calls.
        # A memory summary also needs only the final structured response.
        hostname = (urlsplit(self.settings.base_url).hostname or "").lower()
        if hostname == "api.deepseek.com" or hostname.endswith(".api.deepseek.com"):
            payload["thinking"] = {"type": "disabled"}
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.settings.api_key:
            headers["Authorization"] = f"Bearer {self.settings.api_key}"
        try:
            body = asyncio.run(self._request(encoded, headers))
        except asyncio.CancelledError as exc:
            raise AgentClientError("Model request cancelled.") from exc
        if self._cancelled.is_set():
            raise AgentClientError("Model request cancelled.")
        return body

    def _memory_messages(
        self, messages: Sequence[Mapping[str, str]], *, limit: int,
    ) -> list[dict[str, str]]:
        safe_messages = []
        for message in messages[-limit:]:
            role = message["role"]
            if role not in {"user", "assistant"}:
                raise ValueError("Memory messages must have a user or assistant role.")
            content = sanitize_memory_text(message["content"], api_key=self.settings.api_key)
            safe_messages.append({"role": role, "content": content[:MAX_MESSAGE_CHARS]})
        return safe_messages

    def request_memory_summary(
        self, *, memory: dict[str, Any], messages: Sequence[Mapping[str, str]],
    ) -> dict[str, Any]:
        """Ask the configured model for bounded dialogue memory without tools."""
        user_payload = {
            "memory": validate_memory_summary(memory, api_key=self.settings.api_key),
            "messages": self._memory_messages(messages, limit=SUMMARY_BATCH_SIZE),
        }
        payload = {
            "messages": [
                {"role": "system", "content": MEMORY_SUMMARY_PROMPT},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False, separators=(",", ":"))},
            ],
            "response_format": {"type": "json_object"},
            "max_tokens": 2800,
            "temperature": 0.1,
        }
        try:
            body = self._send_completion(payload)
        except AgentClientError:
            # Provider errors can echo the request. Avoid returning dialogue in error logs.
            raise AgentClientError("Memory summary model request failed.") from None
        try:
            message = json.loads(body.decode("utf-8"))["choices"][0]["message"]
            if message.get("tool_calls") or message.get("function_call"):
                raise ValueError("Memory summary cannot use tools.")
            summary = load_json_object(message["content"])
            return validate_memory_summary(summary, api_key=self.settings.api_key)
        except (AttributeError, KeyError, IndexError, TypeError, ValueError):
            raise AgentClientError("Model did not return a valid memory summary.") from None

    def request_plan(
        self,
        *,
        system_prompt: str,
        original_text: str,
        normalized_text: str,
        conversation_context: dict[str, Any] | None = None,
        available_actions: list[dict[str, Any]] | None = None,
        memory_context: dict[str, Any] | None = None,
    ) -> InteractionPlan:
        if self._cancelled.is_set():
            raise AgentClientError("Model request cancelled.")
        available = []
        for item in (available_actions or [])[:32]:
            if not isinstance(item, dict):
                continue
            action_id = item.get("id", item.get("action_id"))
            name = item.get("name")
            if isinstance(action_id, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,64}", action_id) and isinstance(name, str):
                entry = {"id": action_id, "name": name[:80]}
                description = item.get("description")
                if isinstance(description, str) and description.strip():
                    entry["description"] = description.strip()[:240]
                available.append(entry)
        user_payload: dict[str, Any] = {"original_text": original_text, "normalized_text": normalized_text}
        if conversation_context:
            user_payload["interruption_context"] = conversation_context
        if available:
            user_payload["available_actions"] = available
        if memory_context:
            summary = {key: value for key, value in memory_context.items() if key != "recent_messages"}
            user_payload["memory_context"] = {
                **validate_memory_summary(summary, api_key=self.settings.api_key),
                "recent_messages": self._memory_messages(
                    memory_context.get("recent_messages", ()), limit=RECENT_MESSAGE_COUNT,
                ),
            }
        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False, separators=(",", ":"))},
            ],
            "tools": [interaction_tool_schema()],
            "tool_choice": {"type": "function", "function": {"name": TOOL_NAME}},
            "temperature": 0.1,
        }
        try:
            body = self._send_completion(payload)
        except AgentClientError as exc:
            detail = str(exc)
            if memory_context and detail.startswith("Model API returned HTTP "):
                # Provider error bodies may echo the historical dialogue.
                raise AgentClientError(detail.split(":", 1)[0]) from None
            raise

        try:
            response_payload = json.loads(body.decode("utf-8"))
            choices = response_payload["choices"]
            message = choices[0]["message"]
            calls = message["tool_calls"]
        except (KeyError, IndexError, TypeError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise AgentClientError("Model response did not contain a valid tool call.") from exc
        if not isinstance(calls, list) or len(calls) != 1:
            raise AgentClientError("Model must return exactly one interaction-plan tool call.")
        function = calls[0].get("function") if isinstance(calls[0], dict) else None
        if not isinstance(function, dict) or function.get("name") != TOOL_NAME:
            raise AgentClientError("Model called an unexpected function.")
        arguments = function.get("arguments")
        if isinstance(arguments, dict):
            raw_plan = arguments
        else:
            raw_plan = load_json_object(arguments)
        return parse_interaction_plan(raw_plan)
