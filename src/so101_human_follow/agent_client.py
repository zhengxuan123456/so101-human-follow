from __future__ import annotations

import json
import ipaddress
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from .agent_schema import (
    TOOL_NAME,
    InteractionPlan,
    interaction_tool_schema,
    load_json_object,
    parse_interaction_plan,
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

    def _endpoint(self) -> str:
        suffix = "/chat/completions"
        if self.settings.base_url.endswith(suffix):
            return self.settings.base_url
        return self.settings.base_url + suffix

    def _redact(self, value: str) -> str:
        value = value.replace(self.settings.api_key, "[redacted]") if self.settings.api_key else value
        return " ".join(value.split())[:500]

    def request_plan(
        self,
        *,
        system_prompt: str,
        original_text: str,
        normalized_text: str,
    ) -> InteractionPlan:
        user_payload = json.dumps(
            {"original_text": original_text, "normalized_text": normalized_text},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        payload = {
            "model": self.settings.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_payload},
            ],
            "tools": [interaction_tool_schema()],
            "tool_choice": {"type": "function", "function": {"name": TOOL_NAME}},
            "temperature": 0.1,
        }
        # DeepSeek enables thinking mode by default.  Named/required tool_choice
        # is rejected in that mode, so use its documented non-thinking switch
        # for deterministic Function Calling.  Keep other OpenAI-compatible
        # providers untouched because many reject unknown provider extensions.
        hostname = (urlsplit(self.settings.base_url).hostname or "").lower()
        if hostname == "api.deepseek.com" or hostname.endswith(".api.deepseek.com"):
            payload["thinking"] = {"type": "disabled"}
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.settings.api_key:
            headers["Authorization"] = f"Bearer {self.settings.api_key}"
        request = urllib.request.Request(
            self._endpoint(), data=encoded, headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(request, timeout=self.settings.timeout_seconds) as response:
                body = response.read(MAX_RESPONSE_BYTES + 1)
                if len(body) > MAX_RESPONSE_BYTES:
                    raise AgentClientError("Model response is too large.")
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read(4096).decode("utf-8", errors="replace")
            except Exception:
                detail = ""
            raise AgentClientError(
                f"Model API returned HTTP {exc.code}: {self._redact(detail)}"
            ) from exc
        except urllib.error.URLError as exc:
            raise AgentClientError(f"Could not reach model API: {self._redact(str(exc.reason))}") from exc
        except TimeoutError as exc:
            raise AgentClientError("Model API request timed out.") from exc

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
