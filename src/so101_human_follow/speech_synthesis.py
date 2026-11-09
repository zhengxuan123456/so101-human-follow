from __future__ import annotations

import asyncio
import logging
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass


LOGGER = logging.getLogger("so101_human_follow.speech")
MAX_SPEECH_CHARACTERS = 600
SYNTHESIS_TIMEOUT_S = 15.0
MAX_AUDIO_BYTES = 2 * 1024 * 1024
MAX_CACHE_ENTRIES = 24
MAX_CACHE_BYTES = 8 * 1024 * 1024
CACHE_TTL_S = 300.0


@dataclass(frozen=True)
class VoicePreset:
    voice: str
    rate: str = "+0%"
    pitch: str = "+0Hz"


# These are built-in voice choices, never arbitrary model-supplied SSML or URLs.
VOICE_PRESETS = {
    # Keep the saved preset identifier stable while using HsiaoYu at her native
    # pace/pitch; artificially speeding up every sentence flattens its pauses.
    "taiwan_female": VoicePreset("zh-TW-HsiaoYuNeural"),
    "puppy_male": VoicePreset("zh-CN-YunxiNeural", rate="+8%", pitch="+2Hz"),
}


def _spoken_text(text: str) -> str:
    """Remove chat-only decoration without rewriting words, names or questions."""
    text = re.sub(r"\*\*(.+?)\*\*|__(.+?)__|`([^`]+)`", lambda match: next(
        value for value in match.groups() if value is not None
    ), text)
    # A trailing chat tilde is not a spoken syllable or a request for SSML pitch.
    text = re.sub(r"[~～]+(?=\s*(?:[，。！？!?、,.；;：:]|$))", "", text)
    text = re.sub(r"([！？!?。，,])\1+", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


class SpeechSynthesisError(RuntimeError):
    pass


class SpeechUnavailableError(SpeechSynthesisError):
    pass


class SpeechBusyError(SpeechSynthesisError):
    pass


class SpeechTimeoutError(SpeechSynthesisError):
    pass


class SpeechSynthesizer:
    """Bounded online neural speech with an ephemeral in-memory audio cache.

    Only assistant text and the allowlisted voice are sent to the speech service.
    No model key, robot data, audio files or credentials are stored here.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cache: OrderedDict[tuple[str, str], tuple[float, bytes]] = OrderedDict()
        self._cache_bytes = 0

    def synthesize(self, text: str, voice: str) -> bytes:
        if not isinstance(text, str):
            raise ValueError("语音文本必须是字符串。")
        text = text.strip()
        if not text or len(text) > MAX_SPEECH_CHARACTERS:
            raise ValueError(f"语音文本需要1到{MAX_SPEECH_CHARACTERS}个字符。")
        if any(ord(char) < 32 and char not in "\r\n\t" for char in text):
            raise ValueError("语音文本包含不支持的控制字符。")
        if not isinstance(voice, str) or voice not in VOICE_PRESETS:
            raise ValueError("请选择台湾女生或年轻柔和男声。")
        text = _spoken_text(text)
        if not text:
            raise ValueError("语音文本需要包含可朗读的内容。")
        preset = VOICE_PRESETS[voice]
        # A second request fails promptly instead of building an unbounded queue
        # and playing stale speech after the operator has already moved on.
        if not self._lock.acquire(blocking=False):
            raise SpeechBusyError("语音正在生成，请稍后重试。")
        started = time.monotonic()
        try:
            self._expire_cache(started)
            cache_key = (voice, text)
            cached = self._cache.get(cache_key)
            if cached is not None:
                self._cache.move_to_end(cache_key)
                LOGGER.info(
                    "speech cache hit | voice=%s provider_voice=%s rate=%s pitch=%s chars=%d bytes=%d",
                    voice, preset.voice, preset.rate, preset.pitch, len(text), len(cached[1]),
                )
                return cached[1]
            try:
                audio = asyncio.run(
                    asyncio.wait_for(self._collect_audio(text, preset), SYNTHESIS_TIMEOUT_S)
                )
            except ImportError as exc:
                LOGGER.warning("speech synthesis unavailable | dependency=edge-tts")
                raise SpeechUnavailableError("自然语音组件尚未安装，请安装项目的 web 依赖。") from exc
            except asyncio.TimeoutError as exc:
                LOGGER.warning("speech synthesis timed out | voice=%s chars=%d", voice, len(text))
                raise SpeechTimeoutError("自然语音生成超时，请检查网络后重试。") from exc
            except Exception as exc:
                # External exception strings may contain request URLs; log only
                # the class, never text, model keys, response bodies or headers.
                LOGGER.warning(
                    "speech synthesis failed | voice=%s chars=%d error_type=%s",
                    voice, len(text), type(exc).__name__,
                )
                raise SpeechSynthesisError("自然语音暂时不可用，请检查网络或选择系统语音。") from exc
            self._cache[cache_key] = (time.monotonic(), audio)
            self._cache_bytes += len(audio)
            while len(self._cache) > MAX_CACHE_ENTRIES or self._cache_bytes > MAX_CACHE_BYTES:
                _, (_, removed) = self._cache.popitem(last=False)
                self._cache_bytes -= len(removed)
            LOGGER.info(
                "speech synthesis complete | voice=%s provider_voice=%s rate=%s pitch=%s chars=%d bytes=%d elapsed_s=%.3f",
                voice, preset.voice, preset.rate, preset.pitch, len(text), len(audio),
                time.monotonic() - started,
            )
            return audio
        finally:
            self._lock.release()

    def _expire_cache(self, now: float) -> None:
        expired = [key for key, (created, _) in self._cache.items() if now - created >= CACHE_TTL_S]
        for key in expired:
            _, removed = self._cache.pop(key)
            self._cache_bytes -= len(removed)

    @staticmethod
    async def _collect_audio(text: str, preset: VoicePreset) -> bytes:
        import edge_tts

        audio = bytearray()
        communication = edge_tts.Communicate(
            text, preset.voice, rate=preset.rate, pitch=preset.pitch,
            connect_timeout=5, receive_timeout=10,
        )
        async for chunk in communication.stream():
            if chunk.get("type") != "audio":
                continue
            audio.extend(chunk["data"])
            if len(audio) > MAX_AUDIO_BYTES:
                raise SpeechSynthesisError("Speech audio exceeded the bounded response size.")
        if not audio:
            raise SpeechSynthesisError("Speech provider returned no audio.")
        return bytes(audio)
