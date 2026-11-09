from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


_STOP_PATTERNS = (
    re.compile(r"(?:请|麻烦你)?(?:马上|立即|赶快|快点)?(?:停止|停下(?:来)?|别动|不要动|急停)"),
    re.compile(r"^(?:小\S{0,8}[，,]?\s*)?(?:停|停住|别动)[！!。.]?$"),
)

_PHRASE_REPLACEMENTS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(?:机械)?(?:夹抓|甲爪|手爪|爪子)"), "夹爪"),
    (re.compile(r"(?:让我)?摸一?摸头|摸头|低下头(?:让我摸)?"), "摸摸头"),
    (re.compile(r"摸好(?:了)?|摸完(?:了)?|可以抬头了|抬起来吧?|起来吧"), "摸好了"),
    (re.compile(r"张开夹爪|把夹爪打开|打开(?:一下)?夹爪"), "打开夹爪"),
    (re.compile(r"合上夹爪|把夹爪关上|关闭(?:一下)?夹爪"), "关闭夹爪"),
)

_CHINESE_COUNTS = {
    "一": "1",
    "两": "2",
    "二": "2",
    "三": "3",
    "四": "4",
    "五": "5",
    "六": "6",
    "七": "7",
    "八": "8",
    "九": "9",
    "十": "10",
}


@dataclass(frozen=True)
class SpeechNormalization:
    original_text: str
    normalized_text: str
    corrections: tuple[str, ...]
    emergency_stop: bool


def _compact_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value)
    value = value.replace("\u3000", " ")
    value = re.sub(r"[\t\r\n]+", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def _remove_agent_name(text: str, agent_name: str) -> tuple[str, bool]:
    name = _compact_text(agent_name)
    if not name:
        return text, False
    pattern = re.compile(rf"^(?:你好[，,]?\s*)?{re.escape(name)}(?:[，,、:：]\s*|\s+)")
    updated = pattern.sub("", text, count=1).strip()
    return updated, updated != text


def _normalize_counts(text: str) -> tuple[str, bool]:
    changed = False

    def replace(match: re.Match[str]) -> str:
        nonlocal changed
        changed = True
        return _CHINESE_COUNTS[match.group(1)] + match.group(2)

    result = re.sub(r"([一二两三四五六七八九十])(次|遍|下)", replace, text)
    return result, changed


def normalize_speech(text: str, *, agent_name: str = "") -> SpeechNormalization:
    """Normalize a short Chinese utterance without inventing robot intent.

    The function deliberately performs only deterministic vocabulary cleanup.  It
    never turns uncertain prose into joint positions or motion parameters.  Both
    the original and normalized forms are retained for later diagnostics.
    """

    if not isinstance(text, str):
        raise TypeError("Speech text must be a string.")
    original = _compact_text(text)
    if not original:
        raise ValueError("Speech text is empty.")
    if len(original) > 500:
        raise ValueError("Speech text is too long.")

    emergency_stop = any(pattern.search(original) for pattern in _STOP_PATTERNS)
    normalized, removed_name = _remove_agent_name(original, agent_name)
    corrections: list[str] = []
    if removed_name:
        corrections.append("removed_agent_name")

    for pattern, replacement in _PHRASE_REPLACEMENTS:
        updated = pattern.sub(replacement, normalized)
        if updated != normalized:
            corrections.append(f"term:{replacement}")
            normalized = updated

    normalized, count_changed = _normalize_counts(normalized)
    if count_changed:
        corrections.append("normalized_count")

    normalized = re.sub(r"\s*([，。！？,.!?])\s*", r"\1", normalized)
    normalized = re.sub(r"([，。！？,.!?])\1+", r"\1", normalized).strip()
    if not normalized:
        normalized = original

    emergency_stop = emergency_stop or any(
        pattern.search(normalized) for pattern in _STOP_PATTERNS
    )
    return SpeechNormalization(
        original_text=original,
        normalized_text=normalized,
        corrections=tuple(dict.fromkeys(corrections)),
        emergency_stop=emergency_stop,
    )
