from __future__ import annotations

from crosslingual_tts_lab.backends.base import TTSBackend
from crosslingual_tts_lab.backends.coqui_xtts import CoquiXTTSBackend
from crosslingual_tts_lab.backends.dummy import DummyBackend
from crosslingual_tts_lab.backends.external import ExternalCommandBackend
from crosslingual_tts_lab.backends.f5_tts import F5TTSBackend
from crosslingual_tts_lab.backends.qwen_tts import QwenTTSBackend


def create_backend(name: str, params: dict | None = None) -> TTSBackend:
    normalized = name.strip().lower()
    if normalized == "dummy":
        return DummyBackend()
    if normalized in {"coqui_xtts", "xtts", "xtts_v2"}:
        return CoquiXTTSBackend(params or {})
    if normalized in {"f5_tts", "f5", "f5tts"}:
        return F5TTSBackend(params or {})
    if normalized in {"qwen_tts", "qwen", "qwentts", "qwen3_tts"}:
        return QwenTTSBackend(params or {})
    if normalized in {"external_command", "command", "cli"}:
        return ExternalCommandBackend(params or {})
    raise ValueError(
        f"unknown backend {name!r}; available backends: dummy, coqui_xtts, "
        "f5_tts, qwen_tts, external_command. "
        "Add real TTS integrations in crosslingual_tts_lab.backends."
    )
