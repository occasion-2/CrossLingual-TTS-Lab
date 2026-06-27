from __future__ import annotations

from collections.abc import Callable

from crosslingual_tts_lab.backends.base import TTSBackend
from crosslingual_tts_lab.backends.coqui_xtts import CoquiXTTSBackend
from crosslingual_tts_lab.backends.cosyvoice import CosyVoiceBackend
from crosslingual_tts_lab.backends.dummy import DummyBackend
from crosslingual_tts_lab.backends.external import ExternalCommandBackend
from crosslingual_tts_lab.backends.f5_tts import F5TTSBackend
from crosslingual_tts_lab.backends.qwen_tts import QwenTTSBackend
from crosslingual_tts_lab.backends.spark_tts import SparkTTSBackend


BackendFactory = Callable[[dict], TTSBackend]


_BACKEND_FACTORIES: dict[str, BackendFactory] = {
    "dummy": lambda params: DummyBackend(),
    "coqui_xtts": lambda params: CoquiXTTSBackend(params),
    "f5_tts": lambda params: F5TTSBackend(params),
    "qwen_tts": lambda params: QwenTTSBackend(params),
    "cosyvoice": lambda params: CosyVoiceBackend(params),
    "spark_tts": lambda params: SparkTTSBackend(params),
    "external_command": lambda params: ExternalCommandBackend(params),
}

_BACKEND_ALIASES = {
    "xtts": "coqui_xtts",
    "xtts_v2": "coqui_xtts",
    "f5": "f5_tts",
    "f5tts": "f5_tts",
    "qwen": "qwen_tts",
    "qwentts": "qwen_tts",
    "qwen3_tts": "qwen_tts",
    "cosy": "cosyvoice",
    "cosy_voice": "cosyvoice",
    "spark": "spark_tts",
    "sparktts": "spark_tts",
    "command": "external_command",
    "cli": "external_command",
}


def create_backend(name: str, params: dict | None = None) -> TTSBackend:
    normalized = name.strip().lower()
    canonical = _BACKEND_ALIASES.get(normalized, normalized)
    if canonical in _BACKEND_FACTORIES:
        return _BACKEND_FACTORIES[canonical](params or {})
    raise ValueError(
        f"unknown backend {name!r}; available backends: dummy, coqui_xtts, "
        "f5_tts, qwen_tts, cosyvoice, spark_tts, external_command. "
        "Add real TTS integrations in crosslingual_tts_lab.backends."
    )
