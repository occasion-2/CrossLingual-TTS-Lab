from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from crosslingual_tts_lab.backends.base import SynthesisResult
from crosslingual_tts_lab.device import detect_device_profile
from crosslingual_tts_lab.planner import GenerationJob


@dataclass
class CoquiXTTSBackend:
    """Zero-shot multilingual voice cloning through Coqui TTS/XTTS."""

    params: dict[str, Any] = field(default_factory=dict)
    name: str = "coqui_xtts"
    _tts: Any = field(default=None, init=False, repr=False)

    def synthesize(self, job: GenerationJob, output_dir: Path) -> SynthesisResult:
        if not job.voice.audio_path.exists():
            raise FileNotFoundError(f"reference voice audio does not exist: {job.voice.audio_path}")

        audio_path = output_dir / f"{job.id}.wav"
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        tts = self._load_tts()
        language = _map_language(job.target.language, self.params.get("language_map", {}))
        tts.tts_to_file(
            text=job.target.text,
            speaker_wav=str(job.voice.audio_path),
            language=language,
            file_path=str(audio_path),
        )
        return SynthesisResult(
            audio_path=audio_path,
            metadata={
                "backend": self.name,
                "model_name": self._model_name(),
                "language": language,
                "reference_audio_path": str(job.voice.audio_path),
                "synthetic_placeholder": False,
            },
        )

    def _load_tts(self) -> Any:
        if self._tts is None:
            try:
                from TTS.api import TTS
                import torch
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "Coqui XTTS backend requires the optional 'TTS' package. "
                    "Make sure it is installed in the current environment."
                ) from exc

            gpu = self.params.get("gpu", detect_device_profile().device == "cuda")

            # PyTorch 2.6 workaround for Coqui TTS UnpicklingError
            original_load = torch.load
            def _unsafe_load(*args, **kwargs):
                kwargs['weights_only'] = False
                return original_load(*args, **kwargs)

            try:
                torch.load = _unsafe_load
                self._tts = TTS(self._model_name(), gpu=gpu)
            finally:
                torch.load = original_load

        return self._tts

    def _model_name(self) -> str:
        return str(
            self.params.get("model_name")
            or "tts_models/multilingual/multi-dataset/xtts_v2"
        )


def _map_language(language: str, language_map: dict[str, Any]) -> str:
    return str(language_map.get(language, language))
