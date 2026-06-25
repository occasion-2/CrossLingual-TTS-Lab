from __future__ import annotations

import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from crosslingual_tts_lab.backends.base import SynthesisResult
from crosslingual_tts_lab.device import detect_device_profile
from crosslingual_tts_lab.planner import GenerationJob


@dataclass
class QwenTTSBackend:
    """Qwen3-TTS zero-shot voice cloning through the official Python API."""

    params: dict[str, Any] = field(default_factory=dict)
    name: str = "qwen_tts"
    _model: Any = field(default=None, init=False, repr=False)

    def synthesize(self, job: GenerationJob, output_dir: Path) -> SynthesisResult:
        if not job.voice.audio_path.exists():
            raise FileNotFoundError(f"reference voice audio does not exist: {job.voice.audio_path}")

        audio_path = output_dir / f"{job.id}.wav"
        audio_path.parent.mkdir(parents=True, exist_ok=True)

        model = self._load_model()
        language = _map_language(job.target.language)
        ref_text = self._reference_text(job)
        x_vector_only_mode = self._x_vector_only_mode(ref_text)

        wavs, sr = model.generate_voice_clone(
            text=job.target.text,
            language=language,
            ref_audio=str(job.voice.audio_path),
            ref_text=ref_text,
            x_vector_only_mode=x_vector_only_mode,
        )

        if isinstance(wavs, (list, tuple)):
            wav_data = wavs[0]
        else:
            wav_data = wavs

        if hasattr(wav_data, "cpu"):
            wav_data = wav_data.cpu().numpy()

        _write_wav(audio_path, wav_data, int(sr))

        return SynthesisResult(
            audio_path=audio_path,
            metadata={
                "backend": self.name,
                "model_name": self._model_name(),
                "reference_audio_path": str(job.voice.audio_path),
                "ref_text_mode": self.params.get("ref_text_mode", "transcript"),
                "x_vector_only_mode": x_vector_only_mode,
                "target_language": job.target.language,
                "mapped_language": language,
                "device": self._device(),
                "synthetic_placeholder": False,
            },
        )

    def _load_model(self) -> Any:
        if self._model is None:
            try:
                from qwen_tts import Qwen3TTSModel
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "Qwen3-TTS backend requires the optional 'qwen-tts' package. "
                    "Install qwen-tts (e.g. `pip install qwen-tts`) to use backend='qwen_tts'."
                ) from exc

            import torch

            dtype_str = str(self.params.get("dtype") or "bfloat16").lower()
            if dtype_str == "float16":
                dtype = torch.float16
            elif dtype_str == "float32":
                dtype = torch.float32
            else:
                dtype = torch.bfloat16

            self._model = Qwen3TTSModel.from_pretrained(
                self._model_name(),
                device_map=self._device_map(),
                dtype=dtype,
            )
        return self._model

    def _reference_text(self, job: GenerationJob) -> str:
        mode = str(self.params.get("ref_text_mode", "transcript"))
        if mode == "empty" or mode == "asr":
            return ""
        if mode == "literal":
            return str(self.params.get("ref_text", ""))
        return job.voice.transcript or ""

    def _x_vector_only_mode(self, ref_text: str) -> bool:
        if "x_vector_only_mode" in self.params:
            return bool(self.params["x_vector_only_mode"])
        return not bool(ref_text)

    def _model_name(self) -> str:
        return str(self.params.get("model") or self.params.get("model_name") or "Qwen/Qwen3-TTS-12Hz-1.7B-Base")

    def _device(self) -> str:
        return str(self.params.get("device") or detect_device_profile().device)

    def _device_map(self) -> str:
        device = self._device()
        if device == "cuda":
            return "cuda:0"
        return device


def _map_language(lang_code: str) -> str:
    mapping = {
        "en": "English",
        "zh": "Chinese",
        "ru": "Russian",
        "ja": "Japanese",
        "ko": "Korean",
        "de": "German",
        "fr": "French",
        "pt": "Portuguese",
        "es": "Spanish",
        "it": "Italian",
    }
    normalized = lang_code.strip().lower()
    return mapping.get(normalized, mapping.get(normalized.split("-")[0], "English"))


def _write_wav(path: Path, wav_data: Any, sample_rate: int) -> None:
    try:
        import soundfile as sf
    except ModuleNotFoundError:
        _write_wav_stdlib(path, wav_data, sample_rate)
        return

    sf.write(str(path), wav_data, sample_rate)


def _write_wav_stdlib(path: Path, wav_data: Any, sample_rate: int) -> None:
    if hasattr(wav_data, "tolist"):
        wav_data = wav_data.tolist()
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        frames = bytearray()
        for sample in wav_data:
            if isinstance(sample, (list, tuple)):
                sample = sample[0] if sample else 0.0
            value = max(-1.0, min(1.0, float(sample)))
            frames.extend(int(value * 32767).to_bytes(2, "little", signed=True))
        handle.writeframes(bytes(frames))
