from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from crosslingual_tts_lab.backends.base import SynthesisResult
from crosslingual_tts_lab.device import detect_device_profile
from crosslingual_tts_lab.planner import GenerationJob


@dataclass
class CosyVoiceBackend:
    """CosyVoice zero-shot voice cloning through the official Python API."""

    params: dict[str, Any] = field(default_factory=dict)
    name: str = "cosyvoice"
    _model: Any = field(default=None, init=False, repr=False)

    def synthesize(self, job: GenerationJob, output_dir: Path) -> SynthesisResult:
        if not job.voice.audio_path.exists():
            raise FileNotFoundError(f"reference voice audio does not exist: {job.voice.audio_path}")

        audio_path = output_dir / f"{job.id}.wav"
        audio_path.parent.mkdir(parents=True, exist_ok=True)

        model = self._load_model()
        import numpy as np

        prompt_speech = str(job.voice.audio_path)
        prompt_text = self._reference_text(job)
        if "CosyVoice3" in self._model_name() and "<|endofprompt|>" not in prompt_text:
            prompt_text = "You are a helpful assistant.<|endofprompt|>" + prompt_text

        chunks = []
        for result in model.inference_zero_shot(job.target.text, prompt_text, prompt_speech):
            if isinstance(result, dict) and "tts_speech" in result:
                val = result["tts_speech"]
            elif hasattr(result, "tts_speech"):
                val = result.tts_speech
            else:
                val = result

            if hasattr(val, "cpu"):
                val = val.cpu().numpy()
            chunks.append(val)

        if not chunks:
            raise RuntimeError("CosyVoice generated empty audio")

        wav_data = np.concatenate(chunks, axis=-1)
        if len(wav_data.shape) > 1:
            wav_data = wav_data.flatten()

        from crosslingual_tts_lab.backends.qwen_tts import _write_wav
        _write_wav(audio_path, wav_data, self._sample_rate(model))

        return SynthesisResult(
            audio_path=audio_path,
            metadata={
                "backend": self.name,
                "model": self._model_name(),
                "reference_audio_path": str(job.voice.audio_path),
                "ref_text_mode": self.params.get("ref_text_mode", "transcript"),
                "target_language": job.target.language,
                "device": self._device(),
                "synthetic_placeholder": False,
            },
        )

    def _load_model(self) -> Any:
        if self._model is None:
            try:
                import sys
                from pathlib import Path
                project_root = Path(__file__).resolve().parent.parent.parent.parent
                cosy_path = str((project_root / "CosyVoice").resolve())
                if cosy_path not in sys.path:
                    sys.path.insert(0, cosy_path)
                matcha_path = str((project_root / "CosyVoice/third_party/Matcha-TTS").resolve())
                if matcha_path not in sys.path:
                    sys.path.insert(0, matcha_path)
                from cosyvoice.cli.cosyvoice import AutoModel
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "CosyVoice backend requires the 'cosyvoice' package. "
                    "Make sure CosyVoice is installed in the current environment."
                ) from exc

            self._model = AutoModel(model_dir=self._model_name())
        return self._model

    def _reference_text(self, job: GenerationJob) -> str:
        mode = str(self.params.get("ref_text_mode", "transcript"))
        if mode == "empty" or mode == "asr":
            return ""
        if mode == "literal":
            return str(self.params.get("ref_text", ""))
        return job.voice.transcript or ""

    def _model_name(self) -> str:
        return str(
            self.params.get("model")
            or self.params.get("model_name")
            or "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
        )

    def _device(self) -> str:
        return str(self.params.get("device") or detect_device_profile().device)

    def _sample_rate(self, model: Any) -> int:
        sample_rate = getattr(model, "sample_rate", None)
        if isinstance(sample_rate, (int, float)) and sample_rate > 0:
            return int(sample_rate)

        configured = self.params.get("sample_rate", 22050)
        if isinstance(configured, (int, float)) and configured > 0:
            return int(configured)

        raise ValueError(f"invalid CosyVoice sample_rate: {configured!r}")
