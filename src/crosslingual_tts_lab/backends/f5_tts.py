from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from crosslingual_tts_lab.backends.base import SynthesisResult
from crosslingual_tts_lab.device import detect_device_profile
from crosslingual_tts_lab.planner import GenerationJob


@dataclass
class F5TTSBackend:
    """F5-TTS zero-shot voice cloning through the official Python API."""

    params: dict[str, Any] = field(default_factory=dict)
    name: str = "f5_tts"
    _model: Any = field(default=None, init=False, repr=False)

    def synthesize(self, job: GenerationJob, output_dir: Path) -> SynthesisResult:
        if not job.voice.audio_path.exists():
            raise FileNotFoundError(f"reference voice audio does not exist: {job.voice.audio_path}")

        audio_path = output_dir / f"{job.id}.wav"
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        file_spec = None
        if bool(self.params.get("save_spectrogram", False)):
            file_spec = str(audio_path.with_suffix(".png"))

        model = self._load_model()
        model.infer(
            ref_file=str(job.voice.audio_path),
            ref_text=self._reference_text(job),
            gen_text=job.target.text,
            file_wave=str(audio_path),
            file_spec=file_spec,
            remove_silence=bool(self.params.get("remove_silence", False)),
            seed=self.params.get("seed"),
            nfe_step=int(self.params.get("nfe_step", 32)),
            cfg_strength=float(self.params.get("cfg_strength", 2.0)),
            sway_sampling_coef=float(self.params.get("sway_sampling_coef", -1.0)),
            speed=float(self.params.get("speed", 1.0)),
            show_info=_quiet,
        )

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
                from f5_tts.api import F5TTS
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "F5-TTS backend requires the optional 'f5-tts' package. "
                    "This project does not install f5-tts on Python 3.13 because upstream "
                    "F5-TTS is constrained to older Python stacks. Use Python 3.11/3.12, for "
                    "example: `uv python install 3.11` then "
                    "`UV_CACHE_DIR=/tmp/uv-cache uv run --python 3.11 --extra f5 "
                    "python xttslab.py run --config configs/fleurs_ru_en_zh.toml "
                    "--out runs/fleurs_ru_en_zh`."
                ) from exc

            self._model = F5TTS(
                model=self._model_name(),
                ckpt_file=str(self.params.get("ckpt_file", "")),
                vocab_file=str(self.params.get("vocab_file", "")),
                ode_method=str(self.params.get("ode_method", "euler")),
                use_ema=bool(self.params.get("use_ema", True)),
                vocoder_local_path=self.params.get("vocoder_local_path"),
                device=self._device(),
                hf_cache_dir=self.params.get("hf_cache_dir"),
            )
        return self._model

    def _reference_text(self, job: GenerationJob) -> str:
        mode = str(self.params.get("ref_text_mode", "transcript"))
        if mode == "empty" or mode == "asr":
            return ""
        if mode == "literal":
            return str(self.params.get("ref_text", ""))
        return job.voice.transcript or ""

    def _model_name(self) -> str:
        return str(self.params.get("model") or self.params.get("model_name") or "F5TTS_v1_Base")

    def _device(self) -> str:
        return str(self.params.get("device") or detect_device_profile().device)


def _quiet(*args: Any, **kwargs: Any) -> None:
    return None
