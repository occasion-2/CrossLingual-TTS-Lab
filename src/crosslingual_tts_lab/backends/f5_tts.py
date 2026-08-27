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
    _checkpoint_path: str | None = field(default=None, init=False, repr=False)
    _vocoder_path: str | None = field(default=None, init=False, repr=False)

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
                "checkpoint": self.params.get(
                    "checkpoint",
                    "SWivid/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors",
                ),
                "checkpoint_revision": self.params.get("checkpoint_revision"),
                "resolved_checkpoint_path": self._checkpoint_path or None,
                "vocoder": self.params.get("vocoder_repo", "charactr/vocos-mel-24khz"),
                "vocoder_revision": self.params.get("vocoder_revision"),
                "resolved_vocoder_path": self._vocoder_path or None,
                "inference_config": self._inference_config(),
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
                ckpt_file=self._resolve_checkpoint_file(),
                vocab_file=str(self.params.get("vocab_file", "")),
                ode_method=str(self.params.get("ode_method", "euler")),
                use_ema=bool(self.params.get("use_ema", True)),
                vocoder_local_path=self._resolve_vocoder_path(),
                device=self._device(),
                hf_cache_dir=self.params.get("hf_cache_dir"),
            )
        return self._model

    def _resolve_checkpoint_file(self) -> str:
        if self._checkpoint_path is not None:
            return self._checkpoint_path

        explicit_path = str(self.params.get("ckpt_file", ""))
        revision = self.params.get("checkpoint_revision")
        if explicit_path:
            self._checkpoint_path = explicit_path
        elif revision:
            try:
                from huggingface_hub import hf_hub_download
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "a pinned F5-TTS checkpoint requires the 'huggingface-hub' package"
                ) from exc
            self._checkpoint_path = hf_hub_download(
                repo_id=str(self.params.get("checkpoint_repo", "SWivid/F5-TTS")),
                filename=str(
                    self.params.get(
                        "checkpoint_file",
                        "F5TTS_v1_Base/model_1250000.safetensors",
                    )
                ),
                revision=str(revision),
                cache_dir=self.params.get("hf_cache_dir"),
            )
        else:
            # Preserve upstream behavior when no immutable revision was asked
            # for; F5TTS resolves its default alias itself.
            self._checkpoint_path = ""
        return self._checkpoint_path

    def _resolve_vocoder_path(self) -> str | None:
        if self._vocoder_path is not None:
            return self._vocoder_path or None
        explicit_path = self.params.get("vocoder_local_path")
        revision = self.params.get("vocoder_revision")
        if explicit_path:
            self._vocoder_path = str(explicit_path)
        elif revision:
            try:
                from huggingface_hub import snapshot_download
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "a pinned F5-TTS vocoder requires the 'huggingface-hub' package"
                ) from exc
            self._vocoder_path = snapshot_download(
                repo_id=str(self.params.get("vocoder_repo", "charactr/vocos-mel-24khz")),
                revision=str(revision),
                cache_dir=self.params.get("hf_cache_dir"),
            )
        else:
            self._vocoder_path = ""
        return self._vocoder_path or None

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

    def _inference_config(self) -> dict[str, Any]:
        return {
            "ref_text_mode": str(self.params.get("ref_text_mode", "transcript")),
            "nfe_step": int(self.params.get("nfe_step", 32)),
            "cfg_strength": float(self.params.get("cfg_strength", 2.0)),
            "sway_sampling_coef": float(self.params.get("sway_sampling_coef", -1.0)),
            "speed": float(self.params.get("speed", 1.0)),
            "remove_silence": bool(self.params.get("remove_silence", False)),
            "seed": self.params.get("seed"),
            "ode_method": str(self.params.get("ode_method", "euler")),
            "use_ema": bool(self.params.get("use_ema", True)),
        }


def _quiet(*args: Any, **kwargs: Any) -> None:
    return None
