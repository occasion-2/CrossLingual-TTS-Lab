from __future__ import annotations

from dataclasses import dataclass, field
import inspect
from pathlib import Path
import subprocess
from typing import Any

from crosslingual_tts_lab.backends.base import SynthesisResult
from crosslingual_tts_lab.device import detect_device_profile
from crosslingual_tts_lab.planner import GenerationJob


@dataclass
class SparkTTSBackend:
    """Spark-TTS zero-shot voice cloning through the official Python API."""

    params: dict[str, Any] = field(default_factory=dict)
    name: str = "spark_tts"
    _model: Any = field(default=None, init=False, repr=False)
    _checkpoint_revision_verified: bool | None = field(default=None, init=False, repr=False)
    _source_revision_verified: bool | None = field(default=None, init=False, repr=False)

    def synthesize(self, job: GenerationJob, output_dir: Path) -> SynthesisResult:
        if not job.voice.audio_path.exists():
            raise FileNotFoundError(f"reference voice audio does not exist: {job.voice.audio_path}")

        audio_path = output_dir / f"{job.id}.wav"
        audio_path.parent.mkdir(parents=True, exist_ok=True)

        model = self._load_model()
        prompt_text = self._reference_text(job)

        if job.target.language not in {"en", "zh", "en-US", "zh-CN"}:
            import warnings
            warnings.warn(f"Spark-TTS target language '{job.target.language}' is unsupported. Skipping placeholder audio creation.")
            placeholder = True
        else:
            applied_kwargs = self._inference_kwargs(model, job, prompt_text)
            wav_data = model.inference(**applied_kwargs)
            if hasattr(wav_data, "cpu"):
                wav_data = wav_data.cpu().numpy()

            if len(wav_data.shape) > 1:
                wav_data = wav_data.flatten()

            if wav_data.size == 0:
                raise ValueError("Generated empty waveform")
            placeholder = False
            
            from crosslingual_tts_lab.backends.qwen_tts import _write_wav
            _write_wav(audio_path, wav_data, 16000)
        if placeholder:
            applied_kwargs = {}

        return SynthesisResult(
            audio_path=audio_path,
            metadata={
                "backend": self.name,
                "model": self._model_name(),
                "reference_audio_path": str(job.voice.audio_path),
                "ref_text_mode": self.params.get("ref_text_mode", "transcript"),
                "target_language": job.target.language,
                "device": self._device(),
                "checkpoint_revision": self.params.get("revision"),
                "checkpoint_revision_verified": self._checkpoint_revision_verified,
                "source_revision": self.params.get("source_revision"),
                "source_revision_verified": self._source_revision_verified,
                "inference_config": {
                    "prompt_text_mode": str(self.params.get("ref_text_mode", "transcript")),
                    "temperature": applied_kwargs.get("temperature"),
                    "top_k": applied_kwargs.get("top_k"),
                    "top_p": applied_kwargs.get("top_p"),
                    "max_new_tokens": 3000,
                    "do_sample": True,
                    "seed": applied_kwargs.get("seed"),
                    "seed_status": (
                        "applied_as_inference_kwarg"
                        if "seed" in applied_kwargs
                        else "uncontrolled"
                    ),
                },
                "synthetic_placeholder": placeholder,
            },
        )

    def _load_model(self) -> Any:
        if self._model is None:
            try:
                import sys
                from pathlib import Path
                project_root = Path(__file__).resolve().parent.parent.parent.parent
                self._verify_source_revision(project_root / "Spark-TTS")
                spark_path = str((project_root / "Spark-TTS").resolve())
                if spark_path not in sys.path:
                    sys.path.insert(0, spark_path)
                from cli.SparkTTS import SparkTTS
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "Spark-TTS backend requires the 'cli.SparkTTS' package. "
                    "Make sure Spark-TTS is installed and added to Python path."
                ) from exc

            self._verify_local_checkpoint_revision()
            self._model = SparkTTS(
                model_dir=self._model_name(),
                device=self._device(),
            )
        return self._model

    def _verify_source_revision(self, checkout: Path) -> None:
        expected = self.params.get("source_revision")
        if not expected:
            self._source_revision_verified = None
            return
        try:
            actual = subprocess.check_output(
                ["git", "-C", str(checkout), "rev-parse", "HEAD"],
                text=True,
            ).strip()
        except (OSError, subprocess.CalledProcessError) as exc:
            raise RuntimeError(f"cannot verify Spark-TTS source revision in {checkout}") from exc
        if actual != str(expected):
            raise RuntimeError(
                f"Spark-TTS source revision mismatch: expected {expected}, found {actual}"
            )
        self._source_revision_verified = True

    def _verify_local_checkpoint_revision(self) -> None:
        revision = self.params.get("revision")
        if not revision:
            self._checkpoint_revision_verified = None
            return
        model_dir = Path(self._model_name())
        metadata_files = (
            model_dir / ".cache/huggingface/download/LLM/model.safetensors.metadata",
            model_dir / ".cache/huggingface/download/BiCodec/model.safetensors.metadata",
        )
        recorded_revisions = []
        for metadata_file in metadata_files:
            if not metadata_file.exists():
                raise RuntimeError(
                    "cannot verify pinned Spark-TTS checkpoint revision: missing "
                    f"{metadata_file}"
                )
            recorded_revisions.append(
                metadata_file.read_text(encoding="utf-8").splitlines()[0].strip()
            )
        if any(recorded != str(revision) for recorded in recorded_revisions):
            raise RuntimeError(
                "local Spark-TTS weights do not match requested revision "
                f"{revision}: found {recorded_revisions}"
            )
        self._checkpoint_revision_verified = True

    def _reference_text(self, job: GenerationJob) -> str | None:
        mode = str(self.params.get("ref_text_mode", "transcript"))
        if mode == "empty" or mode == "asr":
            return None
        if mode == "literal":
            return str(self.params.get("ref_text", "")) or None
        return job.voice.transcript or None

    def _inference_kwargs(self, model: Any, job: GenerationJob, prompt_text: str | None) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "text": job.target.text,
            "prompt_speech_path": str(job.voice.audio_path),
            "prompt_text": prompt_text,
            "temperature": float(self.params.get("temperature", 0.8)),
            "top_k": int(self.params.get("top_k", 50)),
            "top_p": float(self.params.get("top_p", 0.95)),
        }
        for key in ("gender", "pitch", "speed", "seed"):
            value = self.params.get(key)
            if value is not None:
                kwargs[key] = value

        try:
            signature = inspect.signature(model.inference)
        except (TypeError, ValueError):
            return kwargs
        if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
            return kwargs
        return {key: value for key, value in kwargs.items() if key in signature.parameters}

    def _model_name(self) -> str:
        return str(
            self.params.get("model")
            or self.params.get("model_name")
            or "pretrained_models/Spark-TTS-0.5B"
        )

    def _device(self) -> str:
        return str(self.params.get("device") or detect_device_profile().device)
