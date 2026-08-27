from __future__ import annotations

import hashlib
import subprocess
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
    _model_source: str | None = field(default=None, init=False, repr=False)
    _resolved_model_class: str | None = field(default=None, init=False, repr=False)
    _artifact_tree_verified: bool | None = field(default=None, init=False, repr=False)
    _source_revision_verified: bool | None = field(default=None, init=False, repr=False)

    def synthesize(self, job: GenerationJob, output_dir: Path) -> SynthesisResult:
        if not job.voice.audio_path.exists():
            raise FileNotFoundError(f"reference voice audio does not exist: {job.voice.audio_path}")

        audio_path = output_dir / f"{job.id}.wav"
        audio_path.parent.mkdir(parents=True, exist_ok=True)

        model = self._load_model()
        import numpy as np

        prompt_speech = str(job.voice.audio_path)
        prompt_text = self._reference_text(job)
        if self._is_cosyvoice3() and "<|endofprompt|>" not in prompt_text:
            prompt_text = "You are a helpful assistant.<|endofprompt|>" + prompt_text

        chunks = []
        for result in model.inference_zero_shot(
            job.target.text,
            prompt_text,
            prompt_speech,
            stream=bool(self.params.get("stream", False)),
            speed=float(self.params.get("speed", 1.0)),
            text_frontend=bool(self.params.get("text_frontend", True)),
        ):
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
                "model_revision": self.params.get("revision"),
                "source_revision": self.params.get("source_revision"),
                "source_revision_verified": self._source_revision_verified,
                "resolved_model_source": self._model_source or self._model_name(),
                "resolved_model_class": self._resolved_model_class or type(model).__name__,
                "artifact_tree_verified": self._artifact_tree_verified,
                "inference_config": {
                    "prompt_mode": "zero_shot_with_transcript",
                    "system_prompt_prefix": "You are a helpful assistant.<|endofprompt|>",
                    "stream": bool(self.params.get("stream", False)),
                    "speed": float(self.params.get("speed", 1.0)),
                    "text_frontend": bool(self.params.get("text_frontend", True)),
                    "fp16": bool(self.params.get("fp16", False)),
                    "load_jit": bool(self.params.get("load_jit", False)),
                    "load_trt": bool(self.params.get("load_trt", False)),
                    "load_vllm": bool(self.params.get("load_vllm", False)),
                },
                "synthetic_placeholder": False,
            },
        )

    def _load_model(self) -> Any:
        if self._model is None:
            try:
                import sys
                from pathlib import Path
                project_root = Path(__file__).resolve().parent.parent.parent.parent
                self._verify_source_revision(project_root / "CosyVoice")
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

            model_name = self._model_name()
            model_source = self._resolve_model_source()
            self._model = AutoModel(**self._model_load_kwargs(model_name, model_source))
            self._resolved_model_class = type(self._model).__name__
            expected_class = self._expected_model_class(model_name, model_source)
            if expected_class and self._resolved_model_class != expected_class:
                raise RuntimeError(
                    "CosyVoice runtime class mismatch: "
                    f"checkpoint requires {expected_class}, but AutoModel returned "
                    f"{self._resolved_model_class}"
                )
        return self._model

    def _model_load_kwargs(self, model_name: str, model_source: str) -> dict[str, Any]:
        load_kwargs: dict[str, Any] = {
            "model_dir": model_source,
            "load_trt": bool(self.params.get("load_trt", False)),
            "fp16": bool(self.params.get("fp16", False)),
        }
        # CosyVoice's constructors are version-specific: CosyVoice3 omits
        # load_jit, while only CosyVoice2/3 accept load_vllm.
        model_class = self._expected_model_class(model_name, model_source)
        if model_class == "CosyVoice3":
            load_kwargs["load_vllm"] = bool(self.params.get("load_vllm", False))
        elif model_class == "CosyVoice2":
            load_kwargs["load_jit"] = bool(self.params.get("load_jit", False))
            load_kwargs["load_vllm"] = bool(self.params.get("load_vllm", False))
        else:
            load_kwargs["load_jit"] = bool(self.params.get("load_jit", False))
        return load_kwargs

    @staticmethod
    def _expected_model_class(model_name: str, model_source: str) -> str | None:
        """Resolve the runtime family from the checkpoint, then its public ID.

        Checking the checkpoint files makes local CosyVoice3 directories safe
        even when their directory name does not contain ``CosyVoice3``.
        """

        source = Path(model_source)
        if source.is_dir():
            if (source / "cosyvoice3.yaml").is_file():
                return "CosyVoice3"
            if (source / "cosyvoice2.yaml").is_file():
                return "CosyVoice2"
            if (source / "cosyvoice.yaml").is_file():
                return "CosyVoice"
        if "CosyVoice3" in model_name:
            return "CosyVoice3"
        if "CosyVoice2" in model_name:
            return "CosyVoice2"
        if "CosyVoice" in model_name:
            return "CosyVoice"
        return None

    def _is_cosyvoice3(self) -> bool:
        if self._resolved_model_class is not None:
            return self._resolved_model_class == "CosyVoice3"
        return self._expected_model_class(
            self._model_name(),
            self._model_source or self._model_name(),
        ) == "CosyVoice3"

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
            raise RuntimeError(f"cannot verify CosyVoice source revision in {checkout}") from exc
        if actual != str(expected):
            raise RuntimeError(
                f"CosyVoice source revision mismatch: expected {expected}, found {actual}"
            )
        self._source_revision_verified = True

    def _resolve_model_source(self) -> str:
        if self._model_source is not None:
            return self._model_source
        model_name = self._model_name()
        expected_tree = self.params.get("artifact_tree_sha256")
        if Path(model_name).exists():
            model_source = Path(model_name)
        elif expected_tree:
            try:
                from modelscope import snapshot_download
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "a hash-pinned CosyVoice checkpoint requires the 'modelscope' package"
                ) from exc
            model_source = Path(
                snapshot_download(
                    model_name,
                    revision=str(self.params.get("revision", "master")),
                )
            )
        else:
            self._artifact_tree_verified = None
            self._model_source = model_name
            return self._model_source

        if expected_tree:
            actual_tree = _artifact_tree_sha256(model_source)
            if actual_tree != str(expected_tree):
                raise RuntimeError(
                    "CosyVoice artifact tree hash mismatch: "
                    f"expected {expected_tree}, found {actual_tree}"
                )
            self._artifact_tree_verified = True
        else:
            self._artifact_tree_verified = None
        self._model_source = str(model_source)
        return self._model_source

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


def _artifact_tree_sha256(root: Path) -> str:
    tree_digest = hashlib.sha256()
    artifacts = sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file() and path.name not in {".mdl", ".msc", ".mv"}
        ),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    for artifact in artifacts:
        artifact_digest = hashlib.sha256()
        with artifact.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                artifact_digest.update(chunk)
        relative_path = artifact.relative_to(root).as_posix()
        tree_digest.update(f"{artifact_digest.hexdigest()}  ./{relative_path}\n".encode())
    return tree_digest.hexdigest()
