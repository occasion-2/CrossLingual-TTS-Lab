from __future__ import annotations

import hashlib
import random
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
    _artifact_hashes_verified: bool | None = field(default=None, init=False, repr=False)

    def synthesize(self, job: GenerationJob, output_dir: Path) -> SynthesisResult:
        if not job.voice.audio_path.exists():
            raise FileNotFoundError(f"reference voice audio does not exist: {job.voice.audio_path}")

        audio_path = output_dir / f"{job.id}.wav"
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        tts = self._load_tts()
        language = _map_language(job.target.language, self.params.get("language_map", {}))
        seed_status = self._apply_seed()
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
                "checkpoint_revision": self.params.get("revision"),
                "artifact_hashes_verified": self._artifact_hashes_verified,
                "inference_config": {
                    "api": "TTS.tts_to_file",
                    "decoder_config": "downloaded checkpoint defaults",
                    "gpu": bool(
                        self.params.get("gpu", detect_device_profile().device == "cuda")
                    ),
                    "seed": self.params.get("seed"),
                    "seed_status": seed_status,
                },
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
            self._verify_model_artifacts(self._tts)

        return self._tts

    def _verify_model_artifacts(self, tts: Any) -> None:
        expected = {
            "model.pth": self.params.get("model_sha256"),
            "config.json": self.params.get("config_sha256"),
            "vocab.json": self.params.get("vocab_sha256"),
            "speakers_xtts.pth": self.params.get("speakers_sha256"),
        }
        expected = {name: str(digest) for name, digest in expected.items() if digest}
        if not expected:
            self._artifact_hashes_verified = None
            return

        explicit_dir = self.params.get("artifact_dir")
        if explicit_dir:
            artifact_dir = Path(str(explicit_dir))
        else:
            manager = getattr(tts, "manager", None)
            output_prefix = getattr(manager, "output_prefix", None)
            if not output_prefix:
                raise RuntimeError("cannot locate downloaded XTTS artifacts for hash verification")
            artifact_dir = Path(str(output_prefix)) / self._model_name().replace("/", "--")

        for filename, expected_digest in expected.items():
            artifact = artifact_dir / filename
            if not artifact.exists():
                raise RuntimeError(f"missing XTTS artifact required for verification: {artifact}")
            digest = hashlib.sha256()
            with artifact.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            actual_digest = digest.hexdigest()
            if actual_digest != expected_digest:
                raise RuntimeError(
                    f"XTTS artifact hash mismatch for {filename}: "
                    f"expected {expected_digest}, found {actual_digest}"
                )
        self._artifact_hashes_verified = True

    def _apply_seed(self) -> str:
        seed = self.params.get("seed")
        if seed is None:
            return "uncontrolled"
        seed_value = int(seed)
        random.seed(seed_value)
        try:
            import numpy as np

            np.random.seed(seed_value % (2**32))
            import torch

            torch.manual_seed(seed_value)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed_value)
        except ModuleNotFoundError:
            pass
        return "global_rng_seeded"

    def _model_name(self) -> str:
        return str(
            self.params.get("model_name")
            or "tts_models/multilingual/multi-dataset/xtts_v2"
        )


def _map_language(language: str, language_map: dict[str, Any]) -> str:
    return str(language_map.get(language, language))
