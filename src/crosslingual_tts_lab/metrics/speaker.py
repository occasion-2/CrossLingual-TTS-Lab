from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from crosslingual_tts_lab.device import DeviceProfile
from crosslingual_tts_lab.metrics.base import MetricResult
from crosslingual_tts_lab.runner_types import GeneratedSample


@dataclass
class SpeechBrainSpeakerSimilarityMetric:
    name: str
    params: dict[str, Any]
    device_profile: DeviceProfile
    _classifier: Any = field(default=None, init=False, repr=False)

    def evaluate(self, sample: GeneratedSample) -> MetricResult:
        reference = sample.job.voice.audio_path
        if not reference.exists():
            return MetricResult(
                name=self.name,
                status="error",
                value=None,
                details={
                    "error_type": "MissingReferenceAudio",
                    "reference_audio_path": str(reference),
                },
            )

        try:
            score = self._cosine_similarity(reference, sample.audio_path)
            return MetricResult(
                name=self.name,
                status="ok",
                value=round(score, 6),
                details={
                    "reference_audio_path": str(reference),
                    "model_id": self._model_id(),
                    "model_revision": self._model_revision(),
                    "device": self._device(),
                },
            )
        except ModuleNotFoundError as exc:
            return MetricResult(
                name=self.name,
                status="missing_backend",
                value=None,
                details={
                    "package": "speechbrain",
                    "reason": f"{type(exc).__name__}: {exc}",
                },
            )
        except Exception as exc:
            return MetricResult(
                name=self.name,
                status="error",
                value=None,
                details={
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )

    def _cosine_similarity(self, reference: Path, generated: Path) -> float:
        import torch

        classifier = self._load_classifier()
        with torch.no_grad():
            ref_embedding = self._encode_file(classifier, reference)
            gen_embedding = self._encode_file(classifier, generated)
            score = torch.nn.functional.cosine_similarity(
                ref_embedding.flatten(),
                gen_embedding.flatten(),
                dim=0,
            )
        return float(score.detach().cpu().item())

    def _encode_file(self, classifier: Any, path: Path) -> Any:
        if hasattr(classifier, "encode_file"):
            return classifier.encode_file(str(path))
        signal = classifier.load_audio(str(path))
        return classifier.encode_batch(signal)

    def _load_classifier(self) -> Any:
        if self._classifier is None:
            from speechbrain.inference.speaker import EncoderClassifier

            run_opts = {"device": self._device()}
            load_kwargs: dict[str, Any] = {
                "source": self._model_id(),
                "savedir": self.params.get("savedir"),
                "run_opts": run_opts,
            }
            revision = self._model_revision()
            if revision is not None:
                from speechbrain.utils.fetching import FetchConfig

                load_kwargs["fetch_config"] = FetchConfig(
                    revision=revision,
                    # SpeechBrain otherwise reuses files already linked into
                    # savedir even when the requested revision has changed.
                    allow_updates=True,
                )
            self._classifier = EncoderClassifier.from_hparams(**load_kwargs)
        return self._classifier

    def _model_id(self) -> str:
        return str(self.params.get("model_id") or "speechbrain/spkrec-ecapa-voxceleb")

    def _model_revision(self) -> str | None:
        value = self.params.get("model_revision", self.params.get("revision"))
        if value is None:
            return None
        revision = str(value).strip()
        return revision or None

    def _device(self) -> str:
        device = str(self.params.get("device") or self.device_profile.device)
        if device == "cuda":
            return "cuda:0"
        return device
