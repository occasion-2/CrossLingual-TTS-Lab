from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from crosslingual_tts_lab.device import DeviceProfile
from crosslingual_tts_lab.metrics.base import MetricResult
from crosslingual_tts_lab.runner_types import GeneratedSample


@dataclass
class SpeechBrainLanguageSimilarityMetric:
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
            try:
                from speechbrain.inference.classifiers import EncoderClassifier
            except ImportError:
                # Fallback to older speechbrain path if needed
                from speechbrain.inference.speaker import EncoderClassifier

            run_opts = {"device": self._device()}
            self._classifier = EncoderClassifier.from_hparams(
                source=self._model_id(),
                savedir=self.params.get("savedir"),
                run_opts=run_opts,
            )
        return self._classifier

    def _model_id(self) -> str:
        return str(self.params.get("model_id") or "speechbrain/lang-id-voxlingua107-ecapa")

    def _device(self) -> str:
        device = str(self.params.get("device") or self.device_profile.device)
        if device == "cuda":
            return "cuda:0"
        return device
