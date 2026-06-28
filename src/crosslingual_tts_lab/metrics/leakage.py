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
            sim_src, sim_tgt, delta = self._calculate_delta(sample)
            return MetricResult(
                name="normalized_leakage_delta",
                status="ok",
                value=round(delta, 6),
                details={
                    "sim_source_centroid": round(sim_src, 6),
                    "sim_target_centroid": round(sim_tgt, 6),
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

    def _calculate_delta(self, sample: GeneratedSample) -> tuple[float, float, float]:
        import json
        import torch

        src_lang = sample.job.voice.language
        tgt_lang = sample.job.target.language

        centroids_file = Path(__file__).parent / "fleurs_centroids.json"
        if not centroids_file.exists():
            raise FileNotFoundError(f"Missing language centroids cache: {centroids_file}")
            
        if getattr(self, "_centroids", None) is None:
            with open(centroids_file, "r") as f:
                self._centroids = json.load(f)

        if src_lang not in self._centroids or tgt_lang not in self._centroids:
            raise ValueError(f"Language centroids not available for pairs ({src_lang}->{tgt_lang})")

        src_centroid = torch.tensor(self._centroids[src_lang]).to(self._device())
        tgt_centroid = torch.tensor(self._centroids[tgt_lang]).to(self._device())

        classifier = self._load_classifier()
        with torch.no_grad():
            gen_embedding = self._encode_file(classifier, sample.audio_path).flatten()
            sim_src = torch.nn.functional.cosine_similarity(gen_embedding, src_centroid, dim=0)
            sim_tgt = torch.nn.functional.cosine_similarity(gen_embedding, tgt_centroid, dim=0)
            
        sim_src_val = float(sim_src.detach().cpu().item())
        sim_tgt_val = float(sim_tgt.detach().cpu().item())
        return sim_src_val, sim_tgt_val, sim_src_val - sim_tgt_val

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
