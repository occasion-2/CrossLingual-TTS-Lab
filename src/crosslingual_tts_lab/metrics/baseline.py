from __future__ import annotations

import hashlib
from dataclasses import dataclass

from crosslingual_tts_lab.metrics.base import MetricResult
from crosslingual_tts_lab.runner_types import GeneratedSample


@dataclass(frozen=True)
class PlaceholderMetric:
    name: str
    purpose: str

    def evaluate(self, sample: GeneratedSample) -> MetricResult:
        key = f"{self.name}|{sample.job.id}|{sample.audio_path.name}"
        value = _stable_unit_interval(key)
        return MetricResult(
            name=self.name,
            status="missing_backend",
            value=round(value, 4),
            details={
                "purpose": self.purpose,
                "reason": "real evaluator is not configured yet",
                "direction": sample.job.direction,
                "cross_lingual": sample.job.is_cross_lingual,
            },
        )


def default_metrics() -> list[PlaceholderMetric]:
    return [
        PlaceholderMetric(
            name="target_asr_error_proxy",
            purpose="Replace with target-language ASR WER/CER.",
        ),
        PlaceholderMetric(
            name="speaker_similarity_proxy",
            purpose="Replace with speaker-verification embedding similarity.",
        ),
        PlaceholderMetric(
            name="target_language_id_confidence_proxy",
            purpose="Replace with generated-audio language identification confidence.",
        ),
        PlaceholderMetric(
            name="source_language_leakage_proxy",
            purpose="Replace with a probe predicting source prompt language after controlling for target language.",
        ),
        PlaceholderMetric(
            name="emotion_preservation_proxy",
            purpose="Replace with SER agreement when voice/target emotion labels are available.",
        ),
    ]


def _stable_unit_interval(key: str) -> float:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    integer = int.from_bytes(digest[:8], "big")
    return integer / ((1 << 64) - 1)
