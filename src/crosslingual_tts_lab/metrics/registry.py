from __future__ import annotations

from crosslingual_tts_lab.config import MetricSpec
from crosslingual_tts_lab.device import DeviceProfile
from crosslingual_tts_lab.metrics.asr import FasterWhisperASRMetric, FasterWhisperLIDMetric
from crosslingual_tts_lab.metrics.baseline import default_metrics
from crosslingual_tts_lab.metrics.base import SampleMetric
from crosslingual_tts_lab.metrics.speaker import SpeechBrainSpeakerSimilarityMetric


def create_metrics(specs: list[MetricSpec], device_profile: DeviceProfile) -> list[SampleMetric]:
    if not specs:
        return default_metrics()

    metrics: list[SampleMetric] = []
    for spec in specs:
        backend = spec.backend.strip().lower()
        if backend == "placeholder":
            metrics.extend(default_metrics())
        elif backend == "faster_whisper_asr":
            metrics.append(FasterWhisperASRMetric(spec.id, spec.params, device_profile))
        elif backend == "faster_whisper_lid":
            metrics.append(FasterWhisperLIDMetric(spec.id, spec.params, device_profile))
        elif backend == "speechbrain_speaker_similarity":
            metrics.append(SpeechBrainSpeakerSimilarityMetric(spec.id, spec.params, device_profile))
        else:
            raise ValueError(
                f"unknown metric backend {spec.backend!r}; available backends: "
                "placeholder, faster_whisper_asr, faster_whisper_lid, "
                "speechbrain_speaker_similarity"
            )
    return metrics
