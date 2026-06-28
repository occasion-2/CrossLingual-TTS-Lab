from __future__ import annotations

from collections.abc import Callable

from crosslingual_tts_lab.config import MetricSpec
from crosslingual_tts_lab.device import DeviceProfile
from crosslingual_tts_lab.metrics.asr import FasterWhisperASRMetric, FasterWhisperLIDMetric
from crosslingual_tts_lab.metrics.baseline import default_metrics
from crosslingual_tts_lab.metrics.base import SampleMetric
from crosslingual_tts_lab.metrics.speaker import SpeechBrainSpeakerSimilarityMetric
from crosslingual_tts_lab.metrics.leakage import SpeechBrainLanguageSimilarityMetric


MetricFactory = Callable[[MetricSpec, DeviceProfile], list[SampleMetric]]


_METRIC_FACTORIES: dict[str, MetricFactory] = {
    "placeholder": lambda spec, device_profile: list(default_metrics()),
    "faster_whisper_asr": lambda spec, device_profile: [
        FasterWhisperASRMetric(spec.id, spec.params, device_profile)
    ],
    "faster_whisper_lid": lambda spec, device_profile: [
        FasterWhisperLIDMetric(spec.id, spec.params, device_profile)
    ],
    "speechbrain_speaker_similarity": lambda spec, device_profile: [
        SpeechBrainSpeakerSimilarityMetric(spec.id, spec.params, device_profile)
    ],
    "speechbrain_language_similarity": lambda spec, device_profile: [
        SpeechBrainLanguageSimilarityMetric(spec.id, spec.params, device_profile)
    ],
}


def create_metrics(specs: list[MetricSpec], device_profile: DeviceProfile) -> list[SampleMetric]:
    if not specs:
        return default_metrics()

    metrics: list[SampleMetric] = []
    for spec in specs:
        backend = spec.backend.strip().lower()
        try:
            factory = _METRIC_FACTORIES[backend]
        except KeyError as exc:
            raise ValueError(
                f"unknown metric backend {spec.backend!r}; available backends: "
                "placeholder, faster_whisper_asr, faster_whisper_lid, "
                "speechbrain_speaker_similarity, speechbrain_language_similarity"
            ) from exc
        metrics.extend(factory(spec, device_profile))
    return metrics
