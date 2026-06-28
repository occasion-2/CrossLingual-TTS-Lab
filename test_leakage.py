import os
import asyncio
from pathlib import Path
from crosslingual_tts_lab.metrics.leakage import SpeechBrainLanguageSimilarityMetric
from crosslingual_tts_lab.runner_types import GeneratedSample
from crosslingual_tts_lab.planner import GenerationJob, VoiceSpec, TargetSpec
from crosslingual_tts_lab.config import ModelSpec
from crosslingual_tts_lab.device import DeviceProfile

profile = DeviceProfile(device="cuda", cuda_available=True)

metric = SpeechBrainLanguageSimilarityMetric(
    name="source_language_similarity",
    params={},
    device_profile=profile
)

job = GenerationJob(
    id="test_job",
    model=ModelSpec(id="test", backend="test", params={}),
    voice=VoiceSpec(id="ru1", language="ru", speaker_id="s1", audio_path=Path("overnight_runs/results_cosyvoice/audio/cosyvoice__fleurs_ru_voice_006__fleurs_en_target_002.wav"), transcript="test"),
    target=TargetSpec(id="en1", language="en", text="test")
)

sample = GeneratedSample(
    job=job,
    audio_path=Path("overnight_runs/results_cosyvoice/audio/cosyvoice__fleurs_ru_voice_006__fleurs_en_target_002.wav"),
)

print(metric.evaluate(sample))
