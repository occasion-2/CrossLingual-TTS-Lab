from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from crosslingual_tts_lab.config import BenchmarkConfig, ModelSpec, TargetSpec, VoiceSpec


_JOB_ID_UNSAFE = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class GenerationJob:
    id: str
    model: ModelSpec
    voice: VoiceSpec
    target: TargetSpec

    @property
    def is_cross_lingual(self) -> bool:
        return self.voice.language != self.target.language

    @property
    def direction(self) -> str:
        return f"{self.voice.language}->{self.target.language}"


def plan_jobs(config: BenchmarkConfig) -> list[GenerationJob]:
    voices = {voice.id: voice for voice in config.voices}
    targets = {target.id: target for target in config.targets}
    jobs: list[GenerationJob] = []

    for model in config.models:
        for pair in config.pairs:
            voice = voices[pair.voice]
            target = targets[pair.target]
            jobs.append(
                GenerationJob(
                    id="__".join(
                        [
                            _job_id_component(model.id),
                            _job_id_component(voice.id),
                            _job_id_component(target.id),
                        ]
                    ),
                    model=model,
                    voice=voice,
                    target=target,
                )
            )
    return jobs


def _job_id_component(value: str) -> str:
    cleaned = _JOB_ID_UNSAFE.sub("_", value).strip("._-")
    if not cleaned:
        cleaned = "item"
    if cleaned == value:
        return cleaned
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:8]
    return f"{cleaned}_{digest}"
