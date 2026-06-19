from __future__ import annotations

from dataclasses import dataclass

from crosslingual_tts_lab.config import BenchmarkConfig, ModelSpec, TargetSpec, VoiceSpec


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
                    id=f"{model.id}__{voice.id}__{target.id}",
                    model=model,
                    voice=voice,
                    target=target,
                )
            )
    return jobs
