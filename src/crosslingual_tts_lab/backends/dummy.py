from __future__ import annotations

from pathlib import Path

from crosslingual_tts_lab.audio import write_tone_wav
from crosslingual_tts_lab.backends.base import SynthesisResult
from crosslingual_tts_lab.planner import GenerationJob


class DummyBackend:
    """Deterministic local backend for pipeline smoke tests."""

    name = "dummy"

    def synthesize(self, job: GenerationJob, output_dir: Path) -> SynthesisResult:
        audio_path = output_dir / f"{job.id}.wav"
        key = "|".join(
            [
                job.model.id,
                job.voice.speaker_id,
                job.voice.language,
                job.target.language,
                job.target.text,
            ]
        )
        duration = max(0.8, min(4.0, 0.55 + len(job.target.text) / 55))
        write_tone_wav(audio_path, key=key, duration_s=duration)
        return SynthesisResult(
            audio_path=audio_path,
            metadata={
                "backend": self.name,
                "synthetic_placeholder": True,
                "duration_s": round(duration, 3),
            },
        )
