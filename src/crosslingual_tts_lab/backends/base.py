from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from crosslingual_tts_lab.planner import GenerationJob


@dataclass(frozen=True)
class SynthesisResult:
    audio_path: Path
    metadata: dict[str, str | int | float | bool | None] = field(default_factory=dict)


class TTSBackend(Protocol):
    name: str

    def synthesize(self, job: GenerationJob, output_dir: Path) -> SynthesisResult:
        """Generate audio for one benchmark job."""
