from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from crosslingual_tts_lab.planner import GenerationJob


@dataclass(frozen=True)
class GeneratedSample:
    job: GenerationJob
    audio_path: Path
    synthesis_metadata: dict[str, str | int | float | bool | None] = field(default_factory=dict)
