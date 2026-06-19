from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from crosslingual_tts_lab.runner_types import GeneratedSample


@dataclass(frozen=True)
class MetricResult:
    name: str
    status: str
    value: float | str | None = None
    details: dict[str, str | int | float | bool | None] = field(default_factory=dict)


class SampleMetric(Protocol):
    name: str

    def evaluate(self, sample: GeneratedSample) -> MetricResult:
        """Evaluate one generated sample."""
