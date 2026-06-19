from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelSpec:
    id: str
    backend: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VoiceSpec:
    id: str
    language: str
    speaker_id: str
    audio_path: Path
    transcript: str | None = None
    emotion: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TargetSpec:
    id: str
    language: str
    text: str
    emotion: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PairSpec:
    voice: str
    target: str


@dataclass(frozen=True)
class MetricSpec:
    id: str
    backend: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkConfig:
    name: str
    description: str | None
    models: list[ModelSpec]
    voices: list[VoiceSpec]
    targets: list[TargetSpec]
    pairs: list[PairSpec]
    metrics: list[MetricSpec]
    root: Path


def load_config(path: Path) -> BenchmarkConfig:
    raw = _load_mapping(path)
    root = path.parent

    models = [
        ModelSpec(
            id=_required(item, "id", "models"),
            backend=_required(item, "backend", "models"),
            params=dict(item.get("params", {})),
        )
        for item in _required_list(raw, "models")
    ]
    voices = [
        VoiceSpec(
            id=_required(item, "id", "voices"),
            language=_required(item, "language", "voices"),
            speaker_id=_required(item, "speaker_id", "voices"),
            audio_path=(root / _required(item, "audio_path", "voices")).resolve(),
            transcript=item.get("transcript"),
            emotion=item.get("emotion"),
            metadata=dict(item.get("metadata", {})),
        )
        for item in _required_list(raw, "voices")
    ]
    targets = [
        TargetSpec(
            id=_required(item, "id", "targets"),
            language=_required(item, "language", "targets"),
            text=_required(item, "text", "targets"),
            emotion=item.get("emotion"),
            metadata=dict(item.get("metadata", {})),
        )
        for item in _required_list(raw, "targets")
    ]
    pairs = [
        PairSpec(
            voice=_required(item, "voice", "pairs"),
            target=_required(item, "target", "pairs"),
        )
        for item in _required_list(raw, "pairs")
    ]
    metrics = [
        MetricSpec(
            id=_required(item, "id", "metrics"),
            backend=_required(item, "backend", "metrics"),
            params=dict(item.get("params", {})),
        )
        for item in raw.get("metrics", [])
    ]

    config = BenchmarkConfig(
        name=str(raw.get("name", path.stem)),
        description=raw.get("description"),
        models=models,
        voices=voices,
        targets=targets,
        pairs=pairs,
        metrics=metrics,
        root=root.resolve(),
    )
    validate_config(config)
    return config


def validate_config(config: BenchmarkConfig) -> None:
    _ensure_unique("models", [model.id for model in config.models])
    _ensure_unique("voices", [voice.id for voice in config.voices])
    _ensure_unique("targets", [target.id for target in config.targets])
    _ensure_unique("metrics", [metric.id for metric in config.metrics])

    voice_ids = {voice.id for voice in config.voices}
    target_ids = {target.id for target in config.targets}
    for pair in config.pairs:
        if pair.voice not in voice_ids:
            raise ValueError(f"pair references unknown voice {pair.voice!r}")
        if pair.target not in target_ids:
            raise ValueError(f"pair references unknown target {pair.target!r}")


def _load_mapping(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    with path.open("rb") as handle:
        if suffix == ".toml":
            data = tomllib.load(handle)
        elif suffix == ".json":
            data = json.load(handle)
        else:
            raise ValueError(f"unsupported config extension {suffix!r}; use .toml or .json")
    if not isinstance(data, dict):
        raise ValueError("config root must be a mapping")
    return data


def _required(item: dict[str, Any], key: str, section: str) -> str:
    value = item.get(key)
    if value is None or value == "":
        raise ValueError(f"{section} entry is missing required key {key!r}")
    return str(value)


def _required_list(raw: dict[str, Any], key: str) -> list[dict[str, Any]]:
    value = raw.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"config must contain a non-empty [{key}] array")
    for item in value:
        if not isinstance(item, dict):
            raise ValueError(f"{key} entries must be mappings")
    return value


def _ensure_unique(label: str, values: list[str]) -> None:
    seen: set[str] = set()
    for value in values:
        if value in seen:
            raise ValueError(f"duplicate {label} id {value!r}")
        seen.add(value)
