from __future__ import annotations

import csv
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LanguageRequest:
    dataset_code: str
    benchmark_code: str


def parse_language_requests(value: str) -> list[LanguageRequest]:
    requests: list[LanguageRequest] = []
    for raw_item in value.split(","):
        item = raw_item.strip()
        if not item:
            continue
        if ":" in item:
            dataset_code, benchmark_code = item.split(":", 1)
        else:
            dataset_code = benchmark_code = item
        requests.append(
            LanguageRequest(
                dataset_code=dataset_code.strip(),
                benchmark_code=benchmark_code.strip(),
            )
        )
    if len(requests) < 2:
        raise ValueError("at least two languages are required for cross-lingual pairs")
    return requests


def build_common_voice_config(
    *,
    out_path: Path,
    languages: list[LanguageRequest],
    dataset_name: str,
    split: str,
    voices_per_language: int,
    targets_per_language: int,
    model_id: str,
    model_backend: str,
    model_params: dict[str, str],
    include_mono_lingual: bool,
    utterances_per_speaker: int = 1,
    max_voice_chars: int | None = None,
    max_target_chars: int | None = None,
    min_target_chars: int | None = None,
) -> Path:
    try:
        from datasets import Audio, load_dataset
        from datasets.exceptions import DatasetNotFoundError
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Common Voice config generation requires the optional 'datasets' package. "
            "Install the open-data extra, then rerun this command."
        ) from exc

    voices: list[dict[str, Any]] = []
    targets: list[dict[str, Any]] = []

    for language in languages:
        try:
            dataset = load_dataset(dataset_name, language.dataset_code, split=split)
        except DatasetNotFoundError as exc:
            raise RuntimeError(
                f"Could not access {dataset_name!r} on Hugging Face. Mozilla Common Voice "
                "datasets on Hugging Face are now placeholder/empty repos after the move "
                "to Mozilla Data Collective. Use `xttslab dataset fleurs` for an open "
                "Hugging Face dataset, or download Common Voice manually and run "
                "`xttslab dataset common-voice --local-root /path/to/cv-corpus ...`."
            ) from exc
        if "audio" in dataset.column_names:
            dataset = dataset.cast_column("audio", Audio(decode=True))

        if utterances_per_speaker > 1:
            selected_voices = _select_speaker_voice_rows(
                dataset,
                speakers_limit=voices_per_language,
                utterances_per_speaker=utterances_per_speaker,
                language=language.benchmark_code,
                max_chars=max_voice_chars,
            )
        else:
            selected_voices = _select_voice_rows(
                dataset,
                voices_per_language,
                language.benchmark_code,
                max_voice_chars,
            )
        selected_targets = _select_target_rows(
            dataset,
            targets_per_language,
            language.benchmark_code,
            max_chars=max_target_chars,
            min_chars=min_target_chars,
        )

        for index, row in enumerate(selected_voices, start=1):
            audio_path = _audio_path(row)
            if audio_path is None:
                continue
            speaker_id = _speaker_id(row, f"{language.dataset_code}-speaker-{index:03d}")
            voices.append(
                {
                    "id": f"cv_{language.benchmark_code}_voice_{index:03d}",
                    "language": language.benchmark_code,
                    "speaker_id": speaker_id,
                    "audio_path": str(audio_path),
                    "transcript": _row_text(row),
                    "metadata": {
                        "dataset": dataset_name,
                        "dataset_language": language.dataset_code,
                        "split": split,
                        "source": "common_voice",
                    },
                }
            )

        for index, row in enumerate(selected_targets, start=1):
            text = _row_text(row)
            if not text:
                continue
            targets.append(
                {
                    "id": f"cv_{language.benchmark_code}_target_{index:03d}",
                    "language": language.benchmark_code,
                    "text": text,
                    "metadata": {
                        "dataset": dataset_name,
                        "dataset_language": language.dataset_code,
                        "split": split,
                        "source": "common_voice",
                    },
                }
            )

    pairs = _build_pairs(voices, targets, include_mono_lingual)
    config_text = render_benchmark_toml(
        name="common-voice-crosslingual",
        description=(
            "Open-data Common Voice slice for cross-lingual voice-language "
            "disentanglement experiments."
        ),
        models=[{"id": model_id, "backend": model_backend, "params": model_params}],
        metrics=_real_metric_specs(),
        voices=voices,
        targets=targets,
        pairs=pairs,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(config_text, encoding="utf-8")
    return out_path


def build_local_common_voice_config(
    *,
    out_path: Path,
    local_root: Path,
    languages: list[LanguageRequest],
    split: str,
    voices_per_language: int,
    targets_per_language: int,
    model_id: str,
    model_backend: str,
    model_params: dict[str, str],
    include_mono_lingual: bool,
    utterances_per_speaker: int = 1,
    max_voice_chars: int | None = None,
    max_target_chars: int | None = None,
    min_target_chars: int | None = None,
) -> Path:
    voices: list[dict[str, Any]] = []
    targets: list[dict[str, Any]] = []
    local_root = local_root.resolve()

    for language in languages:
        rows = _load_local_common_voice_rows(local_root, language.dataset_code, split)
        if utterances_per_speaker > 1:
            selected_voices = _select_speaker_voice_rows(
                rows,
                speakers_limit=voices_per_language,
                utterances_per_speaker=utterances_per_speaker,
                language=language.benchmark_code,
                max_chars=max_voice_chars,
            )
        else:
            selected_voices = _select_voice_rows(
                rows,
                voices_per_language,
                language.benchmark_code,
                max_voice_chars,
            )
        selected_targets = _select_target_rows(
            rows,
            targets_per_language,
            language.benchmark_code,
            max_chars=max_target_chars,
            min_chars=min_target_chars,
        )

        if not selected_voices:
            raise RuntimeError(
                f"no usable Common Voice reference utterances found for {language.dataset_code!r} "
                f"under {local_root}; check split={split!r}, clips/, and utterances-per-speaker"
            )
        if not selected_targets:
            raise RuntimeError(
                f"no usable Common Voice target text found for {language.dataset_code!r} "
                f"under {local_root}; check split={split!r}"
            )

        for index, row in enumerate(selected_voices, start=1):
            audio_path = _audio_path(row)
            if audio_path is None:
                continue
            speaker_id = _speaker_id(row, f"{language.dataset_code}-speaker-{index:03d}")
            voices.append(
                {
                    "id": f"cv_{language.benchmark_code}_voice_{index:03d}",
                    "language": language.benchmark_code,
                    "speaker_id": speaker_id,
                    "audio_path": str(audio_path),
                    "transcript": _normalize_text(language.benchmark_code, _row_text(row)),
                    "metadata": {
                        "dataset": "local-common-voice",
                        "dataset_language": language.dataset_code,
                        "split": split,
                        "source": "common_voice_local",
                    },
                }
            )

        for index, row in enumerate(selected_targets, start=1):
            text = _normalize_text(language.benchmark_code, _row_text(row))
            if not text:
                continue
            targets.append(
                {
                    "id": f"cv_{language.benchmark_code}_target_{index:03d}",
                    "language": language.benchmark_code,
                    "text": text,
                    "metadata": {
                        "dataset": "local-common-voice",
                        "dataset_language": language.dataset_code,
                        "split": split,
                        "source": "common_voice_local",
                    },
                }
            )

    pairs = _build_pairs(voices, targets, include_mono_lingual)
    config_text = render_benchmark_toml(
        name="local-common-voice-crosslingual",
        description=(
            "Local Common Voice slice with known speaker IDs for cross-lingual "
            "voice-language disentanglement and speaker calibration."
        ),
        models=[{"id": model_id, "backend": model_backend, "params": model_params}],
        metrics=_real_metric_specs(),
        voices=voices,
        targets=targets,
        pairs=pairs,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(config_text, encoding="utf-8")
    return out_path


def build_fleurs_config(
    *,
    out_path: Path,
    languages: list[LanguageRequest],
    dataset_name: str,
    split: str,
    voices_per_language: int,
    targets_per_language: int,
    model_id: str,
    model_backend: str,
    model_params: dict[str, str],
    include_mono_lingual: bool,
    voice_languages: set[str] | None = None,
    target_languages: set[str] | None = None,
    max_voice_chars: int | None = None,
    max_target_chars: int | None = None,
    min_target_chars: int | None = None,
) -> Path:
    try:
        from datasets import Audio, load_dataset
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "FLEURS config generation requires the optional 'datasets' package. "
            "Install the open-data extra, then rerun this command."
        ) from exc

    voices: list[dict[str, Any]] = []
    targets: list[dict[str, Any]] = []

    for language in languages:
        include_voices = voice_languages is None or language.benchmark_code in voice_languages
        include_targets = target_languages is None or language.benchmark_code in target_languages
        dataset_code = _fleurs_dataset_code(language.dataset_code)
        dataset = load_dataset(dataset_name, dataset_code, split=split)
        if "audio" in dataset.column_names:
            dataset = dataset.cast_column("audio", Audio(decode=True))

        selected_voices = (
            _select_voice_rows(dataset, voices_per_language, language.benchmark_code, max_voice_chars)
            if include_voices
            else []
        )
        selected_targets = (
            _select_target_rows(
                dataset,
                targets_per_language,
                language.benchmark_code,
                max_chars=max_target_chars,
                min_chars=min_target_chars,
            )
            if include_targets
            else []
        )

        for index, row in enumerate(selected_voices, start=1):
            audio_path = _audio_path(row)
            if audio_path is None:
                continue
            audio_path = _materialize_audio(
                row,
                out_path.parent / "audio_refs" / out_path.stem / f"{language.benchmark_code}_{index:03d}.wav",
            )
            voices.append(
                {
                    "id": f"fleurs_{language.benchmark_code}_voice_{index:03d}",
                    "language": language.benchmark_code,
                    "speaker_id": str(row.get("id") or f"{dataset_code}-row-{index:03d}"),
                    "audio_path": str(audio_path),
                    "transcript": _normalize_text(language.benchmark_code, _row_text(row)),
                    "metadata": {
                        "dataset": dataset_name,
                        "dataset_language": dataset_code,
                        "split": split,
                        "source": "fleurs",
                    },
                }
            )

        for index, row in enumerate(selected_targets, start=1):
            text = _normalize_text(language.benchmark_code, _row_text(row))
            if not text:
                continue
            targets.append(
                {
                    "id": f"fleurs_{language.benchmark_code}_target_{index:03d}",
                    "language": language.benchmark_code,
                    "text": text,
                    "metadata": {
                        "dataset": dataset_name,
                        "dataset_language": dataset_code,
                        "split": split,
                        "source": "fleurs",
                    },
                }
            )

    pairs = _build_pairs(voices, targets, include_mono_lingual)
    config_text = render_benchmark_toml(
        name="fleurs-crosslingual",
        description=(
            "Open-data FLEURS slice for cross-lingual voice-language "
            "disentanglement experiments."
        ),
        models=[{"id": model_id, "backend": model_backend, "params": model_params}],
        metrics=_real_metric_specs(),
        voices=voices,
        targets=targets,
        pairs=pairs,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(config_text, encoding="utf-8")
    return out_path


def render_benchmark_toml(
    *,
    name: str,
    description: str,
    models: list[dict[str, Any]],
    metrics: list[dict[str, Any]],
    voices: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
) -> str:
    lines = [
        f"name = {_toml_string(name)}",
        f"description = {_toml_string(description)}",
        "",
    ]
    for section, rows in [
        ("models", models),
        ("metrics", metrics),
        ("voices", voices),
        ("targets", targets),
        ("pairs", pairs),
    ]:
        for row in rows:
            lines.append(f"[[{section}]]")
            for key, value in row.items():
                if isinstance(value, dict):
                    lines.append(f"{key} = {_inline_table(value)}")
                else:
                    lines.append(f"{key} = {_toml_value(value)}")
            lines.append("")
    return "\n".join(lines)


def _select_voice_rows(
    dataset: Any,
    limit: int,
    language: str | None = None,
    max_chars: int | None = None,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen_speakers: set[str] = set()
    for row in dataset:
        speaker = str(row.get("client_id") or "")
        sentence = _normalize_text(language or "", _row_text(row))
        if not sentence or _audio_path(row) is None:
            continue
        if max_chars is not None and len(sentence) > max_chars:
            continue
        if speaker and speaker in seen_speakers:
            continue
        selected.append(dict(row))
        if speaker:
            seen_speakers.add(speaker)
        if len(selected) >= limit:
            break
    return selected


def _select_target_rows(
    dataset: Any,
    limit: int,
    language: str | None = None,
    max_chars: int | None = None,
    min_chars: int | None = None,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen_texts: set[str] = set()
    for row in dataset:
        text = _normalize_text(language or "", _row_text(row))
        if not text or text in seen_texts:
            continue
        if min_chars is not None and len(text) < min_chars:
            continue
        if max_chars is not None and len(text) > max_chars:
            continue
        selected.append(dict(row))
        seen_texts.add(text)
        if len(selected) >= limit:
            break
    return selected


def _select_speaker_voice_rows(
    dataset: Any,
    *,
    speakers_limit: int,
    utterances_per_speaker: int,
    language: str | None = None,
    max_chars: int | None = None,
) -> list[dict[str, Any]]:
    by_speaker: dict[str, list[dict[str, Any]]] = {}
    speaker_order: list[str] = []
    selected: list[dict[str, Any]] = []

    for row in dataset:
        speaker = _speaker_id(row, "")
        if not speaker:
            continue
        sentence = _normalize_text(language or "", _row_text(row))
        if not sentence or _audio_path(row) is None:
            continue
        if max_chars is not None and len(sentence) > max_chars:
            continue
        if speaker not in by_speaker:
            by_speaker[speaker] = []
            speaker_order.append(speaker)
        if len(by_speaker[speaker]) < utterances_per_speaker:
            by_speaker[speaker].append(dict(row))

    for speaker in speaker_order:
        rows = by_speaker[speaker]
        if len(rows) < utterances_per_speaker:
            continue
        selected.extend(rows[:utterances_per_speaker])
        if len(selected) >= speakers_limit * utterances_per_speaker:
            break

    return selected


def _load_local_common_voice_rows(local_root: Path, dataset_code: str, split: str) -> list[dict[str, Any]]:
    language_root, tsv_path = _local_common_voice_paths(local_root, dataset_code, split)
    rows: list[dict[str, Any]] = []
    with tsv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            audio_path = _local_common_voice_audio_path(language_root, row.get("path", ""))
            if audio_path is None:
                continue
            normalized = dict(row)
            normalized["audio"] = {"path": str(audio_path)}
            normalized["path"] = str(audio_path)
            rows.append(normalized)
    return rows


def _local_common_voice_paths(local_root: Path, dataset_code: str, split: str) -> tuple[Path, Path]:
    codes = [dataset_code]
    alias = _common_voice_locale_alias(dataset_code)
    if alias not in codes:
        codes.append(alias)

    candidates: list[tuple[Path, Path]] = []
    for code in codes:
        language_root = local_root / code
        candidates.append((language_root, language_root / f"{split}.tsv"))
        candidates.append((language_root, language_root / f"{split.replace('-', '_')}.tsv"))
    candidates.append((local_root, local_root / f"{split}.tsv"))
    candidates.append((local_root, local_root / f"{split.replace('-', '_')}.tsv"))

    for language_root, tsv_path in candidates:
        if tsv_path.exists():
            return language_root, tsv_path

    tried = ", ".join(str(path) for _, path in candidates)
    raise RuntimeError(
        f"could not find Common Voice split {split!r} for {dataset_code!r} under {local_root}; tried: {tried}"
    )


def _local_common_voice_audio_path(language_root: Path, raw_path: str) -> Path | None:
    if not raw_path:
        return None
    path = Path(raw_path)
    candidates = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.append(language_root / path)
        candidates.append(language_root / "clips" / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def _common_voice_locale_alias(code: str) -> str:
    aliases = {
        "zh-cn": "zh-CN",
        "zh": "zh-CN",
        "cmn": "zh-CN",
        "en-us": "en",
        "en_us": "en",
        "ru-ru": "ru",
        "ru_ru": "ru",
    }
    return aliases.get(code.casefold(), code)


def _audio_path(row: dict[str, Any]) -> Path | None:
    audio = row.get("audio")
    if hasattr(audio, "get_all_samples"):
        return Path(str(row.get("path") or "decoded_audio.wav"))
    if isinstance(audio, dict) and audio.get("array") is not None:
        return Path(str(audio.get("path") or "decoded_audio.wav"))
    if isinstance(audio, dict) and audio.get("path"):
        return Path(str(audio["path"])).resolve()
    if row.get("path"):
        return Path(str(row["path"])).resolve()
    return None


def _materialize_audio(row: dict[str, Any], out_path: Path) -> Path:
    audio = row.get("audio")
    if hasattr(audio, "get_all_samples"):
        samples = audio.get_all_samples()
        _write_mono_wav(out_path, samples.data, int(samples.sample_rate))
        return out_path.resolve()

    if isinstance(audio, dict) and audio.get("array") is not None:
        sample_rate = int(audio.get("sampling_rate") or 16_000)
        _write_mono_wav(out_path, audio["array"], sample_rate)
        return out_path.resolve()

    existing = _audio_path(row)
    if existing is not None and existing.exists():
        return existing.resolve()
    raise FileNotFoundError(f"could not materialize audio for row with keys: {sorted(row)}")


def _write_mono_wav(path: Path, samples: Any, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(samples, "tolist"):
        samples = samples.tolist()
    samples = _mono_samples(samples)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        frames = bytearray()
        for sample in samples:
            if isinstance(sample, (list, tuple)):
                sample = sample[0] if sample else 0.0
            value = max(-1.0, min(1.0, float(sample)))
            frames.extend(int(value * 32767).to_bytes(2, "little", signed=True))
        handle.writeframes(bytes(frames))


def _mono_samples(samples: Any) -> list[float]:
    if not samples:
        return []
    first = samples[0]
    if isinstance(first, (list, tuple)):
        if len(samples) == 1:
            return [float(value) for value in samples[0]]
        lengths = {len(channel) for channel in samples if isinstance(channel, (list, tuple))}
        if len(lengths) == 1:
            return [
                sum(float(channel[index]) for channel in samples) / len(samples)
                for index in range(next(iter(lengths)))
            ]
    return [float(sample[0] if isinstance(sample, (list, tuple)) else sample) for sample in samples]


def _row_text(row: dict[str, Any]) -> str:
    for key in ("sentence", "transcription", "raw_transcription", "text"):
        value = row.get(key)
        if value:
            return str(value).strip()
    return ""


def _speaker_id(row: dict[str, Any], fallback: str) -> str:
    for key in ("client_id", "speaker_id", "speaker", "reader_id", "user_id"):
        value = row.get(key)
        if value:
            return str(value)
    return fallback


def _normalize_text(language: str, text: str) -> str:
    text = " ".join(text.split())
    if language.casefold() in {"zh", "zh-cn", "cmn", "cmn_hans_cn"}:
        text = text.replace(" ", "")
    return text


def _fleurs_dataset_code(code: str) -> str:
    aliases = {
        "ru": "ru_ru",
        "en": "en_us",
        "zh": "cmn_hans_cn",
        "zh-cn": "cmn_hans_cn",
        "cmn": "cmn_hans_cn",
    }
    return aliases.get(code.casefold(), code)


def _build_pairs(
    voices: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    include_mono_lingual: bool,
) -> list[dict[str, str]]:
    pairs: list[dict[str, str]] = []
    for voice in voices:
        for target in targets:
            if not include_mono_lingual and voice["language"] == target["language"]:
                continue
            pairs.append({"voice": voice["id"], "target": target["id"]})
    return pairs


def _real_metric_specs() -> list[dict[str, Any]]:
    return [
        {
            "id": "asr_error",
            "backend": "faster_whisper_asr",
            "params": {"vad_filter": True, "cpu_model_size": "small", "cpu_compute_type": "int8"},
        },
        {
            "id": "target_language_id",
            "backend": "faster_whisper_lid",
            "params": {"vad_filter": True, "cpu_model_size": "small", "cpu_compute_type": "int8"},
        },
        {
            "id": "speaker_similarity",
            "backend": "speechbrain_speaker_similarity",
            "params": {},
        },
        {
            "id": "source_language_similarity",
            "backend": "speechbrain_language_similarity",
            "params": {},
        },
    ]


def _inline_table(value: dict[str, Any]) -> str:
    if not value:
        return "{}"
    return "{ " + ", ".join(f"{key} = {_toml_value(item)}" for key, item in value.items()) + " }"


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return str(value)
    return _toml_string(str(value))


def _toml_string(value: str) -> str:
    escaped = (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
    )
    return f'"{escaped}"'
