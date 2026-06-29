from __future__ import annotations

import csv
import http.client
import io
import json
import os
import tarfile
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

from crosslingual_tts_lab.open_datasets import (
    LanguageRequest,
    _common_voice_locale_alias,
    _select_speaker_voice_rows,
    _select_target_rows,
)


DEFAULT_MDC_API_BASE = "https://mozilladatacollective.com/api"

# Common Voice Scripted Speech 26.0, published by Mozilla Data Collective.
# Keep this small default map aligned with the calibration script's default languages.
DEFAULT_SCRIPTED_SPEECH_26_IDS = {
    "en": "cmqim2hn800ssnr07gvmpcnwu",
    "ru": "cmqinj9g500vsnr07qf4hmr3j",
    "zh-CN": "cmqim47x700tunq074za20dq1",
}


@dataclass(frozen=True)
class DownloadedSlice:
    locale: str
    dataset_id: str
    split_path: Path
    clips: int


def load_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if key:
            values[key] = value
    return values


def parse_dataset_ids(value: str | None) -> dict[str, str]:
    parsed = dict(DEFAULT_SCRIPTED_SPEECH_26_IDS)
    if not value:
        return parsed
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"expected LOCALE=DATASET_ID in dataset id map, got {item!r}")
        locale, dataset_id = item.split("=", 1)
        locale = _common_voice_locale_alias(locale.strip())
        dataset_id = dataset_id.strip()
        if not locale or not dataset_id:
            raise ValueError(f"empty locale or dataset id in {item!r}")
        parsed[locale] = dataset_id
    return parsed


def parse_locale_filters(value: str | None) -> dict[str, set[str]]:
    filters: dict[str, set[str]] = {}
    if not value:
        return filters
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"expected LOCALE=VALUE|VALUE in locale filter, got {item!r}")
        locale, raw_values = item.split("=", 1)
        locale = _common_voice_locale_alias(locale.strip())
        values = {part.strip().casefold() for part in raw_values.split("|") if part.strip()}
        if not locale or not values:
            raise ValueError(f"empty locale or values in {item!r}")
        filters[locale] = values
    return filters


def mdc_api_key(*, env_file: Path, env_var: str) -> str:
    value = os.environ.get(env_var, "").strip()
    if value:
        return value
    value = load_env_file(env_file).get(env_var, "").strip()
    if value:
        return value
    raise RuntimeError(
        f"missing Mozilla Data Collective API key; set {env_var} or add it to {env_file}"
    )


def download_common_voice_slices(
    *,
    out_root: Path,
    languages: list[LanguageRequest],
    split: str,
    speakers_per_language: int,
    utterances_per_speaker: int,
    targets_per_language: int,
    max_voice_chars: int | None,
    max_target_chars: int | None,
    min_target_chars: int | None,
    api_key: str,
    dataset_ids: dict[str, str],
    accent_filters: dict[str, set[str]] | None = None,
    archive_cache: Path | None = None,
    api_base: str = DEFAULT_MDC_API_BASE,
    skip_existing: bool = True,
) -> list[DownloadedSlice]:
    out_root.mkdir(parents=True, exist_ok=True)
    archive_cache = archive_cache or out_root.parent / "common_voice_archives"
    downloaded: list[DownloadedSlice] = []
    for language in languages:
        locale = _common_voice_locale_alias(language.dataset_code)
        accent_filter = (accent_filters or {}).get(locale)
        dataset_id = dataset_ids.get(locale)
        if dataset_id is None:
            raise RuntimeError(
                f"no Mozilla Data Collective dataset id configured for locale {locale!r}; "
                "set CV_DATASET_IDS='locale=dataset_id,...'"
            )

        split_path = out_root / locale / f"{split}.tsv"
        if skip_existing and _existing_slice_complete(
            out_root=out_root,
            locale=locale,
            split=split,
            speakers_per_language=speakers_per_language,
            utterances_per_speaker=utterances_per_speaker,
            targets_per_language=targets_per_language,
            benchmark_language=language.benchmark_code,
            max_voice_chars=max_voice_chars,
            max_target_chars=max_target_chars,
            min_target_chars=min_target_chars,
            accent_filter=accent_filter,
        ):
            downloaded.append(
                DownloadedSlice(locale=locale, dataset_id=dataset_id, split_path=split_path, clips=-1)
            )
            continue

        archive_path = _download_archive_to_cache(
            api_base=api_base,
            api_key=api_key,
            dataset_id=dataset_id,
            locale=locale,
            archive_cache=archive_cache,
        )
        with archive_path.open("rb") as archive:
            try:
                result = extract_common_voice_slice(
                    archive,
                    out_root=out_root,
                    locale=locale,
                    split=split,
                    speakers_per_language=speakers_per_language,
                    utterances_per_speaker=utterances_per_speaker,
                    targets_per_language=targets_per_language,
                    benchmark_language=language.benchmark_code,
                    max_voice_chars=max_voice_chars,
                    max_target_chars=max_target_chars,
                    min_target_chars=min_target_chars,
                    accent_filter=accent_filter,
                )
            except tarfile.TarError as exc:
                archive_path.unlink(missing_ok=True)
                raise RuntimeError(
                    f"cached MDC archive for {locale} could not be read and was removed: {archive_path}. "
                    "Rerun the command to resume the archive download."
                ) from exc
        downloaded.append(DownloadedSlice(locale=locale, dataset_id=dataset_id, **result))
    return downloaded


def extract_common_voice_slice(
    archive: BinaryIO,
    *,
    out_root: Path,
    locale: str,
    split: str,
    speakers_per_language: int,
    utterances_per_speaker: int,
    targets_per_language: int,
    benchmark_language: str,
    max_voice_chars: int | None = None,
    max_target_chars: int | None = None,
    min_target_chars: int | None = None,
    accent_filter: set[str] | None = None,
) -> dict[str, Any]:
    language_root = out_root / locale
    clips_root = language_root / "clips"
    language_root.mkdir(parents=True, exist_ok=True)
    clips_root.mkdir(parents=True, exist_ok=True)

    selected_rows: list[dict[str, str]] | None = None
    selected_clip_names: set[str] = set()
    fieldnames: list[str] | None = None
    clips_written = 0

    with tarfile.open(fileobj=archive, mode="r|gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            name = member.name
            if selected_rows is None and _is_split_member(name, locale, split):
                extracted = tar.extractfile(member)
                if extracted is None:
                    continue
                rows, fieldnames = _read_tsv_rows(extracted)
                selected_rows = _selected_subset_rows(
                    rows,
                    speakers_per_language=speakers_per_language,
                    utterances_per_speaker=utterances_per_speaker,
                    targets_per_language=targets_per_language,
                    benchmark_language=benchmark_language,
                    max_voice_chars=max_voice_chars,
                    max_target_chars=max_target_chars,
                    min_target_chars=min_target_chars,
                    accent_filter=accent_filter,
                )
                selected_clip_names = {
                    _clip_name(str(row.get("path", ""))) for row in selected_rows if row.get("path")
                }
                selected_clip_names.discard("")
                _write_subset_tsv(language_root / f"{split}.tsv", fieldnames, selected_rows)
                continue

            if selected_clip_names and _clip_member_name(name) in selected_clip_names:
                extracted = tar.extractfile(member)
                if extracted is None:
                    continue
                target = clips_root / _clip_member_name(name)
                target.write_bytes(extracted.read())
                clips_written += 1
                if clips_written >= len(selected_clip_names):
                    break

    if selected_rows is None:
        raise RuntimeError(f"could not find {split}.tsv for locale {locale!r} in MDC archive")
    if clips_written < len(selected_clip_names):
        missing = len(selected_clip_names) - clips_written
        raise RuntimeError(
            f"Common Voice archive for {locale!r} ended before all selected clips were found "
            f"({missing} missing)"
        )

    return {"split_path": language_root / f"{split}.tsv", "clips": clips_written}


def _request_download_url(*, api_base: str, api_key: str, dataset_id: str) -> str:
    url = f"{api_base.rstrip('/')}/datasets/{dataset_id}/download"
    request = urllib.request.Request(
        url,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            "User-Agent": "xttslab-common-voice-calibration/1.0",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        message = exc.read().decode("utf-8", errors="replace")
        if exc.code == 403 and "Terms must be accepted" in message:
            raise RuntimeError(
                "Mozilla Data Collective refused the download because dataset terms have not "
                f"been accepted for {dataset_id}. Open https://mozilladatacollective.com/datasets/{dataset_id} "
                "while signed in, accept the terms, then rerun the calibration script."
            ) from exc
        raise RuntimeError(f"MDC download URL request failed for {dataset_id}: HTTP {exc.code}: {message}") from exc
    download_url = str(payload.get("downloadUrl") or "")
    if not download_url:
        raise RuntimeError(f"MDC download response for {dataset_id} did not include downloadUrl")
    return download_url


def _download_archive_to_cache(
    *,
    api_base: str,
    api_key: str,
    dataset_id: str,
    locale: str,
    archive_cache: Path,
) -> Path:
    archive_cache.mkdir(parents=True, exist_ok=True)
    url = _request_download_url(api_base=api_base, api_key=api_key, dataset_id=dataset_id)
    remote_name, total_size = _remote_archive_info(url)
    archive_path = archive_cache / remote_name
    part_path = archive_path.with_suffix(archive_path.suffix + ".part")

    if archive_path.exists() and archive_path.stat().st_size == total_size:
        print(f"reusing cached MDC archive for {locale}: {archive_path}")
        return archive_path
    if archive_path.exists() and archive_path.stat().st_size != total_size:
        archive_path.unlink()
    if _path_size(part_path) == total_size:
        part_path.rename(archive_path)
        return archive_path
    if _path_size(part_path) > total_size:
        part_path.unlink()

    attempts = 0
    while _path_size(part_path) < total_size:
        attempts += 1
        if attempts > 20:
            raise RuntimeError(
                f"could not complete MDC archive download for {locale} after {attempts - 1} attempts; "
                f"partial file remains at {part_path}"
            )
        start = part_path.stat().st_size if part_path.exists() else 0
        if start:
            print(f"resuming MDC archive for {locale}: {start}/{total_size} bytes")
        else:
            print(f"downloading MDC archive for {locale}: {total_size} bytes")
        url = _request_download_url(api_base=api_base, api_key=api_key, dataset_id=dataset_id)
        try:
            _append_url_range(url, part_path, start)
        except (OSError, urllib.error.URLError, http.client.IncompleteRead) as exc:
            print(f"MDC archive download interrupted for {locale}: {type(exc).__name__}: {exc}")
            continue

    if part_path.stat().st_size != total_size:
        raise RuntimeError(
            f"MDC archive download size mismatch for {locale}: "
            f"{part_path.stat().st_size} != {total_size}"
        )
    part_path.rename(archive_path)
    return archive_path


def _remote_archive_info(url: str) -> tuple[str, int]:
    request = urllib.request.Request(
        url,
        headers={"Range": "bytes=0-0", "User-Agent": "xttslab-common-voice-calibration/1.0"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        content_range = response.headers.get("content-range", "")
        response.read(1)
    total_size = _total_size_from_content_range(content_range)
    if total_size is None:
        raise RuntimeError("MDC archive server did not return a Content-Range size for ranged request")
    return Path(urllib.parse.urlparse(url).path).name, total_size


def _append_url_range(url: str, part_path: Path, start: int) -> None:
    headers = {"User-Agent": "xttslab-common-voice-calibration/1.0"}
    if start:
        headers["Range"] = f"bytes={start}-"
    request = urllib.request.Request(
        url,
        headers=headers,
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        if start and getattr(response, "status", None) != 206:
            raise RuntimeError("MDC archive server did not honor ranged resume request")
        part_path.parent.mkdir(parents=True, exist_ok=True)
        with part_path.open("ab" if start else "wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)


def _total_size_from_content_range(value: str) -> int | None:
    if "/" not in value:
        return None
    raw_size = value.rsplit("/", 1)[-1].strip()
    if not raw_size.isdigit():
        return None
    return int(raw_size)


def _path_size(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def _existing_slice_complete(
    *,
    out_root: Path,
    locale: str,
    split: str,
    speakers_per_language: int,
    utterances_per_speaker: int,
    targets_per_language: int,
    benchmark_language: str,
    max_voice_chars: int | None,
    max_target_chars: int | None,
    min_target_chars: int | None,
    accent_filter: set[str] | None,
) -> bool:
    split_path = out_root / locale / f"{split}.tsv"
    clips_root = out_root / locale / "clips"
    if not split_path.exists() or not clips_root.exists():
        return False
    with split_path.open("rb") as handle:
        rows, _ = _read_tsv_rows(handle)
    if not rows:
        return False
    if any(not (clips_root / _clip_name(str(row.get("path") or ""))).exists() for row in rows):
        return False
    try:
        _selected_subset_rows(
            rows,
            speakers_per_language=speakers_per_language,
            utterances_per_speaker=utterances_per_speaker,
            targets_per_language=targets_per_language,
            benchmark_language=benchmark_language,
            max_voice_chars=max_voice_chars,
            max_target_chars=max_target_chars,
            min_target_chars=min_target_chars,
            accent_filter=accent_filter,
        )
    except RuntimeError:
        return False
    return True


def _read_tsv_rows(handle: BinaryIO) -> tuple[list[dict[str, str]], list[str]]:
    text = io.StringIO(handle.read().decode("utf-8"), newline="")
    reader = csv.DictReader(text, delimiter="\t")
    rows = [dict(row) for row in reader]
    return rows, list(reader.fieldnames or [])


def _selected_subset_rows(
    rows: list[dict[str, str]],
    *,
    speakers_per_language: int,
    utterances_per_speaker: int,
    targets_per_language: int,
    benchmark_language: str,
    max_voice_chars: int | None,
    max_target_chars: int | None,
    min_target_chars: int | None = None,
    accent_filter: set[str] | None = None,
) -> list[dict[str, str]]:
    if accent_filter:
        rows = [row for row in rows if _matches_accent_filter(row, accent_filter)]

    voice_rows = _select_speaker_voice_rows(
        rows,
        speakers_limit=speakers_per_language,
        utterances_per_speaker=utterances_per_speaker,
        language=benchmark_language,
        max_chars=max_voice_chars,
    )
    target_rows = _select_target_rows(
        rows,
        targets_per_language,
        language=benchmark_language,
        max_chars=max_target_chars,
        min_chars=min_target_chars,
    )
    selected: list[dict[str, str]] = []
    seen_paths: set[str] = set()
    for row in voice_rows + target_rows:
        path = str(row.get("path") or "")
        if not path or path in seen_paths:
            continue
        selected.append(dict(row))
        seen_paths.add(path)

    if len(voice_rows) < speakers_per_language * utterances_per_speaker:
        raise RuntimeError(
            "not enough repeated-speaker Common Voice rows found for the requested calibration slice"
        )
    if len(target_rows) < targets_per_language:
        raise RuntimeError("not enough Common Voice target rows found for the requested calibration slice")
    return selected


def _matches_accent_filter(row: dict[str, str], allowed: set[str]) -> bool:
    accents = str(row.get("accents") or "").strip()
    if not accents:
        return False
    normalized = {part.strip().casefold() for part in accents.replace(";", ",").split(",") if part.strip()}
    return bool(normalized & allowed)


def _write_subset_tsv(path: Path, fieldnames: list[str] | None, rows: list[dict[str, str]]) -> None:
    if not fieldnames:
        fieldnames = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _is_split_member(name: str, locale: str, split: str) -> bool:
    normalized = name.replace("\\", "/")
    return normalized.endswith(f"/{locale}/{split}.tsv") or normalized.endswith(f"/{split}.tsv")


def _clip_member_name(name: str) -> str:
    return Path(name.replace("\\", "/")).name


def _clip_name(path: str) -> str:
    return Path(path.replace("\\", "/")).name
