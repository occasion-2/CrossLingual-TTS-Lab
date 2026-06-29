from __future__ import annotations

import argparse
import json
from pathlib import Path

from crosslingual_tts_lab.common_voice_mdc import (
    DEFAULT_MDC_API_BASE,
    download_common_voice_slices,
    mdc_api_key,
    parse_dataset_ids,
    parse_locale_filters,
)
from crosslingual_tts_lab.config import load_config
from crosslingual_tts_lab.device import detect_device_profile
from crosslingual_tts_lab.open_datasets import (
    build_common_voice_config,
    build_fleurs_config,
    build_local_common_voice_config,
    parse_language_requests,
)
from crosslingual_tts_lab.planner import plan_jobs
from crosslingual_tts_lab.report import write_reports
from crosslingual_tts_lab.runner import run_benchmark, score_existing_run
from crosslingual_tts_lab.calibration import compute_calibration


MINI_CONFIG = """name = "mini-ru-crosslingual"
description = "Tiny smoke-test benchmark for RU prompt voices across EN/ZH targets."

[[models]]
id = "dummy_tts"
backend = "dummy"

[[voices]]
id = "ru_ref_001"
language = "ru"
speaker_id = "cv-ru-demo-001"
audio_path = "data/reference/ru_ref_001.wav"
transcript = "eto korotkaya russkaya referensnaya fraza"
emotion = "neutral"

[[targets]]
id = "en_weather"
language = "en"
text = "The weather changed quickly, but the speaker stayed calm."
emotion = "neutral"

[[targets]]
id = "zh_station"
language = "zh"
text = "今天晚上火车站会非常安静。"
emotion = "neutral"

[[pairs]]
voice = "ru_ref_001"
target = "en_weather"

[[pairs]]
voice = "ru_ref_001"
target = "zh_station"
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="xttslab",
        description="Cross-lingual TTS disentanglement benchmark harness.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="write a starter benchmark config")
    init_parser.add_argument("path", type=Path)

    doctor_parser = subparsers.add_parser("doctor", help="print runtime/GPU recommendations")
    doctor_parser.add_argument("--json", action="store_true", help="print machine-readable JSON")

    dataset_parser = subparsers.add_parser("dataset", help="build configs from open datasets")
    dataset_subparsers = dataset_parser.add_subparsers(dest="dataset_command", required=True)
    cv_parser = dataset_subparsers.add_parser(
        "common-voice",
        help="build a benchmark config from Mozilla Common Voice via Hugging Face datasets",
    )
    cv_parser.add_argument("--out", type=Path, required=True)
    cv_parser.add_argument(
        "--languages",
        required=True,
        help="comma-separated dataset_code[:benchmark_code], e.g. ru:ru,en:en,zh-CN:zh",
    )
    cv_parser.add_argument(
        "--dataset",
        default="mozilla-foundation/common_voice_19_0",
        help="Hugging Face dataset id",
    )
    cv_parser.add_argument(
        "--local-root",
        type=Path,
        help=(
            "local Common Voice corpus root containing locale directories such as en/validated.tsv "
            "and en/clips/*.mp3; when set, no Hugging Face access is used"
        ),
    )
    cv_parser.add_argument("--split", default="validation")
    cv_parser.add_argument("--voices-per-language", type=int, default=2)
    cv_parser.add_argument(
        "--utterances-per-speaker",
        type=int,
        default=1,
        help=(
            "number of reference utterances to keep for each selected Common Voice speaker; "
            "values >1 enable known-speaker calibration while preserving old one-utterance mode by default"
        ),
    )
    cv_parser.add_argument("--targets-per-language", type=int, default=4)
    cv_parser.add_argument("--max-voice-chars", type=int)
    cv_parser.add_argument("--max-target-chars", type=int)
    cv_parser.add_argument("--min-target-chars", type=int)
    cv_parser.add_argument("--model-id", default="dummy_tts")
    cv_parser.add_argument("--model-backend", default="dummy")
    cv_parser.add_argument(
        "--model-param",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="model parameter to write into the generated config; may be repeated",
    )
    cv_parser.add_argument("--include-mono-lingual", action="store_true")

    cv_download_parser = dataset_subparsers.add_parser(
        "common-voice-download",
        help="download a small Common Voice slice from Mozilla Data Collective",
    )
    cv_download_parser.add_argument("--out-root", type=Path, required=True)
    cv_download_parser.add_argument(
        "--languages",
        required=True,
        help="comma-separated dataset_code[:benchmark_code], e.g. ru:ru,en:en,zh-CN:zh",
    )
    cv_download_parser.add_argument("--split", default="validated")
    cv_download_parser.add_argument("--speakers-per-language", type=int, default=5)
    cv_download_parser.add_argument("--utterances-per-speaker", type=int, default=2)
    cv_download_parser.add_argument("--targets-per-language", type=int, default=10)
    cv_download_parser.add_argument("--max-voice-chars", type=int)
    cv_download_parser.add_argument("--max-target-chars", type=int)
    cv_download_parser.add_argument("--min-target-chars", type=int)
    cv_download_parser.add_argument(
        "--dataset-ids",
        help=(
            "comma-separated Mozilla Data Collective ids, e.g. en=...,ru=...; "
            "defaults cover en, ru, and zh-CN for Common Voice Scripted Speech 26.0"
        ),
    )
    cv_download_parser.add_argument(
        "--accent-filters",
        help=(
            "optional locale-specific Common Voice accent labels, e.g. "
            "'en=United States English|England English'; rows with blank or other accents are excluded"
        ),
    )
    cv_download_parser.add_argument("--api-base", default=DEFAULT_MDC_API_BASE)
    cv_download_parser.add_argument("--api-key-env", default="COMMONVOICE_APIKEY")
    cv_download_parser.add_argument("--env-file", type=Path, default=Path(".env"))
    cv_download_parser.add_argument(
        "--archive-cache",
        type=Path,
        help="directory for resumable MDC .tar.gz archive downloads",
    )
    cv_download_parser.add_argument("--force", action="store_true", help="redownload existing locale slices")

    fleurs_parser = dataset_subparsers.add_parser(
        "fleurs",
        help="build a benchmark config from Google FLEURS via Hugging Face datasets",
    )
    fleurs_parser.add_argument("--out", type=Path, required=True)
    fleurs_parser.add_argument(
        "--languages",
        required=True,
        help=(
            "comma-separated dataset_code[:benchmark_code], e.g. "
            "ru:ru,en:en,zh-CN:zh or ru_ru:ru,en_us:en,cmn_hans_cn:zh"
        ),
    )
    fleurs_parser.add_argument("--dataset", default="google/fleurs", help="Hugging Face dataset id")
    fleurs_parser.add_argument("--split", default="validation")
    fleurs_parser.add_argument("--voices-per-language", type=int, default=2)
    fleurs_parser.add_argument("--targets-per-language", type=int, default=4)
    fleurs_parser.add_argument(
        "--voice-languages",
        help="comma-separated benchmark voice languages to keep, e.g. ru,en,zh",
    )
    fleurs_parser.add_argument(
        "--target-languages",
        help="comma-separated benchmark target languages to keep, e.g. en,zh",
    )
    fleurs_parser.add_argument("--max-voice-chars", type=int)
    fleurs_parser.add_argument("--max-target-chars", type=int)
    fleurs_parser.add_argument("--min-target-chars", type=int)
    fleurs_parser.add_argument("--model-id", default="dummy_tts")
    fleurs_parser.add_argument("--model-backend", default="dummy")
    fleurs_parser.add_argument(
        "--model-param",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="model parameter to write into the generated config; may be repeated",
    )
    fleurs_parser.add_argument("--include-mono-lingual", action="store_true")

    plan_parser = subparsers.add_parser("plan", help="validate config and print planned jobs")
    plan_parser.add_argument("--config", "-c", type=Path, required=True)

    run_parser = subparsers.add_parser("run", help="run synthesis and metrics")
    run_parser.add_argument("--config", "-c", type=Path, required=True)
    run_parser.add_argument("--out", "-o", type=Path, required=True)

    score_parser = subparsers.add_parser("score", help="recompute metrics for an existing run")
    score_parser.add_argument("--config", "-c", type=Path, required=True)
    score_parser.add_argument("--run", type=Path, required=True)

    report_parser = subparsers.add_parser("report", help="regenerate Markdown report from a run")
    report_parser.add_argument("--run", type=Path, required=True)

    calibrate_parser = subparsers.add_parser("calibrate", help="compute speaker similarity calibration baselines")
    calibrate_parser.add_argument("--run", type=Path, required=True)

    args = parser.parse_args(argv)

    if args.command == "init":
        return _init(args.path)
    if args.command == "doctor":
        return _doctor(args.json)
    if args.command == "dataset":
        if args.dataset_command == "common-voice":
            return _dataset_common_voice(args)
        if args.dataset_command == "common-voice-download":
            return _dataset_common_voice_download(args)
        if args.dataset_command == "fleurs":
            return _dataset_fleurs(args)
        raise AssertionError(f"unhandled dataset command {args.dataset_command}")
    if args.command == "plan":
        return _plan(args.config)
    if args.command == "run":
        return _run(args.config, args.out)
    if args.command == "score":
        return _score(args.config, args.run)
    if args.command == "report":
        return _report(args.run)
    if args.command == "calibrate":
        return _calibrate(args.run)
    raise AssertionError(f"unhandled command {args.command}")


def _init(path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise SystemExit(f"refusing to overwrite existing config: {path}")
    path.write_text(MINI_CONFIG, encoding="utf-8")
    print(f"wrote {path}")
    return 0


def _doctor(as_json: bool) -> int:
    profile = detect_device_profile()
    if as_json:
        print(json.dumps(profile.to_dict(), indent=2))
        return 0

    print(f"device: {profile.device}")
    print(f"cuda_available: {profile.cuda_available}")
    if profile.torch_version:
        print(f"torch_version: {profile.torch_version}")
    if profile.torch_cuda_version:
        print(f"torch_cuda_version: {profile.torch_cuda_version}")
    if profile.cuda_device_count is not None:
        print(f"cuda_device_count: {profile.cuda_device_count}")
    if profile.gpu_name:
        print(f"gpu_name: {profile.gpu_name}")
    if profile.total_vram_gb is not None:
        print(f"total_vram_gb: {profile.total_vram_gb}")
    print(f"recommended_whisper_model: {profile.recommended_whisper_model}")
    print(f"recommended_compute_type: {profile.recommended_compute_type}")
    for note in profile.notes:
        print(f"note: {note}")
    return 0


def _dataset_common_voice(args: argparse.Namespace) -> int:
    languages = parse_language_requests(args.languages)
    try:
        if args.local_root:
            out_path = build_local_common_voice_config(
                out_path=args.out,
                local_root=args.local_root,
                languages=languages,
                split=args.split,
                voices_per_language=args.voices_per_language,
                targets_per_language=args.targets_per_language,
                model_id=args.model_id,
                model_backend=args.model_backend,
                model_params=_parse_key_values(args.model_param),
                include_mono_lingual=args.include_mono_lingual,
                utterances_per_speaker=args.utterances_per_speaker,
                max_voice_chars=args.max_voice_chars,
                max_target_chars=args.max_target_chars,
                min_target_chars=args.min_target_chars,
            )
        else:
            out_path = build_common_voice_config(
                out_path=args.out,
                languages=languages,
                dataset_name=args.dataset,
                split=args.split,
                voices_per_language=args.voices_per_language,
                targets_per_language=args.targets_per_language,
                model_id=args.model_id,
                model_backend=args.model_backend,
                model_params=_parse_key_values(args.model_param),
                include_mono_lingual=args.include_mono_lingual,
                utterances_per_speaker=args.utterances_per_speaker,
                max_voice_chars=args.max_voice_chars,
                max_target_chars=args.max_target_chars,
                min_target_chars=args.min_target_chars,
            )
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"wrote {out_path}")
    return 0


def _dataset_fleurs(args: argparse.Namespace) -> int:
    languages = parse_language_requests(args.languages)
    try:
        out_path = build_fleurs_config(
            out_path=args.out,
            languages=languages,
            dataset_name=args.dataset,
            split=args.split,
            voices_per_language=args.voices_per_language,
            targets_per_language=args.targets_per_language,
            model_id=args.model_id,
            model_backend=args.model_backend,
            model_params=_parse_key_values(args.model_param),
            include_mono_lingual=args.include_mono_lingual,
            voice_languages=_parse_optional_csv(args.voice_languages),
            target_languages=_parse_optional_csv(args.target_languages),
            max_voice_chars=args.max_voice_chars,
            max_target_chars=args.max_target_chars,
            min_target_chars=args.min_target_chars,
        )
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"wrote {out_path}")
    return 0


def _dataset_common_voice_download(args: argparse.Namespace) -> int:
    languages = parse_language_requests(args.languages)
    try:
        slices = download_common_voice_slices(
            out_root=args.out_root,
            languages=languages,
            split=args.split,
            speakers_per_language=args.speakers_per_language,
            utterances_per_speaker=args.utterances_per_speaker,
            targets_per_language=args.targets_per_language,
            max_voice_chars=args.max_voice_chars,
            max_target_chars=args.max_target_chars,
            min_target_chars=args.min_target_chars,
            api_key=mdc_api_key(env_file=args.env_file, env_var=args.api_key_env),
            dataset_ids=parse_dataset_ids(args.dataset_ids),
            accent_filters=parse_locale_filters(args.accent_filters),
            archive_cache=args.archive_cache,
            api_base=args.api_base,
            skip_existing=not args.force,
        )
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    for item in slices:
        status = "reused" if item.clips < 0 else f"downloaded {item.clips} clips"
        print(f"{status}: {item.locale} -> {item.split_path}")
    return 0


def _parse_key_values(items: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"expected KEY=VALUE for --model-param, got {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise SystemExit(f"empty key in --model-param {item!r}")
        parsed[key] = value.strip()
    return parsed


def _parse_optional_csv(value: str | None) -> set[str] | None:
    if value is None:
        return None
    parsed = {item.strip() for item in value.split(",") if item.strip()}
    return parsed or None


def _plan(config_path: Path) -> int:
    config = load_config(config_path)
    jobs = plan_jobs(config)
    print(f"benchmark: {config.name}")
    print(f"jobs: {len(jobs)}")
    if config.metrics:
        print("metrics:")
        for metric in config.metrics:
            print(f"- {metric.id}: {metric.backend}")
    for job in jobs:
        cross = "cross-lingual" if job.is_cross_lingual else "monolingual"
        print(f"- {job.id}: {job.direction} ({cross})")
    return 0


def _run(config_path: Path, out_dir: Path) -> int:
    config = load_config(config_path)
    manifest_path = run_benchmark(config, out_dir)
    print(f"wrote {manifest_path}")
    print(f"wrote {manifest_path.with_name('report.md')}")
    return 0


def _score(config_path: Path, run_dir: Path) -> int:
    config = load_config(config_path)
    manifest_path = score_existing_run(config, run_dir)
    print(f"wrote {manifest_path}")
    print(f"wrote {manifest_path.with_name('report.md')}")
    return 0


def _report(run_dir: Path) -> int:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"missing run manifest: {manifest_path}")
    report_path = write_reports(manifest_path)
    print(f"wrote {report_path}")
    return 0


def _calibrate(run_dir: Path) -> int:
    try:
        compute_calibration(run_dir)
    except Exception as exc:
        raise SystemExit(str(exc)) from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
