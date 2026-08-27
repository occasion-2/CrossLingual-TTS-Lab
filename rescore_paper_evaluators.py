from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
from dataclasses import replace
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Sequence

from crosslingual_tts_lab.config import BenchmarkConfig, MetricSpec, PairSpec, load_config
from crosslingual_tts_lab.cuda_libs import prepare_ctranslate2_cuda_libraries
from crosslingual_tts_lab.planner import GenerationJob, plan_jobs
from crosslingual_tts_lab.report import write_reports
from crosslingual_tts_lab.runner import score_existing_run_to


PROFILE_ID = "paper-evaluators-medium-cuda-fp16-v1"
WHISPER_MODEL = "medium"
WHISPER_REPOSITORY = "Systran/faster-whisper-medium"
WHISPER_REVISION = "08e178d48790749d25932bbc082711ddcfdfbc4f"
SPEAKER_MODEL = "speechbrain/spkrec-ecapa-voxceleb"
SPEAKER_REVISION = "0f99f2d0ebe89ac095bcc5903c4dd8f72b367286"
LANGUAGE_MODEL = "speechbrain/lang-id-voxlingua107-ecapa"
LANGUAGE_REVISION = "0253049ae131d6a4be1c4f0d8b0ff483a0f8c8e9"
CENTROIDS_SHA256 = "9adca9f8ef996c21f5f3473359bfc7e3b0d852d19139b0af5049621514373d91"

PAPER_RUNS = {
    "f5tts": ("config_f5tts.toml", "results_f5tts", 600),
    "cosyvoice": ("config_cosyvoice.toml", "results_cosyvoice", 600),
    "qwen0_6b": ("config_qwen0_6b.toml", "results_qwen0_6b", 600),
    "qwen1_7b": ("config_qwen1_7b.toml", "results_qwen1_7b", 600),
    "spark_tts": ("config_spark_tts.toml", "results_spark_tts", 399),
    "xtts": ("config_xtts.toml", "results_xtts", 600),
}

EXPECTED_PACKAGES = {
    "faster-whisper": "1.2.1",
    "ctranslate2": "4.8.0",
    "speechbrain": "1.1.0",
    "torch": "2.11.0+cu130",
    "torchaudio": "2.11.0+cu130",
}


def evaluator_metrics() -> list[MetricSpec]:
    whisper_common = {
        "model_size": WHISPER_MODEL,
        "model_revision": WHISPER_REVISION,
        "device": "cuda",
        "compute_type": "float16",
        "vad_filter": True,
        "allow_cpu_fallback": False,
    }
    return [
        MetricSpec(
            id="asr_error",
            backend="faster_whisper_asr",
            params={**whisper_common, "beam_size": 5},
        ),
        MetricSpec(
            id="target_language_id",
            backend="faster_whisper_lid",
            params={**whisper_common, "beam_size": 1},
        ),
        MetricSpec(
            id="speaker_similarity",
            backend="speechbrain_speaker_similarity",
            params={
                "model_id": SPEAKER_MODEL,
                "model_revision": SPEAKER_REVISION,
                "device": "cuda:0",
            },
        ),
        MetricSpec(
            id="source_language_similarity",
            backend="speechbrain_language_similarity",
            params={
                "model_id": LANGUAGE_MODEL,
                "model_revision": LANGUAGE_REVISION,
                "device": "cuda:0",
            },
        ),
    ]


def evaluator_profile() -> dict:
    return {
        "profile_id": PROFILE_ID,
        "package_versions": EXPECTED_PACKAGES,
        "metric_specs": [
            {"id": metric.id, "backend": metric.backend, "params": metric.params}
            for metric in evaluator_metrics()
        ],
        "language_centroids_sha256": CENTROIDS_SHA256,
        "cpu_fallback": False,
    }


def validate_runtime() -> dict[str, str]:
    problems: list[str] = []
    installed: dict[str, str] = {}
    for package, expected in EXPECTED_PACKAGES.items():
        try:
            actual = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            problems.append(f"missing evaluator package {package}=={expected}")
            continue
        installed[package] = actual
        if actual != expected:
            problems.append(f"{package}: expected {expected}, found {actual}")

    try:
        import torch

        if not torch.cuda.is_available():
            problems.append("CUDA is unavailable; CPU fallback is intentionally disabled")
        elif torch.cuda.device_count() < 1:
            problems.append("no CUDA device is visible")
    except Exception as exc:
        problems.append(f"PyTorch CUDA preflight failed: {type(exc).__name__}: {exc}")

    try:
        prepare_ctranslate2_cuda_libraries()
        import ctranslate2

        compute_types = set(ctranslate2.get_supported_compute_types("cuda"))
        if "float16" not in compute_types:
            problems.append(f"CTranslate2 CUDA does not support float16: {sorted(compute_types)}")
    except Exception as exc:
        problems.append(f"CTranslate2 CUDA preflight failed: {type(exc).__name__}: {exc}")

    centroids = Path(__file__).parent / "src/crosslingual_tts_lab/metrics/fleurs_centroids.json"
    if not centroids.exists():
        problems.append(f"missing centroid file: {centroids}")
    elif _sha256_file(centroids) != CENTROIDS_SHA256:
        problems.append("FLEURS centroid file hash does not match the paper evaluator profile")

    if problems:
        raise RuntimeError("evaluator preflight failed:\n- " + "\n- ".join(problems))
    return installed


def prefetch_evaluator_checkpoints() -> dict[str, str]:
    """Populate the three pinned Hub snapshots sequentially before metrics load."""
    from huggingface_hub import snapshot_download

    snapshots: dict[str, str] = {}
    for repository, revision in (
        (WHISPER_REPOSITORY, WHISPER_REVISION),
        (SPEAKER_MODEL, SPEAKER_REVISION),
        (LANGUAGE_MODEL, LANGUAGE_REVISION),
    ):
        snapshots[repository] = snapshot_download(repo_id=repository, revision=revision)
    return snapshots


def paper_rescore_config(
    config: BenchmarkConfig,
    source_manifest: dict,
    model_key: str,
) -> tuple[BenchmarkConfig, list[GenerationJob]]:
    if len(config.models) != 1:
        raise ValueError(f"{model_key}: expected exactly one model in the source config")

    jobs = plan_jobs(config)
    jobs_by_id = {job.id: job for job in jobs}
    samples = source_manifest.get("samples", [])
    samples_by_id = {sample["job_id"]: sample for sample in samples}
    if len(samples_by_id) != len(samples):
        raise ValueError(f"{model_key}: duplicate job IDs in source manifest")
    unknown = sorted(set(samples_by_id) - set(jobs_by_id))
    if unknown:
        raise ValueError(f"{model_key}: source manifest has jobs absent from its config: {unknown[:3]}")

    selected: list[GenerationJob] = []
    for job in jobs:
        sample = samples_by_id.get(job.id)
        if sample is None:
            raise ValueError(f"{model_key}: source manifest is missing planned job {job.id}")
        if sample.get("synthesis_metadata", {}).get("synthetic_placeholder", False):
            continue
        if model_key == "spark_tts" and job.target.language == "ru":
            continue
        selected.append(job)

    expected = PAPER_RUNS[model_key][2]
    if len(selected) != expected:
        raise ValueError(f"{model_key}: expected {expected} paper WAVs, found {len(selected)}")

    filtered = replace(
        config,
        description=f"Evaluator-only rescore profile {PROFILE_ID}; synthesis WAVs are read-only.",
        pairs=[PairSpec(voice=job.voice.id, target=job.target.id) for job in selected],
        metrics=evaluator_metrics(),
    )
    replanned = plan_jobs(filtered)
    if [job.id for job in replanned] != [job.id for job in selected]:
        raise ValueError(f"{model_key}: filtered rescore plan changed source job identity/order")
    return filtered, selected


def validate_source_wavs(source_run: Path, jobs: Sequence[GenerationJob]) -> tuple[str, str]:
    generated_digest = hashlib.sha256()
    reference_digest = hashlib.sha256()
    checked_references: set[Path] = set()
    for job in jobs:
        audio = source_run / "audio" / f"{job.id}.wav"
        if not audio.is_file() or audio.stat().st_size <= 1000:
            raise ValueError(f"missing or invalid source WAV: {audio}")
        audio_hash = _sha256_file(audio)
        generated_digest.update(
            f"{job.id}\0{audio.stat().st_size}\0{audio_hash}\n".encode("utf-8")
        )

        reference = job.voice.audio_path
        if reference not in checked_references:
            if not reference.is_file() or reference.stat().st_size <= 0:
                raise ValueError(f"missing reference audio: {reference}")
            reference_hash = _sha256_file(reference)
            reference_digest.update(
                f"{reference.resolve()}\0{reference.stat().st_size}\0{reference_hash}\n".encode(
                    "utf-8"
                )
            )
            checked_references.add(reference)
    return generated_digest.hexdigest(), reference_digest.hexdigest()


def _sample_validation_failures(sample: dict) -> list[str]:
    job_id = sample.get("job_id", "<missing-job-id>")
    expected_names = {
        "asr_error",
        "target_language_id",
        "speaker_similarity",
        "normalized_leakage_delta",
    }
    raw_metrics = sample.get("metrics", [])
    metrics = {metric.get("name"): metric for metric in raw_metrics}
    failures: list[str] = []
    if len(raw_metrics) != len(expected_names) or set(metrics) != expected_names:
        return [f"{job_id}: metric names {[metric.get('name') for metric in raw_metrics]}"]
    bad = [name for name, metric in metrics.items() if metric.get("status") != "ok"]
    if bad:
        return [f"{job_id}: non-ok metrics {bad}"]
    nonnumeric = [
        name
        for name, metric in metrics.items()
        if isinstance(metric.get("value"), bool)
        or not isinstance(metric.get("value"), (int, float))
        or not math.isfinite(metric["value"])
    ]
    if nonnumeric:
        failures.append(f"{job_id}: non-finite/non-numeric metric values {nonnumeric}")
    for name, beam_size in (("asr_error", 5), ("target_language_id", 1)):
        details = metrics[name].get("details", {})
        expected = {
            "model_size": WHISPER_MODEL,
            "model_revision": WHISPER_REVISION,
            "device": "cuda",
            "compute_type": "float16",
            "beam_size": beam_size,
            "fallback_reason": None,
        }
        if any(key not in details or details[key] != value for key, value in expected.items()):
            failures.append(f"{job_id}: {name} evaluator details mismatch")
    speaker = metrics["speaker_similarity"].get("details", {})
    if (
        speaker.get("model_id") != SPEAKER_MODEL
        or speaker.get("model_revision") != SPEAKER_REVISION
        or speaker.get("device") != "cuda:0"
    ):
        failures.append(f"{job_id}: speaker evaluator details mismatch")
    language = metrics["normalized_leakage_delta"].get("details", {})
    if (
        language.get("model_id") != LANGUAGE_MODEL
        or language.get("model_revision") != LANGUAGE_REVISION
        or language.get("device") != "cuda:0"
    ):
        failures.append(f"{job_id}: language evaluator details mismatch")
    return failures


def validate_rescore_manifest(manifest_path: Path, expected_job_ids: Sequence[str]) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples = manifest.get("samples", [])
    actual_ids = [sample.get("job_id") for sample in samples]
    if actual_ids != list(expected_job_ids):
        raise RuntimeError(
            f"rescore job IDs/order differ from plan: wrote {len(actual_ids)}, "
            f"expected {len(expected_job_ids)}"
        )
    failures: list[str] = []
    for sample in samples:
        failures.extend(_sample_validation_failures(sample))

    if failures:
        raise RuntimeError("rescore validation failed:\n- " + "\n- ".join(failures[:20]))


def _manifest_profile_matches(manifest: dict) -> bool:
    reproducibility = manifest.get("reproducibility", {})
    expected_specs = [
        {"id": metric.id, "backend": metric.backend, "params": metric.params}
        for metric in evaluator_metrics()
    ]
    packages = reproducibility.get("package_versions", {})
    required_attested_packages = EXPECTED_PACKAGES.keys() - {"ctranslate2"}
    return (
        reproducibility.get("metric_specs") == expected_specs
        and all(packages.get(package) == EXPECTED_PACKAGES[package] for package in required_attested_packages)
        # Older partial manifests predate CTranslate2's addition to the runner's
        # package record. Accept absence, but never accept a recorded mismatch.
        and packages.get("ctranslate2", EXPECTED_PACKAGES["ctranslate2"])
        == EXPECTED_PACKAGES["ctranslate2"]
    )


def _sample_matches_job(sample: dict, job: GenerationJob, source_run: Path) -> bool:
    return (
        sample.get("job_id") == job.id
        and sample.get("model", {}).get("id") == job.model.id
        and sample.get("model", {}).get("backend") == job.model.backend
        and sample.get("model", {}).get("params") == job.model.params
        and sample.get("voice", {}).get("id") == job.voice.id
        and sample.get("voice", {}).get("audio_path") == str(job.voice.audio_path)
        and sample.get("target", {}).get("id") == job.target.id
        and sample.get("target", {}).get("language") == job.target.language
        and sample.get("target", {}).get("text") == job.target.text
        and Path(sample.get("audio_path", "")).resolve()
        == (source_run / "audio" / f"{job.id}.wav").resolve()
    )


def _completed_provenance_matches(manifest: dict, provenance: dict) -> bool:
    recorded = manifest.get("evaluator_rescore", {})
    profile = evaluator_profile()
    return all(recorded.get(key) == value for key, value in {**profile, **provenance}.items())


def rescore_one(
    source_root: Path,
    output_root: Path,
    model_key: str,
    *,
    overwrite: bool,
    resume: bool,
    dry_run: bool,
) -> dict:
    config_name, result_name, _ = PAPER_RUNS[model_key]
    config_path = source_root / config_name
    source_run = source_root / result_name
    source_manifest_path = source_run / "manifest.json"
    if not source_manifest_path.is_file():
        raise FileNotFoundError(f"missing source manifest: {source_manifest_path}")
    source_manifest_hash = _sha256_file(source_manifest_path)
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    config, jobs = paper_rescore_config(load_config(config_path), source_manifest, model_key)
    wav_inventory_hash, reference_inventory_hash = validate_source_wavs(source_run, jobs)

    out_run = output_root / result_name
    resolved_out_run = out_run.resolve()
    resolved_source_root = source_root.resolve()
    if resolved_out_run == source_run.resolve() or resolved_out_run.is_relative_to(
        resolved_source_root
    ):
        raise ValueError(f"destination run resolves into the historical source tree: {out_run}")
    out_manifest = out_run / "manifest.json"
    if out_manifest.exists() and not overwrite and not resume:
        raise FileExistsError(f"refusing to overwrite evaluator result: {out_manifest}")

    provenance = {
        "profile_id": PROFILE_ID,
        "source_manifest": str(source_manifest_path),
        "source_manifest_sha256": source_manifest_hash,
        "source_wav_inventory_sha256": wav_inventory_hash,
        "source_wav_count": len(jobs),
        "reference_wav_inventory_sha256": reference_inventory_hash,
        "reference_wav_count": len({job.voice.audio_path.resolve() for job in jobs}),
    }
    if dry_run:
        return provenance

    expected_ids = [job.id for job in jobs]
    reusable: dict[str, dict] = {}
    existing_manifest: dict | None = None
    if resume and out_manifest.exists():
        existing_manifest = json.loads(out_manifest.read_text(encoding="utf-8"))
        existing_samples = existing_manifest.get("samples", [])
        existing_by_id = {sample.get("job_id"): sample for sample in existing_samples}
        if len(existing_by_id) != len(existing_samples):
            raise RuntimeError(f"cannot resume duplicate job IDs in {out_manifest}")
        unknown = sorted(set(existing_by_id) - set(expected_ids))
        if unknown:
            raise RuntimeError(f"cannot resume unknown job IDs in {out_manifest}: {unknown[:3]}")
        if _manifest_profile_matches(existing_manifest):
            reusable = {
                job.id: existing_by_id[job.id]
                for job in jobs
                if job.id in existing_by_id
                and _sample_matches_job(existing_by_id[job.id], job, source_run)
                and not _sample_validation_failures(existing_by_id[job.id])
            }
        if len(reusable) == len(jobs) and _completed_provenance_matches(
            existing_manifest, provenance
        ):
            provenance["resume_reused_rows"] = len(reusable)
            provenance["resume_rescored_rows"] = 0
            provenance["skipped_completed"] = True
            return provenance

    pending = [job for job in jobs if job.id not in reusable]
    out_run.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{result_name}-resume-", dir=out_run.parent) as tmp:
        temporary_run = Path(tmp)
        if pending:
            pending_config = replace(
                config,
                pairs=[PairSpec(voice=job.voice.id, target=job.target.id) for job in pending],
            )
            pending_manifest_path = score_existing_run_to(
                pending_config, source_run, temporary_run
            )
            validate_rescore_manifest(pending_manifest_path, [job.id for job in pending])
            pending_manifest = json.loads(pending_manifest_path.read_text(encoding="utf-8"))
            rescored = {sample["job_id"]: sample for sample in pending_manifest["samples"]}
        else:
            if existing_manifest is None:
                raise RuntimeError("internal error: no existing manifest for an empty resume")
            pending_manifest = existing_manifest
            rescored = {}

        merged = dict(pending_manifest if existing_manifest is None else existing_manifest)
        merged["benchmark"] = pending_manifest["benchmark"]
        merged["device_profile"] = pending_manifest["device_profile"]
        merged["reproducibility"] = pending_manifest["reproducibility"]
        merged["summary"] = {
            "models": len(config.models),
            "voices": len(config.voices),
            "targets": len(config.targets),
            "pairs": len(config.pairs),
            "jobs": len(jobs),
            "cross_lingual_jobs": sum(job.is_cross_lingual for job in jobs),
        }
        merged["samples"] = [
            reusable[job.id] if job.id in reusable else rescored[job.id] for job in jobs
        ]

        candidate = temporary_run / "merged-manifest.json"
        candidate.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
        validate_rescore_manifest(candidate, expected_ids)
        merged["evaluator_rescore"] = {
            **evaluator_profile(),
            **provenance,
            "resume_reused_rows": len(reusable),
            "resume_rescored_rows": len(pending),
            "skipped_completed": False,
        }

        out_run.mkdir(parents=True, exist_ok=True)
        temporary = out_manifest.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
        temporary.replace(out_manifest)
    write_reports(out_manifest)
    if _sha256_file(source_manifest_path) != source_manifest_hash:
        raise RuntimeError(f"historical source manifest changed during rescore: {source_manifest_path}")
    provenance["resume_reused_rows"] = len(reusable)
    provenance["resume_rescored_rows"] = len(pending)
    provenance["skipped_completed"] = False
    return provenance


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_roots(source_root: Path, output_root: Path) -> tuple[Path, Path]:
    source = source_root.resolve()
    output = output_root.resolve()
    if source == output or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("source and output roots must be separate, non-overlapping trees")
    return source, output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rescore every paper-analyzed historical WAV with the pinned evaluator stack."
    )
    parser.add_argument("--source-root", type=Path, default=Path("overnight_runs"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model", action="append", choices=tuple(PAPER_RUNS))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--skip-runtime-checks",
        action="store_true",
        help="only for filesystem/config dry-runs; scoring always requires runtime checks",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.overwrite and args.resume:
        raise SystemExit("--overwrite and --resume are mutually exclusive")
    source_root, output_root = _safe_roots(args.source_root, args.output_root)
    if args.skip_runtime_checks and not args.dry_run:
        raise SystemExit("--skip-runtime-checks is allowed only with --dry-run")
    if not args.skip_runtime_checks:
        validate_runtime()
    if not args.dry_run:
        prefetch_evaluator_checkpoints()

    selected = args.model or list(PAPER_RUNS)
    results = {}
    for model_key in selected:
        results[model_key] = rescore_one(
            source_root,
            output_root,
            model_key,
            overwrite=args.overwrite,
            resume=args.resume,
            dry_run=args.dry_run,
        )
        print(f"{model_key}: validated {results[model_key]['source_wav_count']} source WAVs")

    if not args.dry_run:
        output_root.mkdir(parents=True, exist_ok=True)
        profile_path = output_root / "evaluator_profile.json"
        profile_path.write_text(
            json.dumps({**evaluator_profile(), "runs": results}, indent=2),
            encoding="utf-8",
        )
        print(f"wrote {profile_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
