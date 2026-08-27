from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing
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


# Bump this whenever evaluator value semantics or implementations change.
PROFILE_ID = "paper-evaluators-medium-cuda-fp16-v1"
WHISPER_MODEL = "medium"
WHISPER_REPOSITORY = "Systran/faster-whisper-medium"
WHISPER_REVISION = "08e178d48790749d25932bbc082711ddcfdfbc4f"
SPEAKER_MODEL = "speechbrain/spkrec-ecapa-voxceleb"
SPEAKER_REVISION = "0f99f2d0ebe89ac095bcc5903c4dd8f72b367286"
LANGUAGE_MODEL = "speechbrain/lang-id-voxlingua107-ecapa"
LANGUAGE_REVISION = "0253049ae131d6a4be1c4f0d8b0ff483a0f8c8e9"
CENTROIDS_SHA256 = "9adca9f8ef996c21f5f3473359bfc7e3b0d852d19139b0af5049621514373d91"

METRIC_OUTPUT_NAMES = {
    "asr_error": "asr_error",
    "target_language_id": "target_language_id",
    "speaker_similarity": "speaker_similarity",
    "source_language_similarity": "normalized_leakage_delta",
}

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


def assert_source_inputs_unchanged(
    source_manifest_path: Path,
    source_manifest_hash: str,
    source_run: Path,
    jobs: Sequence[GenerationJob],
    generated_inventory_hash: str,
    reference_inventory_hash: str,
) -> None:
    """Reject a result assembled from inputs that changed between metric passes."""

    failures: list[str] = []
    if _sha256_file(source_manifest_path) != source_manifest_hash:
        failures.append(f"source manifest changed: {source_manifest_path}")
    current_generated, current_references = validate_source_wavs(source_run, jobs)
    if current_generated != generated_inventory_hash:
        failures.append("generated WAV inventory changed")
    if current_references != reference_inventory_hash:
        failures.append("reference WAV inventory changed")
    if failures:
        raise RuntimeError(
            "historical source inputs changed during rescore:\n- " + "\n- ".join(failures)
        )


def _sample_validation_failures(sample: dict) -> list[str]:
    job_id = sample.get("job_id", "<missing-job-id>")
    expected_names = [
        "asr_error",
        "target_language_id",
        "speaker_similarity",
        "normalized_leakage_delta",
    ]
    raw_metrics = sample.get("metrics", [])
    failures: list[str] = []
    if not isinstance(raw_metrics, list):
        return [f"{job_id}: metrics is not a list"]

    actual_names = [metric.get("name") for metric in raw_metrics if isinstance(metric, dict)]
    if len(raw_metrics) != len(expected_names) or actual_names != expected_names:
        failures.append(
            f"{job_id}: expected canonical metric names {expected_names}, got "
            f"{[_metric_diagnostic(metric) for metric in raw_metrics]}"
        )

    metrics_by_name: dict[str, list[dict]] = {}
    for metric in raw_metrics:
        if isinstance(metric, dict):
            metrics_by_name.setdefault(str(metric.get("name")), []).append(metric)
    for name in expected_names:
        matches = metrics_by_name.get(name, [])
        if len(matches) != 1:
            if not matches:
                failures.append(f"{job_id}: missing metric {name}")
            else:
                failures.append(f"{job_id}: duplicate metric {name} ({len(matches)} records)")
            continue
        failures.extend(_metric_validation_failures(job_id, name, matches[0]))
    return failures


def _metric_diagnostic(metric: object) -> str:
    if not isinstance(metric, dict):
        return repr(metric)
    details = metric.get("details") if isinstance(metric.get("details"), dict) else {}
    diagnostic = f"{metric.get('name')}[{metric.get('status')}]"
    error_type = details.get("error_type")
    message = details.get("error") or details.get("reason")
    if error_type:
        diagnostic += f" {error_type}"
    if message:
        diagnostic += f": {message}"
    return diagnostic


def _metric_validation_failures(job_id: str, name: str, metric: dict) -> list[str]:
    status = metric.get("status")
    if status != "ok":
        return [f"{job_id}: {_metric_diagnostic(metric)}"]

    value = metric.get("value")
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        return [f"{job_id}: {name} has non-finite/non-numeric value {value!r}"]

    failures: list[str] = []
    if name in {"asr_error", "target_language_id"}:
        beam_size = 5 if name == "asr_error" else 1
        details = metric.get("details", {})
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
    elif name == "speaker_similarity":
        details = metric.get("details", {})
        if (
            details.get("model_id") != SPEAKER_MODEL
            or details.get("model_revision") != SPEAKER_REVISION
            or details.get("device") != "cuda:0"
        ):
            failures.append(f"{job_id}: speaker evaluator details mismatch")
    elif name == "normalized_leakage_delta":
        details = metric.get("details", {})
        if (
            details.get("model_id") != LANGUAGE_MODEL
            or details.get("model_revision") != LANGUAGE_REVISION
            or details.get("device") != "cuda:0"
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
        and sample.get("voice", {}).get("language") == job.voice.language
        and sample.get("voice", {}).get("speaker_id") == job.voice.speaker_id
        and sample.get("voice", {}).get("audio_path") == str(job.voice.audio_path)
        and sample.get("voice", {}).get("emotion") == job.voice.emotion
        and sample.get("target", {}).get("id") == job.target.id
        and sample.get("target", {}).get("language") == job.target.language
        and sample.get("target", {}).get("text") == job.target.text
        and sample.get("target", {}).get("emotion") == job.target.emotion
        and sample.get("direction") == job.direction
        and sample.get("is_cross_lingual") == job.is_cross_lingual
        and Path(sample.get("audio_path", "")).resolve()
        == (source_run / "audio" / f"{job.id}.wav").resolve()
    )


def _completed_provenance_matches(manifest: dict, provenance: dict) -> bool:
    recorded = manifest.get("evaluator_rescore", {})
    profile = evaluator_profile()
    return all(recorded.get(key) == value for key, value in {**profile, **provenance}.items())


def _metric_spec_record(spec: MetricSpec) -> dict:
    return {"id": spec.id, "backend": spec.backend, "params": spec.params}


def _metric_pass_attestation(
    provenance: dict,
    spec: MetricSpec,
    jobs: Sequence[GenerationJob],
) -> dict:
    job_ids = [job.id for job in jobs]
    return {
        "profile_id": PROFILE_ID,
        "metric_spec": _metric_spec_record(spec),
        "output_metric_name": METRIC_OUTPUT_NAMES[spec.id],
        "job_count": len(job_ids),
        "job_ids_sha256": hashlib.sha256("\0".join(job_ids).encode("utf-8")).hexdigest(),
        **provenance,
    }


def _metric_pass_validation_failures(
    manifest: dict,
    jobs: Sequence[GenerationJob],
    spec: MetricSpec,
    source_run: Path,
) -> list[str]:
    failures: list[str] = []
    reproducibility = manifest.get("reproducibility", {})
    if not isinstance(reproducibility, dict):
        return [f"{spec.id}: reproducibility metadata is not an object"]
    if reproducibility.get("metric_specs") != [_metric_spec_record(spec)]:
        failures.append(f"{spec.id}: metric spec does not match the pinned isolated pass")
    packages = reproducibility.get("package_versions", {})
    if not isinstance(packages, dict):
        failures.append(f"{spec.id}: package versions are not an object")
        packages = {}
    mismatched_packages = [
        f"{package}={packages.get(package)!r} (expected {expected!r})"
        for package, expected in EXPECTED_PACKAGES.items()
        if packages.get(package) != expected
    ]
    if mismatched_packages:
        failures.append(f"{spec.id}: package mismatch: {', '.join(mismatched_packages)}")

    samples = manifest.get("samples", [])
    if not isinstance(samples, list):
        failures.append(f"{spec.id}: samples is not a list")
        return failures
    if any(not isinstance(sample, dict) for sample in samples):
        failures.append(f"{spec.id}: samples contains a non-object record")
        return failures
    expected_ids = [job.id for job in jobs]
    actual_ids = [sample.get("job_id") for sample in samples]
    if len(samples) != len(jobs) or actual_ids != expected_ids:
        failures.append(
            f"{spec.id}: job IDs/order differ from plan: wrote {len(samples)}, "
            f"expected {len(jobs)}"
        )
        return failures

    output_name = METRIC_OUTPUT_NAMES[spec.id]
    for sample, job in zip(samples, jobs, strict=True):
        if not _sample_matches_job(sample, job, source_run):
            failures.append(f"{job.id}: sample provenance differs from the source job")
            continue
        metrics = sample.get("metrics", [])
        if not isinstance(metrics, list) or len(metrics) != 1:
            diagnostics = (
                [_metric_diagnostic(metric) for metric in metrics]
                if isinstance(metrics, list)
                else repr(metrics)
            )
            failures.append(
                f"{job.id}: {spec.id} pass expected one {output_name} result, got {diagnostics}"
            )
            continue
        metric = metrics[0]
        if not isinstance(metric, dict) or metric.get("name") != output_name:
            failures.append(
                f"{job.id}: {spec.id} pass expected {output_name}, got "
                f"{_metric_diagnostic(metric)}"
            )
            continue
        failures.extend(_metric_validation_failures(job.id, output_name, metric))
    return failures


def validate_metric_pass_manifest(
    manifest_path: Path,
    jobs: Sequence[GenerationJob],
    spec: MetricSpec,
    source_run: Path,
) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures = _metric_pass_validation_failures(manifest, jobs, spec, source_run)
    if failures:
        raise RuntimeError(
            f"isolated metric pass {spec.id} failed validation:\n- "
            + "\n- ".join(failures[:20])
        )
    return manifest


def _metric_pass_is_reusable(
    manifest_path: Path,
    jobs: Sequence[GenerationJob],
    spec: MetricSpec,
    source_run: Path,
    attestation: dict,
) -> bool:
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    try:
        return (
            manifest.get("evaluator_metric_pass") == attestation
            and not _metric_pass_validation_failures(manifest, jobs, spec, source_run)
        )
    except (AttributeError, KeyError, TypeError, ValueError):
        return False


def _score_metric_pass_worker(
    config: BenchmarkConfig,
    source_run: Path,
    pass_run: Path,
) -> None:
    """Run inside a spawned process so every metric gets a fresh CUDA context."""

    score_existing_run_to(config, source_run, pass_run)


def _run_isolated_metric_pass(
    config: BenchmarkConfig,
    source_run: Path,
    pass_run: Path,
) -> None:
    context = multiprocessing.get_context("spawn")
    process = context.Process(
        target=_score_metric_pass_worker,
        args=(config, source_run, pass_run),
        name=f"paper-rescore-{config.metrics[0].id}",
    )
    process.start()
    try:
        process.join()
    except BaseException:
        if process.is_alive():
            process.terminate()
            process.join()
        raise
    if process.exitcode != 0:
        raise RuntimeError(
            f"isolated metric process {config.metrics[0].id} exited with code "
            f"{process.exitcode}; its preceding stderr contains the worker failure"
        )


def _write_attested_metric_pass(manifest_path: Path, manifest: dict, attestation: dict) -> None:
    manifest["evaluator_metric_pass"] = attestation
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(manifest_path)


def _merge_metric_passes(
    pass_manifests: Sequence[dict],
    config: BenchmarkConfig,
    jobs: Sequence[GenerationJob],
) -> dict:
    if len(pass_manifests) != len(config.metrics):
        raise RuntimeError("isolated metric pass count does not match evaluator profile")

    first = pass_manifests[0]
    first_samples = {sample["job_id"]: sample for sample in first["samples"]}
    merged_samples: list[dict] = []
    for job in jobs:
        base = dict(first_samples[job.id])
        canonical_metrics: list[dict] = []
        base_without_metrics = {key: value for key, value in base.items() if key != "metrics"}
        for manifest, spec in zip(pass_manifests, config.metrics, strict=True):
            samples_by_id = {sample["job_id"]: sample for sample in manifest["samples"]}
            sample = samples_by_id[job.id]
            if {key: value for key, value in sample.items() if key != "metrics"} != base_without_metrics:
                raise RuntimeError(f"{job.id}: isolated metric passes disagree on sample metadata")
            metric = sample["metrics"][0]
            if metric.get("name") != METRIC_OUTPUT_NAMES[spec.id]:
                raise RuntimeError(f"{job.id}: isolated metric result order/name mismatch")
            canonical_metrics.append(metric)
        base["metrics"] = canonical_metrics
        merged_samples.append(base)

    for manifest in pass_manifests[1:]:
        if manifest.get("benchmark") != first.get("benchmark"):
            raise RuntimeError("isolated metric passes disagree on benchmark metadata")
        if manifest.get("device_profile") != first.get("device_profile"):
            raise RuntimeError("isolated metric passes disagree on device profile")
        if (
            manifest.get("reproducibility", {}).get("model_specs")
            != first.get("reproducibility", {}).get("model_specs")
        ):
            raise RuntimeError("isolated metric passes disagree on model specs")
        if (
            manifest.get("reproducibility", {}).get("package_versions")
            != first.get("reproducibility", {}).get("package_versions")
        ):
            raise RuntimeError("isolated metric passes disagree on package versions")

    merged = dict(first)
    merged["reproducibility"] = dict(first["reproducibility"])
    merged["reproducibility"]["metric_specs"] = [
        _metric_spec_record(spec) for spec in config.metrics
    ]
    merged["samples"] = merged_samples
    merged.pop("evaluator_metric_pass", None)
    return merged


def _score_pending_in_isolated_passes(
    config: BenchmarkConfig,
    jobs: Sequence[GenerationJob],
    source_run: Path,
    out_run: Path,
    provenance: dict,
    *,
    resume: bool,
) -> tuple[dict, int, int]:
    partial_root = out_run / ".metric-passes"
    partial_root.mkdir(parents=True, exist_ok=True)
    pass_manifests: list[dict] = []
    reused_passes = 0
    rescored_passes = 0

    for index, spec in enumerate(config.metrics, start=1):
        pass_run = partial_root / f"{index:02d}-{spec.id}"
        manifest_path = pass_run / "manifest.json"
        attestation = _metric_pass_attestation(provenance, spec, jobs)
        if resume and _metric_pass_is_reusable(
            manifest_path, jobs, spec, source_run, attestation
        ):
            reused_passes += 1
            print(f"{spec.id}: reusing validated isolated metric pass")
        else:
            print(f"{spec.id}: starting isolated metric process")
            pass_config = replace(config, metrics=[spec])
            _run_isolated_metric_pass(pass_config, source_run, pass_run)
            raw_manifest = validate_metric_pass_manifest(
                manifest_path, jobs, spec, source_run
            )
            _write_attested_metric_pass(manifest_path, raw_manifest, attestation)
            rescored_passes += 1

        pass_manifests.append(
            validate_metric_pass_manifest(manifest_path, jobs, spec, source_run)
        )

    return _merge_metric_passes(pass_manifests, config, jobs), reused_passes, rescored_passes


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
    partial_root = out_run / ".metric-passes"
    if (
        partial_root.exists()
        and any(partial_root.glob("*/manifest.json"))
        and not overwrite
        and not resume
    ):
        raise FileExistsError(
            f"validated or diagnostic metric passes already exist under {partial_root}; "
            "use --resume to reuse them or --overwrite to rescore them"
        )

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
        provenance_matches = _completed_provenance_matches(existing_manifest, provenance)
        if _manifest_profile_matches(existing_manifest) and provenance_matches:
            reusable = {
                job.id: existing_by_id[job.id]
                for job in jobs
                if job.id in existing_by_id
                and _sample_matches_job(existing_by_id[job.id], job, source_run)
                and not _sample_validation_failures(existing_by_id[job.id])
            }
        if len(reusable) == len(jobs):
            write_reports(out_manifest)
            provenance["resume_reused_rows"] = len(reusable)
            provenance["resume_rescored_rows"] = 0
            provenance["skipped_completed"] = True
            return provenance

    pending = [job for job in jobs if job.id not in reusable]
    out_run.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{result_name}-resume-", dir=out_run.parent) as tmp:
        temporary_run = Path(tmp)
        reused_metric_passes = 0
        rescored_metric_passes = 0
        if pending:
            pending_config = replace(
                config,
                pairs=[PairSpec(voice=job.voice.id, target=job.target.id) for job in pending],
            )
            pending_manifest, reused_metric_passes, rescored_metric_passes = (
                _score_pending_in_isolated_passes(
                    pending_config,
                    pending,
                    source_run,
                    out_run,
                    provenance,
                    resume=resume,
                )
            )
            pending_manifest_path = temporary_run / "pending-merged-manifest.json"
            pending_manifest_path.write_text(
                json.dumps(pending_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            validate_rescore_manifest(pending_manifest_path, [job.id for job in pending])
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
            "resume_reused_metric_passes": reused_metric_passes,
            "resume_rescored_metric_passes": rescored_metric_passes,
            "metric_execution": "spawned_process_per_metric",
            "skipped_completed": False,
        }

        assert_source_inputs_unchanged(
            source_manifest_path,
            source_manifest_hash,
            source_run,
            jobs,
            wav_inventory_hash,
            reference_inventory_hash,
        )
        out_run.mkdir(parents=True, exist_ok=True)
        temporary = out_manifest.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
        temporary.replace(out_manifest)
    write_reports(out_manifest)
    provenance["resume_reused_rows"] = len(reusable)
    provenance["resume_rescored_rows"] = len(pending)
    provenance["resume_reused_metric_passes"] = reused_metric_passes
    provenance["resume_rescored_metric_passes"] = rescored_metric_passes
    provenance["metric_execution"] = "spawned_process_per_metric"
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
