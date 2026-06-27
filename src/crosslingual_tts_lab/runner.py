from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

from tqdm import tqdm

from crosslingual_tts_lab.backends import create_backend
from crosslingual_tts_lab.backends.base import TTSBackend
from crosslingual_tts_lab.config import BenchmarkConfig
from crosslingual_tts_lab.device import detect_device_profile
from crosslingual_tts_lab.metrics import MetricResult, create_metrics
from crosslingual_tts_lab.planner import GenerationJob, plan_jobs
from crosslingual_tts_lab.report import write_reports
from crosslingual_tts_lab.runner_types import GeneratedSample


def run_benchmark(config: BenchmarkConfig, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = out_dir / "audio"
    return _execute_benchmark(config, out_dir, SynthesizingSampleSource(audio_dir))


def score_existing_run(config: BenchmarkConfig, out_dir: Path) -> Path:
    audio_dir = out_dir / "audio"
    if not audio_dir.exists():
        raise FileNotFoundError(f"missing run audio directory: {audio_dir}")

    existing_metadata = _load_existing_synthesis_metadata(out_dir / "manifest.json")
    return _execute_benchmark(
        config,
        out_dir,
        ExistingRunSampleSource(audio_dir, existing_metadata),
    )


class SampleSource(Protocol):
    def sample_for(self, job: GenerationJob) -> GeneratedSample:
        """Return a generated sample for a planned job."""


@dataclass
class SynthesizingSampleSource:
    audio_dir: Path
    backend_cache: dict[tuple[str, str], TTSBackend] = field(default_factory=dict)
    existing_metadata: dict[str, dict] = field(default_factory=dict, init=False)

    def __post_init__(self):
        manifest_path = self.audio_dir.parent / "manifest.json"
        self.existing_metadata = _load_existing_synthesis_metadata(manifest_path)

    def sample_for(self, job: GenerationJob) -> GeneratedSample:
        audio_path = self.audio_dir / f"{job.id}.wav"
        if audio_path.exists() and audio_path.stat().st_size > 1000 and job.id in self.existing_metadata:
            metadata = self.existing_metadata[job.id]
            return GeneratedSample(
                job=job,
                audio_path=audio_path,
                synthesis_metadata=metadata,
            )

        result = self._backend_for(job).synthesize(job, self.audio_dir)
        return GeneratedSample(
            job=job,
            audio_path=result.audio_path,
            synthesis_metadata=result.metadata,
        )

    def _backend_for(self, job: GenerationJob) -> TTSBackend:
        backend_key = _backend_cache_key(job)
        if backend_key not in self.backend_cache:
            self.backend_cache[backend_key] = create_backend(job.model.backend, job.model.params)
        return self.backend_cache[backend_key]


@dataclass(frozen=True)
class ExistingRunSampleSource:
    audio_dir: Path
    metadata_by_job_id: dict[str, dict]

    def sample_for(self, job: GenerationJob) -> GeneratedSample:
        audio_path = self.audio_dir / f"{job.id}.wav"
        if not audio_path.exists():
            raise FileNotFoundError(f"missing generated audio for {job.id}: {audio_path}")
        return GeneratedSample(
            job=job,
            audio_path=audio_path,
            synthesis_metadata=self.metadata_by_job_id.get(job.id, {}),
        )


def _execute_benchmark(
    config: BenchmarkConfig,
    out_dir: Path,
    sample_source: SampleSource,
) -> Path:
    device_profile = detect_device_profile()
    metrics = create_metrics(config.metrics, device_profile)
    jobs = plan_jobs(config)

    samples: list[GeneratedSample] = []
    metric_results: dict[str, list[MetricResult]] = {}
    for job in tqdm(jobs, desc="Running benchmark"):
        sample = sample_source.sample_for(job)
        samples.append(sample)
        if sample.synthesis_metadata.get("synthetic_placeholder", False):
            metric_results[job.id] = [
                MetricResult(
                    name=metric.name,
                    status="missing_backend",
                    value=None,
                    details={"reason": "synthetic placeholder due to unsupported synthesis language"},
                )
                for metric in metrics
            ]
        else:
            metric_results[job.id] = [metric.evaluate(sample) for metric in metrics]

        import gc
        gc.collect()
        import sys
        if "torch" in sys.modules:
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    manifest = _build_manifest(config, jobs, samples, metric_results, device_profile.to_dict())
    return _write_manifest(out_dir, manifest)


def _backend_cache_key(job: GenerationJob) -> tuple[str, str]:
    return (job.model.backend, json.dumps(job.model.params, sort_keys=True))


def _write_manifest(out_dir: Path, manifest: dict) -> Path:
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    write_reports(manifest_path)
    return manifest_path


def _load_existing_synthesis_metadata(manifest_path: Path) -> dict[str, dict]:
    if not manifest_path.exists():
        return {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        sample["job_id"]: dict(sample.get("synthesis_metadata", {}))
        for sample in manifest.get("samples", [])
    }


def _build_manifest(
    config: BenchmarkConfig,
    jobs: list[GenerationJob],
    samples: list[GeneratedSample],
    metric_results: dict[str, list[MetricResult]],
    device_profile: dict,
) -> dict:
    sample_by_job = {sample.job.id: sample for sample in samples}
    return {
        "benchmark": {
            "name": config.name,
            "description": config.description,
            "root": str(config.root),
        },
        "device_profile": device_profile,
        "summary": {
            "models": len(config.models),
            "voices": len(config.voices),
            "targets": len(config.targets),
            "pairs": len(config.pairs),
            "jobs": len(jobs),
            "cross_lingual_jobs": sum(job.is_cross_lingual for job in jobs),
        },
        "samples": [
            {
                "job_id": job.id,
                "model": {"id": job.model.id, "backend": job.model.backend},
                "voice": {
                    "id": job.voice.id,
                    "language": job.voice.language,
                    "speaker_id": job.voice.speaker_id,
                    "audio_path": str(job.voice.audio_path),
                    "emotion": job.voice.emotion,
                },
                "target": {
                    "id": job.target.id,
                    "language": job.target.language,
                    "text": job.target.text,
                    "emotion": job.target.emotion,
                },
                "direction": job.direction,
                "is_cross_lingual": job.is_cross_lingual,
                "audio_path": str(sample_by_job[job.id].audio_path),
                "synthesis_metadata": sample_by_job[job.id].synthesis_metadata,
                "metrics": [asdict(result) for result in metric_results[job.id]],
            }
            for job in jobs
        ],
    }
