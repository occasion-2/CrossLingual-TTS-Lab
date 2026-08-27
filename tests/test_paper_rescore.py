from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, call, patch

from crosslingual_tts_lab.config import load_config
from crosslingual_tts_lab.planner import plan_jobs
from crosslingual_tts_lab.metrics.leakage import SpeechBrainLanguageSimilarityMetric
from crosslingual_tts_lab.runner import run_benchmark, score_existing_run_to
from rescore_paper_evaluators import (
    EXPECTED_PACKAGES,
    LANGUAGE_MODEL,
    LANGUAGE_REVISION,
    PAPER_RUNS,
    SPEAKER_MODEL,
    SPEAKER_REVISION,
    WHISPER_MODEL,
    WHISPER_REPOSITORY,
    WHISPER_REVISION,
    _sample_validation_failures,
    _safe_roots,
    _score_pending_in_isolated_passes,
    assert_source_inputs_unchanged,
    evaluator_profile,
    evaluator_metrics,
    prefetch_evaluator_checkpoints,
    rescore_one,
    validate_source_wavs,
)


class PaperRescoreTests(unittest.TestCase):
    @staticmethod
    def _valid_metrics() -> list[dict]:
        whisper = {
            "model_size": WHISPER_MODEL,
            "model_revision": WHISPER_REVISION,
            "device": "cuda",
            "compute_type": "float16",
            "fallback_reason": None,
        }
        return [
            {"name": "asr_error", "status": "ok", "value": 0.01, "details": {**whisper, "beam_size": 5}},
            {"name": "target_language_id", "status": "ok", "value": 0.99, "details": {**whisper, "beam_size": 1}},
            {
                "name": "speaker_similarity",
                "status": "ok",
                "value": 0.8,
                "details": {"model_id": SPEAKER_MODEL, "model_revision": SPEAKER_REVISION, "device": "cuda:0"},
            },
            {
                "name": "normalized_leakage_delta",
                "status": "ok",
                "value": -0.2,
                "details": {"model_id": LANGUAGE_MODEL, "model_revision": LANGUAGE_REVISION, "device": "cuda:0"},
            },
        ]

    def test_profile_pins_evaluators_and_disables_cpu_fallback(self) -> None:
        metrics = {metric.id: metric for metric in evaluator_metrics()}

        for metric_id, beam_size in (("asr_error", 5), ("target_language_id", 1)):
            params = metrics[metric_id].params
            self.assertEqual(params["model_size"], "medium")
            self.assertEqual(params["model_revision"], WHISPER_REVISION)
            self.assertEqual(params["device"], "cuda")
            self.assertEqual(params["compute_type"], "float16")
            self.assertEqual(params["beam_size"], beam_size)
            self.assertIs(params["allow_cpu_fallback"], False)

        self.assertEqual(EXPECTED_PACKAGES["faster-whisper"], "1.2.1")
        self.assertEqual(EXPECTED_PACKAGES["speechbrain"], "1.1.0")
        for metric_id in ("speaker_similarity", "source_language_similarity"):
            params = metrics[metric_id].params
            self.assertTrue(params["model_revision"])
            self.assertEqual(params["device"], "cuda:0")

    def test_separate_rescore_preserves_source_manifest(self) -> None:
        config = load_config(Path("configs/mini.toml"))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "historical"
            destination = root / "rescore"
            source_manifest = run_benchmark(config, source)
            before = source_manifest.read_bytes()

            destination_manifest = score_existing_run_to(config, source, destination)

            self.assertEqual(source_manifest.read_bytes(), before)
            self.assertTrue(destination_manifest.is_file())
            self.assertTrue(destination_manifest.with_name("report.md").is_file())
            result = json.loads(destination_manifest.read_text(encoding="utf-8"))
            self.assertEqual(len(result["samples"]), 3)
            self.assertTrue(
                all(Path(sample["audio_path"]).parent == source / "audio" for sample in result["samples"])
            )
            self.assertFalse((destination / "audio").exists())

    def test_source_inventory_hashes_generated_and_reference_wavs(self) -> None:
        config = load_config(Path("configs/mini.toml"))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            audio_dir = source / "audio"
            audio_dir.mkdir(parents=True)
            jobs = []
            for index, job in enumerate(plan_jobs(config)):
                reference = root / f"reference-{index}.wav"
                reference.write_bytes(b"r" * 1200)
                job = replace(
                    job,
                    voice=replace(job.voice, audio_path=reference),
                )
                (audio_dir / f"{job.id}.wav").write_bytes(b"g" * (1201 + index))
                jobs.append(job)

            generated, references = validate_source_wavs(source, jobs)
            source_manifest = source / "manifest.json"
            source_manifest.write_text('{"samples": []}', encoding="utf-8")
            source_manifest_hash = hashlib.sha256(source_manifest.read_bytes()).hexdigest()

            self.assertEqual(len(generated), hashlib.sha256().digest_size * 2)
            self.assertEqual(len(references), hashlib.sha256().digest_size * 2)
            assert_source_inputs_unchanged(
                source_manifest,
                source_manifest_hash,
                source,
                jobs,
                generated,
                references,
            )
            (jobs[0].voice.audio_path).write_bytes(b"changed" * 200)
            _, changed_references = validate_source_wavs(source, jobs)
            self.assertNotEqual(references, changed_references)
            with self.assertRaisesRegex(RuntimeError, "reference WAV inventory changed"):
                assert_source_inputs_unchanged(
                    source_manifest,
                    source_manifest_hash,
                    source,
                    jobs,
                    generated,
                    references,
                )

    def test_output_root_must_be_outside_historical_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "historical"
            with self.assertRaises(ValueError):
                _safe_roots(source, source / "rescored")
            with self.assertRaises(ValueError):
                _safe_roots(source, Path(tmp))

    def test_strict_row_validation_rejects_missing_or_mismatched_details(self) -> None:
        sample = {"job_id": "job", "metrics": self._valid_metrics()}
        self.assertEqual(_sample_validation_failures(sample), [])
        del sample["metrics"][0]["details"]["fallback_reason"]
        self.assertTrue(_sample_validation_failures(sample))
        sample["metrics"][0]["details"]["fallback_reason"] = None
        sample["metrics"][1]["details"]["model_revision"] = "wrong"
        self.assertTrue(_sample_validation_failures(sample))

        failed = {"job_id": "long-job", "metrics": self._valid_metrics()}
        failed["metrics"][3] = {
            "name": "normalized_leakage_delta",
            "status": "error",
            "value": None,
            "details": {"error_type": "OutOfMemoryError", "error": "CUDA out of memory"},
        }
        diagnostic = "\n".join(_sample_validation_failures(failed))
        self.assertIn("OutOfMemoryError", diagnostic)
        self.assertIn("CUDA out of memory", diagnostic)

    def test_leakage_output_name_is_invariant_when_evaluation_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            reference = Path(tmp) / "reference.wav"
            reference.write_bytes(b"audio")
            sample = Mock()
            sample.job.voice.audio_path = reference
            metric = SpeechBrainLanguageSimilarityMetric(
                "source_language_similarity", {}, Mock(device="cpu")
            )
            with patch.object(metric, "_calculate_delta", side_effect=RuntimeError("oom")):
                result = metric.evaluate(sample)

        self.assertEqual(result.name, "normalized_leakage_delta")
        self.assertEqual(result.status, "error")
        self.assertEqual(result.details["error_type"], "RuntimeError")

    def test_isolated_passes_persist_and_resume_after_interruption(self) -> None:
        config = replace(load_config(Path("configs/mini.toml")), metrics=evaluator_metrics())
        jobs = plan_jobs(config)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            # The dummy synthesis creates stable source WAVs and sample metadata;
            # no real evaluator is loaded in this test.
            source_config = replace(config, metrics=[])
            source_manifest_path = run_benchmark(source_config, source)
            source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
            source_samples = {
                sample["job_id"]: sample for sample in source_manifest["samples"]
            }
            output = root / "rescore"
            provenance = {
                "source_manifest": str(source_manifest_path),
                "source_manifest_sha256": "source-hash",
                "source_wav_inventory_sha256": "wav-hash",
            }

            output_by_spec = {
                "asr_error": self._valid_metrics()[0],
                "target_language_id": self._valid_metrics()[1],
                "speaker_similarity": self._valid_metrics()[2],
                "source_language_similarity": self._valid_metrics()[3],
            }

            def write_pass(pass_config, _source_run, destination):
                spec = pass_config.metrics[0]
                manifest = {
                    "benchmark": {"name": config.name},
                    "device_profile": {"device": "cuda"},
                    "reproducibility": {
                        "model_specs": [
                            {
                                "id": config.models[0].id,
                                "backend": config.models[0].backend,
                                "params": config.models[0].params,
                            }
                        ],
                        "metric_specs": [
                            {"id": spec.id, "backend": spec.backend, "params": spec.params}
                        ],
                        "package_versions": EXPECTED_PACKAGES,
                    },
                    "summary": {"jobs": len(jobs)},
                    "samples": [],
                }
                for job in jobs:
                    sample = deepcopy(source_samples[job.id])
                    sample["metrics"] = [deepcopy(output_by_spec[spec.id])]
                    manifest["samples"].append(sample)
                destination.mkdir(parents=True, exist_ok=True)
                destination.joinpath("manifest.json").write_text(
                    json.dumps(manifest), encoding="utf-8"
                )

            first_calls: list[str] = []

            def interrupt_on_third(pass_config, source_run, destination):
                metric_id = pass_config.metrics[0].id
                first_calls.append(metric_id)
                if metric_id == "speaker_similarity":
                    raise KeyboardInterrupt()
                write_pass(pass_config, source_run, destination)

            with patch(
                "rescore_paper_evaluators._run_isolated_metric_pass",
                side_effect=interrupt_on_third,
            ):
                with self.assertRaises(KeyboardInterrupt):
                    _score_pending_in_isolated_passes(
                        config, jobs, source, output, provenance, resume=True
                    )

            self.assertEqual(
                first_calls,
                ["asr_error", "target_language_id", "speaker_similarity"],
            )
            self.assertTrue(
                output.joinpath(".metric-passes/01-asr_error/manifest.json").is_file()
            )
            self.assertTrue(
                output.joinpath(".metric-passes/02-target_language_id/manifest.json").is_file()
            )

            resumed_calls: list[str] = []

            def finish_remaining(pass_config, source_run, destination):
                resumed_calls.append(pass_config.metrics[0].id)
                write_pass(pass_config, source_run, destination)

            with patch(
                "rescore_paper_evaluators._run_isolated_metric_pass",
                side_effect=finish_remaining,
            ):
                merged, reused, rescored = _score_pending_in_isolated_passes(
                    config, jobs, source, output, provenance, resume=True
                )

            self.assertEqual(
                resumed_calls, ["speaker_similarity", "source_language_similarity"]
            )
            self.assertEqual((reused, rescored), (2, 2))
            self.assertTrue(
                all(
                    [metric["name"] for metric in sample["metrics"]]
                    == [
                        "asr_error",
                        "target_language_id",
                        "speaker_similarity",
                        "normalized_leakage_delta",
                    ]
                    for sample in merged["samples"]
                )
            )

    def test_checkpoint_prefetch_is_sequential_and_revision_pinned(self) -> None:
        downloader = Mock(side_effect=lambda repo_id, revision: f"/{repo_id}/{revision}")
        with patch("huggingface_hub.snapshot_download", downloader):
            prefetch_evaluator_checkpoints()
        self.assertEqual(
            downloader.call_args_list,
            [
                call(repo_id=WHISPER_REPOSITORY, revision=WHISPER_REVISION),
                call(repo_id=SPEAKER_MODEL, revision=SPEAKER_REVISION),
                call(repo_id=LANGUAGE_MODEL, revision=LANGUAGE_REVISION),
            ],
        )

    def test_resume_reuses_only_exact_valid_rows_and_skips_completed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_root = root / "source"
            output_root = root / "output"
            source_run = source_root / "results_test"
            source_run.joinpath("audio").mkdir(parents=True)
            reference = source_root / "reference.wav"
            reference.write_bytes(b"r" * 1200)
            config_path = source_root / "config_test.toml"
            config_path.write_text(
                f'''name = "resume-test"
[[models]]
id = "model"
backend = "dummy"
[[voices]]
id = "voice"
language = "ru"
speaker_id = "speaker"
audio_path = "{reference}"
[[targets]]
id = "one"
language = "en"
text = "one"
[[targets]]
id = "two"
language = "en"
text = "two"
[[pairs]]
voice = "voice"
target = "one"
[[pairs]]
voice = "voice"
target = "two"
''',
                encoding="utf-8",
            )
            config = load_config(config_path)
            jobs = plan_jobs(config)
            for job in jobs:
                source_run.joinpath("audio", f"{job.id}.wav").write_bytes(b"g" * 1201)
            source_manifest_path = source_run / "manifest.json"
            source_manifest_path.write_text(
                json.dumps({"samples": [{"job_id": job.id, "synthesis_metadata": {}} for job in jobs]}),
                encoding="utf-8",
            )
            wav_inventory_hash, reference_inventory_hash = validate_source_wavs(
                source_run, jobs
            )
            source_provenance = {
                "source_manifest": str(source_manifest_path),
                "source_manifest_sha256": hashlib.sha256(
                    source_manifest_path.read_bytes()
                ).hexdigest(),
                "source_wav_inventory_sha256": wav_inventory_hash,
                "source_wav_count": len(jobs),
                "reference_wav_inventory_sha256": reference_inventory_hash,
                "reference_wav_count": 1,
            }

            def sample_for(job, metrics):
                return {
                    "job_id": job.id,
                    "model": {"id": job.model.id, "backend": job.model.backend, "params": job.model.params},
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
                    "audio_path": str(source_run / "audio" / f"{job.id}.wav"),
                    "metrics": metrics,
                }

            specs = [{"id": metric.id, "backend": metric.backend, "params": metric.params} for metric in evaluator_metrics()]
            valid = sample_for(jobs[0], self._valid_metrics())
            valid["reuse_marker"] = True
            invalid_metrics = self._valid_metrics()
            invalid_metrics[0] = {"name": "asr_error", "status": "error", "value": None, "details": {}}
            invalid = sample_for(jobs[1], invalid_metrics)
            out_run = output_root / "results_test"
            out_run.mkdir(parents=True)
            out_run.joinpath("manifest.json").write_text(
                json.dumps(
                    {
                        "reproducibility": {
                            "metric_specs": specs,
                            "package_versions": EXPECTED_PACKAGES,
                        },
                        "evaluator_rescore": {
                            **evaluator_profile(),
                            **source_provenance,
                        },
                        "samples": [valid, invalid],
                    }
                ),
                encoding="utf-8",
            )

            scored_batches: list[list[str]] = []

            def fake_score(pending_config, _source_run, destination):
                pending_jobs = plan_jobs(pending_config)
                scored_batches.append([job.id for job in pending_jobs])
                self.assertEqual(len(pending_config.metrics), 1)
                metric_spec = pending_config.metrics[0]
                output_by_spec = {
                    "asr_error": self._valid_metrics()[0],
                    "target_language_id": self._valid_metrics()[1],
                    "speaker_similarity": self._valid_metrics()[2],
                    "source_language_similarity": self._valid_metrics()[3],
                }
                manifest = {
                    "benchmark": {"name": "resume-test"},
                    "device_profile": {},
                    "reproducibility": {
                        "model_specs": [
                            {
                                "id": pending_jobs[0].model.id,
                                "backend": pending_jobs[0].model.backend,
                                "params": pending_jobs[0].model.params,
                            }
                        ],
                        "metric_specs": [
                            {
                                "id": metric_spec.id,
                                "backend": metric_spec.backend,
                                "params": metric_spec.params,
                            }
                        ],
                        "package_versions": EXPECTED_PACKAGES,
                    },
                    "summary": {},
                    "samples": [
                        sample_for(job, [output_by_spec[metric_spec.id]])
                        for job in pending_jobs
                    ],
                }
                destination.mkdir(parents=True, exist_ok=True)
                path = destination / "manifest.json"
                path.write_text(json.dumps(manifest), encoding="utf-8")

            with patch.dict(PAPER_RUNS, {"test": ("config_test.toml", "results_test", 2)}), patch(
                "rescore_paper_evaluators._run_isolated_metric_pass", side_effect=fake_score
            ) as scorer:
                result = rescore_one(source_root, output_root, "test", overwrite=False, resume=True, dry_run=False)
                self.assertEqual(result["resume_reused_rows"], 1)
                self.assertEqual(result["resume_rescored_rows"], 1)
                self.assertEqual(scored_batches, [[jobs[1].id]] * 4)
                merged = json.loads(out_run.joinpath("manifest.json").read_text(encoding="utf-8"))
                self.assertTrue(merged["samples"][0]["reuse_marker"])
                self.assertEqual(_sample_validation_failures(merged["samples"][1]), [])

                second = rescore_one(source_root, output_root, "test", overwrite=False, resume=True, dry_run=False)
                self.assertTrue(second["skipped_completed"])
                self.assertEqual(scorer.call_count, 4)

                source_run.joinpath("audio", f"{jobs[0].id}.wav").write_bytes(b"changed" * 200)
                third = rescore_one(
                    source_root,
                    output_root,
                    "test",
                    overwrite=False,
                    resume=True,
                    dry_run=False,
                )
                self.assertEqual(third["resume_reused_rows"], 0)
                self.assertEqual(third["resume_rescored_rows"], 2)
                self.assertEqual(scored_batches[4:], [[job.id for job in jobs]] * 4)
                replaced = json.loads(out_run.joinpath("manifest.json").read_text(encoding="utf-8"))
                self.assertNotIn("reuse_marker", replaced["samples"][0])


if __name__ == "__main__":
    unittest.main()
