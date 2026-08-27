from __future__ import annotations

import json
import sys
import tarfile
import tempfile
import tomllib
import unittest
from io import BytesIO
from pathlib import Path

from crosslingual_tts_lab.backends import create_backend
from crosslingual_tts_lab.backends.coqui_xtts import CoquiXTTSBackend
from crosslingual_tts_lab.backends.external import ExternalCommandBackend
from crosslingual_tts_lab.common_voice_mdc import (
    extract_common_voice_slice,
    load_env_file,
    parse_dataset_ids,
    parse_locale_filters,
    _total_size_from_content_range,
)
from crosslingual_tts_lab.cli import _calibrate, _parse_key_values
from crosslingual_tts_lab.config import load_config
from crosslingual_tts_lab.config import (
    BenchmarkConfig,
    MetricSpec,
    ModelSpec,
    PairSpec,
    TargetSpec,
    VoiceSpec,
)
from crosslingual_tts_lab.device import DeviceProfile
from crosslingual_tts_lab.metrics import create_metrics
from crosslingual_tts_lab.open_datasets import (
    LanguageRequest,
    build_local_common_voice_config,
    _fleurs_dataset_code,
    _normalize_text,
    _real_metric_specs,
    _select_speaker_voice_rows,
    _select_target_rows,
    parse_language_requests,
    render_benchmark_toml,
)
from crosslingual_tts_lab.planner import GenerationJob
from crosslingual_tts_lab.planner import plan_jobs
from crosslingual_tts_lab.runner import run_benchmark, score_existing_run


class ConfigAndRunnerTests(unittest.TestCase):
    def test_mini_config_plans_jobs(self) -> None:
        config = load_config(Path("configs/mini.toml"))

        jobs = plan_jobs(config)

        self.assertEqual(len(jobs), 3)
        self.assertEqual({job.direction for job in jobs}, {"ru->en", "ru->zh", "en->ru"})
        self.assertTrue(all(job.is_cross_lingual for job in jobs))

    def test_plan_jobs_sanitizes_path_unsafe_model_ids(self) -> None:
        config = BenchmarkConfig(
            name="unsafe-id-smoke",
            description=None,
            models=[ModelSpec(id="Qwen/Qwen3-TTS-12Hz-1.7B-Base", backend="qwen_tts")],
            voices=[
                VoiceSpec(
                    id="voice",
                    language="ru",
                    speaker_id="speaker",
                    audio_path=Path("/tmp/reference.wav"),
                )
            ],
            targets=[TargetSpec(id="target", language="en", text="hello")],
            pairs=[PairSpec(voice="voice", target="target")],
            metrics=[],
            root=Path("."),
        )

        job = plan_jobs(config)[0]

        self.assertNotIn("/", job.id)
        self.assertTrue(job.id.startswith("Qwen_Qwen3-TTS-12Hz-1.7B-Base_"))

    def test_run_benchmark_writes_manifest_report_and_audio(self) -> None:
        config = load_config(Path("configs/mini.toml"))

        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = run_benchmark(config, Path(tmp) / "run")

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["summary"]["jobs"], 3)
            self.assertEqual(manifest["summary"]["cross_lingual_jobs"], 3)
            self.assertEqual(
                manifest["reproducibility"]["model_specs"][0]["backend"],
                "dummy",
            )
            self.assertEqual(
                manifest["reproducibility"]["configuration_role"],
                "requested_runtime_configuration",
            )
            self.assertIn("python_version", manifest["reproducibility"])
            self.assertEqual(manifest["samples"][0]["model"]["params"], {})
            self.assertEqual(
                manifest["samples"][0]["synthesis_metadata"]["synthesis_provenance"],
                "generated_current_config",
            )
            self.assertIn(
                "model_config_fingerprint",
                manifest["samples"][0]["synthesis_metadata"],
            )
            self.assertTrue(manifest_path.with_name("report.md").exists())
            for sample in manifest["samples"]:
                self.assertTrue(sample["metrics"])
                self.assertEqual(
                    {metric["status"] for metric in sample["metrics"]},
                    {"missing_backend"},
                )
                self.assertTrue(Path(sample["audio_path"]).exists())

    def test_score_existing_run_reuses_audio(self) -> None:
        config = load_config(Path("configs/mini.toml"))

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_benchmark(config, run_dir)
            manifest_path = score_existing_run(config, run_dir)

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["summary"]["jobs"], 3)
            self.assertTrue(manifest_path.with_name("report.md").exists())
            self.assertEqual(
                {sample["synthesis_metadata"]["backend"] for sample in manifest["samples"]},
                {"dummy"},
            )
            self.assertEqual(
                {
                    sample["synthesis_metadata"]["synthetic_placeholder"]
                    for sample in manifest["samples"]
                },
                {True},
            )

    def test_parse_dataset_language_requests(self) -> None:
        requests = parse_language_requests("ru:ru,en:en,zh-CN:zh")

        self.assertEqual(
            [(item.dataset_code, item.benchmark_code) for item in requests],
            [("ru", "ru"), ("en", "en"), ("zh-CN", "zh")],
        )

    def test_fleurs_language_aliases(self) -> None:
        self.assertEqual(_fleurs_dataset_code("ru"), "ru_ru")
        self.assertEqual(_fleurs_dataset_code("en"), "en_us")
        self.assertEqual(_fleurs_dataset_code("zh-CN"), "cmn_hans_cn")

    def test_fleurs_text_normalization_removes_cjk_spaces(self) -> None:
        self.assertEqual(_normalize_text("zh", "亚 马 逊 河"), "亚马逊河")
        self.assertEqual(_normalize_text("en", "hello   world"), "hello world")

    def test_target_selection_can_skip_too_short_text(self) -> None:
        rows = [
            {"sentence": "六"},
            {"sentence": "此时必须通报警察才能解除闪红灯。"},
            {"sentence": "hello"},
        ]

        selected = _select_target_rows(rows, limit=2, language="zh", min_chars=4)

        self.assertEqual([row["sentence"] for row in selected], ["此时必须通报警察才能解除闪红灯。", "hello"])

    def test_run_benchmark_records_synthesis_failures_without_aborting(self) -> None:
        text = render_benchmark_toml(
            name="synthesis-failure-smoke",
            description="backend failure should be non-fatal",
            models=[
                {
                    "id": "failing",
                    "backend": "external_command",
                    "params": {"command": f"{sys.executable} -c \"import sys; sys.exit(7)\""},
                }
            ],
            metrics=[{"id": "placeholder", "backend": "placeholder"}],
            voices=[
                {
                    "id": "voice",
                    "language": "ru",
                    "speaker_id": "speaker",
                    "audio_path": "/tmp/ref.wav",
                }
            ],
            targets=[{"id": "target", "language": "zh", "text": "六"}],
            pairs=[{"voice": "voice", "target": "target"}],
        )

        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.toml"
            config_path.write_text(text, encoding="utf-8")
            manifest_path = run_benchmark(load_config(config_path), Path(tmp) / "run")

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        sample = manifest["samples"][0]
        self.assertTrue(sample["synthesis_metadata"]["synthesis_failed"])
        self.assertEqual({metric["status"] for metric in sample["metrics"]}, {"synthesis_failed"})

    def test_common_voice_speaker_selection_keeps_repeated_known_speakers(self) -> None:
        rows = [
            {"client_id": "speaker-a", "sentence": "first", "audio": {"path": "/tmp/a1.wav"}},
            {"client_id": "speaker-b", "sentence": "only one", "audio": {"path": "/tmp/b1.wav"}},
            {"client_id": "speaker-a", "sentence": "second", "audio": {"path": "/tmp/a2.wav"}},
            {"client_id": "speaker-c", "sentence": "third", "audio": {"path": "/tmp/c1.wav"}},
            {"client_id": "speaker-c", "sentence": "fourth", "audio": {"path": "/tmp/c2.wav"}},
        ]

        selected = _select_speaker_voice_rows(
            rows,
            speakers_limit=2,
            utterances_per_speaker=2,
            language="en",
        )

        self.assertEqual([row["client_id"] for row in selected], ["speaker-a", "speaker-a", "speaker-c", "speaker-c"])

    def test_local_common_voice_config_uses_known_repeated_speaker_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "cv"
            clips = root / "en" / "clips"
            clips.mkdir(parents=True)
            for name in ["a1.mp3", "a2.mp3", "b1.mp3"]:
                (clips / name).write_bytes(b"fake mp3")
            (root / "en" / "validated.tsv").write_text(
                "\n".join(
                    [
                        "client_id\tpath\tsentence",
                        "speaker-a\ta1.mp3\tfirst sentence",
                        "speaker-a\ta2.mp3\tsecond sentence",
                        "speaker-b\tb1.mp3\tthird sentence",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            out_path = Path(tmp) / "config.toml"
            build_local_common_voice_config(
                out_path=out_path,
                local_root=root,
                languages=[LanguageRequest("en", "en"), LanguageRequest("en", "en2")],
                split="validated",
                voices_per_language=1,
                utterances_per_speaker=2,
                targets_per_language=1,
                model_id="dummy_tts",
                model_backend="dummy",
                model_params={},
                include_mono_lingual=False,
            )

            config = load_config(out_path)

        self.assertEqual([voice.speaker_id for voice in config.voices[:2]], ["speaker-a", "speaker-a"])
        self.assertTrue(config.pairs)

    def test_common_voice_mdc_env_parser_accepts_spaced_assignments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text(
                "COMMONVOICE_APIKEY = 'secret-value'\nOTHER=value\n",
                encoding="utf-8",
            )

            values = load_env_file(env_path)

        self.assertEqual(values["COMMONVOICE_APIKEY"], "secret-value")
        self.assertEqual(values["OTHER"], "value")

    def test_common_voice_mdc_dataset_id_overrides_defaults(self) -> None:
        ids = parse_dataset_ids("en=english-id,zh=zh-id")

        self.assertEqual(ids["en"], "english-id")
        self.assertEqual(ids["zh-CN"], "zh-id")
        self.assertIn("ru", ids)

    def test_common_voice_mdc_parses_accent_filters(self) -> None:
        filters = parse_locale_filters("en=United States English|England English,zh=Mandarin")

        self.assertEqual(filters["en"], {"united states english", "england english"})
        self.assertEqual(filters["zh-CN"], {"mandarin"})

    def test_common_voice_mdc_parses_ranged_total_size(self) -> None:
        self.assertEqual(_total_size_from_content_range("bytes 0-0/94639372950"), 94_639_372_950)
        self.assertIsNone(_total_size_from_content_range(""))

    def test_common_voice_mdc_extracts_selected_rows_and_clips(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            archive = BytesIO()
            with tarfile.open(fileobj=archive, mode="w:gz") as tar:
                tsv = (
                    "client_id\tpath\tsentence\n"
                    "speaker-a\ta1.mp3\tfirst sentence\n"
                    "speaker-a\ta2.mp3\tsecond sentence\n"
                    "speaker-b\tb1.mp3\tthird sentence\n"
                    "speaker-c\tc1.mp3\tfourth sentence\n"
                ).encode("utf-8")
                info = tarfile.TarInfo("cv-corpus/en/validated.tsv")
                info.size = len(tsv)
                tar.addfile(info, BytesIO(tsv))
                for name in ["a1.mp3", "a2.mp3", "b1.mp3", "c1.mp3"]:
                    payload = f"fake {name}".encode("utf-8")
                    info = tarfile.TarInfo(f"cv-corpus/en/clips/{name}")
                    info.size = len(payload)
                    tar.addfile(info, BytesIO(payload))

            archive.seek(0)
            result = extract_common_voice_slice(
                archive,
                out_root=Path(tmp) / "cv",
                locale="en",
                split="validated",
                speakers_per_language=1,
                utterances_per_speaker=2,
                targets_per_language=1,
                benchmark_language="en",
            )

            split_path = Path(result["split_path"])
            self.assertTrue(split_path.exists())
            self.assertEqual(result["clips"], 2)
            self.assertTrue((split_path.parent / "clips" / "a1.mp3").exists())
            self.assertTrue((split_path.parent / "clips" / "a2.mp3").exists())

    def test_common_voice_mdc_accent_filter_excludes_non_native_english_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            archive = BytesIO()
            with tarfile.open(fileobj=archive, mode="w:gz") as tar:
                tsv = (
                    "client_id\tpath\tsentence\taccents\n"
                    "speaker-a\ta1.mp3\tfirst sentence\tNepalese\n"
                    "speaker-a\ta2.mp3\tsecond sentence\tNepalese\n"
                    "speaker-b\tb1.mp3\tthird sentence\tUnited States English\n"
                    "speaker-b\tb2.mp3\tfourth sentence\tUnited States English\n"
                ).encode("utf-8")
                info = tarfile.TarInfo("cv-corpus/en/validated.tsv")
                info.size = len(tsv)
                tar.addfile(info, BytesIO(tsv))
                for name in ["a1.mp3", "a2.mp3", "b1.mp3", "b2.mp3"]:
                    payload = f"fake {name}".encode("utf-8")
                    info = tarfile.TarInfo(f"cv-corpus/en/clips/{name}")
                    info.size = len(payload)
                    tar.addfile(info, BytesIO(payload))

            archive.seek(0)
            result = extract_common_voice_slice(
                archive,
                out_root=Path(tmp) / "cv",
                locale="en",
                split="validated",
                speakers_per_language=1,
                utterances_per_speaker=2,
                targets_per_language=1,
                benchmark_language="en",
                accent_filter={"united states english"},
            )

            split_path = Path(result["split_path"])
            tsv_text = split_path.read_text(encoding="utf-8")
            self.assertNotIn("Nepalese", tsv_text)
            self.assertIn("United States English", tsv_text)
            self.assertTrue((split_path.parent / "clips" / "b1.mp3").exists())
            self.assertTrue((split_path.parent / "clips" / "b2.mp3").exists())

    def test_render_open_dataset_config_loads(self) -> None:
        text = render_benchmark_toml(
            name="open-data-smoke",
            description="config generated from open data",
            models=[{"id": "dummy_tts", "backend": "dummy"}],
            metrics=[
                {
                    "id": "asr_error",
                    "backend": "faster_whisper_asr",
                    "params": {"vad_filter": True},
                }
            ],
            voices=[
                {
                    "id": "v_ru",
                    "language": "ru",
                    "speaker_id": "speaker-1",
                    "audio_path": "/tmp/ref.wav",
                    "transcript": "privet",
                }
            ],
            targets=[
                {
                    "id": "t_en",
                    "language": "en",
                    "text": "hello",
                }
            ],
            pairs=[{"voice": "v_ru", "target": "t_en"}],
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "generated.toml"
            path.write_text(text, encoding="utf-8")
            config = load_config(path)

        self.assertEqual(config.metrics[0].backend, "faster_whisper_asr")
        self.assertEqual(config.voices[0].language, "ru")

    def test_render_config_includes_model_params(self) -> None:
        text = render_benchmark_toml(
            name="model-param-smoke",
            description="model params",
            models=[
                {
                    "id": "f5",
                    "backend": "f5_tts",
                    "params": {"model": "F5TTS_v1_Base", "ref_text_mode": "transcript"},
                }
            ],
            metrics=[],
            voices=[
                {
                    "id": "v",
                    "language": "ru",
                    "speaker_id": "speaker",
                    "audio_path": "/tmp/ref.wav",
                }
            ],
            targets=[{"id": "t", "language": "en", "text": "hello"}],
            pairs=[{"voice": "v", "target": "t"}],
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model_params.toml"
            path.write_text(text, encoding="utf-8")
            config = load_config(path)

        self.assertEqual(config.models[0].params["model"], "F5TTS_v1_Base")

    def test_config_rejects_qwen_backend_with_f5_model_param(self) -> None:
        text = render_benchmark_toml(
            name="bad-qwen-config",
            description="backend/model mismatch",
            models=[
                {
                    "id": "qwen",
                    "backend": "qwen_tts",
                    "params": {"model": "F5TTS_v1_Base"},
                }
            ],
            metrics=[],
            voices=[
                {
                    "id": "v",
                    "language": "ru",
                    "speaker_id": "speaker",
                    "audio_path": "/tmp/ref.wav",
                }
            ],
            targets=[{"id": "t", "language": "en", "text": "hello"}],
            pairs=[{"voice": "v", "target": "t"}],
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad_qwen.toml"
            path.write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "backend 'qwen_tts'.*F5TTS_v1_Base"):
                load_config(path)

    def test_real_model_template_loads(self) -> None:
        config = load_config(Path("configs/models_real.template.toml"))

        self.assertEqual(
            {model.backend for model in config.models},
            {
                "coqui_xtts",
                "f5_tts",
                "qwen_tts",
                "cosyvoice",
                "spark_tts",
                "external_command",
            },
        )
        self.assertEqual(len(config.metrics), 4)
        self.assertTrue(all(metric.params.get("model_revision") for metric in config.metrics))

    def test_generated_dataset_metrics_pin_evaluator_revisions(self) -> None:
        metrics = _real_metric_specs()
        by_backend = {metric["backend"]: metric["params"] for metric in metrics}

        self.assertEqual(
            by_backend["faster_whisper_asr"]["model_revision"],
            "08e178d48790749d25932bbc082711ddcfdfbc4f",
        )
        self.assertEqual(
            by_backend["faster_whisper_asr"]["cpu_model_revision"],
            "536b0662742c02347bc0e980a01041f333bce120",
        )
        self.assertEqual(
            by_backend["speechbrain_speaker_similarity"]["model_revision"],
            "0f99f2d0ebe89ac095bcc5903c4dd8f72b367286",
        )
        self.assertEqual(
            by_backend["speechbrain_language_similarity"]["model_revision"],
            "0253049ae131d6a4be1c4f0d8b0ff483a0f8c8e9",
        )

    def test_metric_revision_selection_supports_cpu_fallback(self) -> None:
        from crosslingual_tts_lab.metrics.asr import FasterWhisperASRMetric
        from crosslingual_tts_lab.metrics.leakage import SpeechBrainLanguageSimilarityMetric
        from crosslingual_tts_lab.metrics.speaker import SpeechBrainSpeakerSimilarityMetric

        profile = DeviceProfile(device="cuda", cuda_available=True)
        asr = FasterWhisperASRMetric(
            "asr",
            {"model_revision": "medium-rev", "cpu_model_revision": "small-rev"},
            profile,
        )
        self.assertEqual(asr._model_revision(), "medium-rev")
        asr._forced_model_size = "small"
        self.assertEqual(asr._model_revision(), "small-rev")
        cpu_asr = FasterWhisperASRMetric(
            "asr",
            {
                "model_size": "medium",
                "model_revision": "medium-rev",
                "cpu_model_size": "small",
                "cpu_model_revision": "small-rev",
                "cpu_compute_type": "int8",
            },
            DeviceProfile(device="cpu", cuda_available=False),
        )
        self.assertEqual(cpu_asr._model_size(), "small")
        self.assertEqual(cpu_asr._model_revision(), "small-rev")
        self.assertEqual(cpu_asr._compute_type(), "int8")
        self.assertEqual(
            SpeechBrainSpeakerSimilarityMetric(
                "speaker", {"model_revision": "speaker-rev"}, profile
            )._model_revision(),
            "speaker-rev",
        )
        self.assertEqual(
            SpeechBrainLanguageSimilarityMetric(
                "leakage", {"revision": "language-rev"}, profile
            )._model_revision(),
            "language-rev",
        )

    def test_calibration_forwards_pinned_speaker_evaluator(self) -> None:
        from unittest.mock import patch

        with patch("crosslingual_tts_lab.cli.compute_calibration") as compute:
            result = _calibrate(
                Path("/tmp/calibration-run"),
                "speechbrain/spkrec-ecapa-voxceleb",
                "immutable-revision",
                "cuda:0",
            )

        self.assertEqual(result, 0)
        compute.assert_called_once_with(
            Path("/tmp/calibration-run"),
            model_id="speechbrain/spkrec-ecapa-voxceleb",
            model_revision="immutable-revision",
            device="cuda:0",
        )

    def test_paper_snapshot_covers_models_and_evaluators(self) -> None:
        snapshot = tomllib.loads(
            Path("configs/paper_model_snapshot.toml").read_text(encoding="utf-8")
        )

        self.assertEqual(len(snapshot["models"]), 6)
        self.assertEqual(len(snapshot["evaluators"]), 3)
        self.assertTrue(
            all(item["provenance_status"] == "reconstructed" for item in snapshot["models"])
        )
        self.assertTrue(
            all(
                item["provenance_status"] == "run-attested-rescore"
                for item in snapshot["evaluators"]
            )
        )
        self.assertEqual(
            snapshot["evaluator_rescore"]["profile_id"],
            "paper-evaluators-medium-cuda-fp16-v1",
        )
        self.assertIs(snapshot["evaluator_rescore"]["cpu_fallback"], False)
        self.assertEqual(
            set(snapshot["evaluator_rescore"]["runs"]),
            {"f5tts", "cosyvoice", "qwen0_6b", "qwen1_7b", "spark_tts", "xtts"},
        )
        self.assertTrue(
            all(
                len(run["result_manifest_sha256"]) == 64
                for run in snapshot["evaluator_rescore"]["runs"].values()
            )
        )
        f5 = next(item for item in snapshot["models"] if item["id"] == "f5tts_v1_base")
        self.assertEqual(f5["documented_target_languages"], ["en", "zh"])
        self.assertEqual(f5["reference_asr_checkpoint"], "openai/whisper-large-v3-turbo")
        leakage = next(
            item
            for item in snapshot["evaluators"]
            if item["id"] == "language_centroid_leakage_proxy"
        )
        self.assertEqual(
            leakage["centroid_sha256"],
            "9adca9f8ef996c21f5f3473359bfc7e3b0d852d19139b0af5049621514373d91",
        )

    def test_model_param_parser_preserves_scalar_types(self) -> None:
        self.assertEqual(
            _parse_key_values(
                ["enabled=true", "disabled=false", "steps=32", "temperature=0.8", "model=repo/id"]
            ),
            {
                "enabled": True,
                "disabled": False,
                "steps": 32,
                "temperature": 0.8,
                "model": "repo/id",
            },
        )

    def test_backend_registry_has_nondummy_backends(self) -> None:
        self.assertEqual(create_backend("f5_tts").name, "f5_tts")
        self.assertEqual(create_backend("coqui_xtts").name, "coqui_xtts")
        self.assertEqual(create_backend("qwen_tts").name, "qwen_tts")
        self.assertEqual(create_backend("qwentts").name, "qwen_tts")
        self.assertIsInstance(create_backend("external_command"), ExternalCommandBackend)

    def test_backend_registry_resolves_aliases(self) -> None:
        self.assertIsInstance(create_backend("xtts"), CoquiXTTSBackend)
        self.assertEqual(create_backend("f5").name, "f5_tts")
        self.assertEqual(create_backend("qwen3_tts").name, "qwen_tts")
        self.assertIsInstance(create_backend("cli"), ExternalCommandBackend)

    def test_metric_registry_expands_configured_metrics(self) -> None:
        profile = DeviceProfile(device="cpu", cuda_available=False)

        metrics = create_metrics(
            [
                MetricSpec(id="asr", backend="faster_whisper_asr"),
                MetricSpec(id="lid", backend="faster_whisper_lid"),
                MetricSpec(id="placeholders", backend="placeholder"),
            ],
            profile,
        )

        self.assertEqual(metrics[0].name, "asr")
        self.assertEqual(metrics[1].name, "lid")
        self.assertIn("source_language_leakage_proxy", {metric.name for metric in metrics})

    def test_external_command_backend_creates_expected_audio(self) -> None:
        backend = ExternalCommandBackend(
            {
                "command": [
                    sys.executable,
                    "-c",
                    "from pathlib import Path; import sys; Path(sys.argv[1]).write_bytes(b'RIFF')",
                    "{audio_path}",
                ]
            }
        )
        job = GenerationJob(
            id="external_smoke",
            model=ModelSpec(id="external", backend="external_command"),
            voice=VoiceSpec(
                id="voice",
                language="ru",
                speaker_id="speaker",
                audio_path=Path("/tmp/reference.wav"),
                transcript="privet",
            ),
            target=TargetSpec(id="target", language="en", text="hello"),
        )

        with tempfile.TemporaryDirectory() as tmp:
            result = backend.synthesize(job, Path(tmp))

            self.assertTrue(result.audio_path.exists())
            self.assertEqual(result.audio_path.read_bytes(), b"RIFF")

    def test_external_command_backend_string_format_safety(self) -> None:
        backend = ExternalCommandBackend(
            {
                "command": f"{sys.executable} -c \"from pathlib import Path; import sys; Path(sys.argv[1]).write_bytes(sys.argv[2].encode('utf-8'))\" {{audio_path}} \"{{target_text}}\""
            }
        )
        job = GenerationJob(
            id="external_smoke_string",
            model=ModelSpec(id="external", backend="external_command"),
            voice=VoiceSpec(
                id="voice",
                language="ru",
                speaker_id="speaker",
                audio_path=Path("/tmp/reference.wav"),
                transcript="privet",
            ),
            target=TargetSpec(id="target", language="en", text="hello world"),
        )

        with tempfile.TemporaryDirectory() as tmp:
            result = backend.synthesize(job, Path(tmp))

            self.assertTrue(result.audio_path.exists())
            self.assertEqual(result.audio_path.read_bytes(), b"hello world")

    def test_qwen_tts_backend_synthesizes_using_mock(self) -> None:
        from unittest.mock import MagicMock, patch
        import numpy as np
        from crosslingual_tts_lab.backends.qwen_tts import QwenTTSBackend

        backend = QwenTTSBackend()
        mock_model = MagicMock()
        mock_model.generate_voice_clone.return_value = (np.zeros(16000), 16000)

        with tempfile.TemporaryDirectory() as tmp:
            ref_path = Path(tmp) / "reference.wav"
            ref_path.write_bytes(b"mock audio")

            job = GenerationJob(
                id="qwen_smoke",
                model=ModelSpec(id="qwen", backend="qwen_tts"),
                voice=VoiceSpec(
                    id="voice",
                    language="ru",
                    speaker_id="speaker",
                    audio_path=ref_path,
                    transcript="privet",
                ),
                target=TargetSpec(id="target", language="en", text="hello"),
            )

            with patch.object(backend, "_load_model", return_value=mock_model):
                result = backend.synthesize(job, Path(tmp))

                self.assertTrue(result.audio_path.exists())
                mock_model.generate_voice_clone.assert_called_once_with(
                    text="hello",
                    language="English",
                    ref_audio=str(ref_path),
                    ref_text="privet",
                    x_vector_only_mode=False,
                )

    def test_qwen_tts_backend_uses_x_vector_mode_without_reference_text(self) -> None:
        from unittest.mock import MagicMock, patch
        import numpy as np
        from crosslingual_tts_lab.backends.qwen_tts import QwenTTSBackend

        backend = QwenTTSBackend({"ref_text_mode": "empty"})
        mock_model = MagicMock()
        mock_model.generate_voice_clone.return_value = (np.zeros(16000), 16000)

        with tempfile.TemporaryDirectory() as tmp:
            ref_path = Path(tmp) / "reference.wav"
            ref_path.write_bytes(b"mock audio")
            job = GenerationJob(
                id="qwen_empty_ref_text",
                model=ModelSpec(id="qwen", backend="qwen_tts"),
                voice=VoiceSpec(
                    id="voice",
                    language="ru",
                    speaker_id="speaker",
                    audio_path=ref_path,
                    transcript="privet",
                ),
                target=TargetSpec(id="target", language="en", text="hello"),
            )

            with patch.object(backend, "_load_model", return_value=mock_model):
                result = backend.synthesize(job, Path(tmp))

                self.assertTrue(result.audio_path.exists())
                mock_model.generate_voice_clone.assert_called_once_with(
                    text="hello",
                    language="English",
                    ref_audio=str(ref_path),
                    ref_text="",
                    x_vector_only_mode=True,
                )

    def test_qwen_revision_resolves_complete_hub_snapshot(self) -> None:
        from unittest.mock import patch
        from crosslingual_tts_lab.backends.qwen_tts import QwenTTSBackend

        backend = QwenTTSBackend({"model": "org/model", "revision": "immutable"})
        with patch("huggingface_hub.snapshot_download", return_value="/cache/snapshot") as download:
            self.assertEqual(backend._resolve_model_source(), "/cache/snapshot")
            self.assertEqual(backend._resolve_model_source(), "/cache/snapshot")

        download.assert_called_once_with(
            repo_id="org/model",
            revision="immutable",
            cache_dir=None,
        )

    def test_f5_revision_resolves_checkpoint_file(self) -> None:
        from unittest.mock import patch
        from crosslingual_tts_lab.backends.f5_tts import F5TTSBackend

        backend = F5TTSBackend({"checkpoint_revision": "immutable"})
        with patch("huggingface_hub.hf_hub_download", return_value="/cache/model.safetensors") as download:
            self.assertEqual(backend._resolve_checkpoint_file(), "/cache/model.safetensors")
            self.assertEqual(backend._resolve_checkpoint_file(), "/cache/model.safetensors")

        download.assert_called_once_with(
            repo_id="SWivid/F5-TTS",
            filename="F5TTS_v1_Base/model_1250000.safetensors",
            revision="immutable",
            cache_dir=None,
        )

    def test_f5_revision_resolves_complete_vocoder_snapshot(self) -> None:
        from unittest.mock import patch
        from crosslingual_tts_lab.backends.f5_tts import F5TTSBackend

        backend = F5TTSBackend({"vocoder_revision": "immutable"})
        with patch("huggingface_hub.snapshot_download", return_value="/cache/vocoder") as download:
            self.assertEqual(backend._resolve_vocoder_path(), "/cache/vocoder")
            self.assertEqual(backend._resolve_vocoder_path(), "/cache/vocoder")

        download.assert_called_once_with(
            repo_id="charactr/vocos-mel-24khz",
            revision="immutable",
            cache_dir=None,
        )

    def test_cosyvoice3_load_kwargs_omit_unsupported_load_jit(self) -> None:
        from crosslingual_tts_lab.backends.cosyvoice import CosyVoiceBackend

        backend = CosyVoiceBackend({"load_jit": True, "load_vllm": False})
        kwargs = backend._model_load_kwargs("Fun-CosyVoice3-0.5B", "/cache/model")

        self.assertNotIn("load_jit", kwargs)
        self.assertIn("load_vllm", kwargs)

    def test_cosyvoice3_family_is_detected_from_local_checkpoint(self) -> None:
        from crosslingual_tts_lab.backends.cosyvoice import CosyVoiceBackend

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "local-checkpoint"
            checkpoint.mkdir()
            (checkpoint / "cosyvoice3.yaml").write_text("sample_rate: 24000\n")

            backend = CosyVoiceBackend({"load_jit": True, "load_vllm": False})
            kwargs = backend._model_load_kwargs("local-checkpoint", str(checkpoint))

            self.assertEqual(
                backend._expected_model_class("local-checkpoint", str(checkpoint)),
                "CosyVoice3",
            )

        self.assertNotIn("load_jit", kwargs)
        self.assertIn("load_vllm", kwargs)

    def test_xtts_artifact_hash_verification(self) -> None:
        import hashlib
        from types import SimpleNamespace
        from crosslingual_tts_lab.backends.coqui_xtts import CoquiXTTSBackend

        payloads = {
            "model.pth": b"model",
            "config.json": b"config",
            "vocab.json": b"vocab",
            "speakers_xtts.pth": b"speakers",
        }
        params = {
            "model_name": "tts_models/multilingual/multi-dataset/xtts_v2",
            "model_sha256": hashlib.sha256(payloads["model.pth"]).hexdigest(),
            "config_sha256": hashlib.sha256(payloads["config.json"]).hexdigest(),
            "vocab_sha256": hashlib.sha256(payloads["vocab.json"]).hexdigest(),
            "speakers_sha256": hashlib.sha256(payloads["speakers_xtts.pth"]).hexdigest(),
        }
        with tempfile.TemporaryDirectory() as tmp:
            artifact_dir = Path(tmp) / "tts_models--multilingual--multi-dataset--xtts_v2"
            artifact_dir.mkdir()
            for filename, payload in payloads.items():
                (artifact_dir / filename).write_bytes(payload)
            backend = CoquiXTTSBackend(params)
            backend._verify_model_artifacts(
                SimpleNamespace(manager=SimpleNamespace(output_prefix=tmp))
            )

        self.assertTrue(backend._artifact_hashes_verified)

    def test_cosyvoice_backend_synthesizes_using_mock(self) -> None:
        import sys
        from unittest.mock import MagicMock, patch
        import numpy as np

        # Create mock module structure for cosyvoice
        mock_file_utils = MagicMock()
        mock_file_utils.load_wav.return_value = np.zeros(16000)
        sys.modules["cosyvoice"] = MagicMock()
        sys.modules["cosyvoice.utils"] = MagicMock()
        sys.modules["cosyvoice.utils.file_utils"] = mock_file_utils

        try:
            from crosslingual_tts_lab.backends.cosyvoice import CosyVoiceBackend

            backend = CosyVoiceBackend()
            mock_model = MagicMock()
            mock_model.inference_zero_shot.return_value = [{"tts_speech": np.zeros(16000)}]

            with tempfile.TemporaryDirectory() as tmp:
                ref_path = Path(tmp) / "reference.wav"
                ref_path.write_bytes(b"mock audio")

                job = GenerationJob(
                    id="cosyvoice_smoke",
                    model=ModelSpec(id="cosy", backend="cosyvoice"),
                    voice=VoiceSpec(
                        id="voice",
                        language="ru",
                        speaker_id="speaker",
                        audio_path=ref_path,
                        transcript="privet",
                    ),
                    target=TargetSpec(id="target", language="en", text="hello"),
                )

                with patch.object(backend, "_load_model", return_value=mock_model):
                    result = backend.synthesize(job, Path(tmp))

                    self.assertTrue(result.audio_path.exists())
                    mock_model.inference_zero_shot.assert_called_once()
        finally:
            sys.modules.pop("cosyvoice", None)
            sys.modules.pop("cosyvoice.utils", None)
            sys.modules.pop("cosyvoice.utils.file_utils", None)

    def test_spark_tts_backend_synthesizes_using_mock(self) -> None:
        from unittest.mock import patch
        import numpy as np
        from crosslingual_tts_lab.backends.spark_tts import SparkTTSBackend

        class FakeSparkModel:
            def __init__(self) -> None:
                self.calls: list[dict[str, object]] = []

            def inference(
                self,
                text: str,
                prompt_speech_path: str | None = None,
                prompt_text: str | None = None,
                gender: str | None = None,
                pitch: str | None = None,
                speed: str | None = None,
                temperature: float = 0.8,
                top_k: float = 50,
                top_p: float = 0.95,
            ):
                self.calls.append(
                    {
                        "text": text,
                        "prompt_speech_path": prompt_speech_path,
                        "prompt_text": prompt_text,
                        "gender": gender,
                        "pitch": pitch,
                        "speed": speed,
                        "temperature": temperature,
                        "top_k": top_k,
                        "top_p": top_p,
                    }
                )
                return np.zeros(16000)

        backend = SparkTTSBackend(params={"seed": 1234})
        fake_model = FakeSparkModel()

        with tempfile.TemporaryDirectory() as tmp:
            ref_path = Path(tmp) / "reference.wav"
            ref_path.write_bytes(b"mock audio")

            job = GenerationJob(
                id="spark_smoke",
                model=ModelSpec(id="spark", backend="spark_tts"),
                voice=VoiceSpec(
                    id="voice",
                    language="ru",
                    speaker_id="speaker",
                    audio_path=ref_path,
                    transcript="privet",
                ),
                target=TargetSpec(id="target", language="en", text="hello"),
            )

            with patch.object(backend, "_load_model", return_value=fake_model):
                result = backend.synthesize(job, Path(tmp))

                self.assertTrue(result.audio_path.exists())
                self.assertEqual(
                    fake_model.calls,
                    [
                        {
                            "text": "hello",
                            "prompt_speech_path": str(ref_path),
                            "prompt_text": "privet",
                            "gender": None,
                            "pitch": None,
                            "speed": None,
                            "temperature": 0.8,
                            "top_k": 50,
                            "top_p": 0.95,
                        }
                    ],
                )
                self.assertEqual(
                    result.metadata["inference_config"]["seed_status"],
                    "uncontrolled",
                )
                self.assertIsNone(result.metadata["inference_config"]["seed"])

    def test_spark_tts_verifies_local_huggingface_revision(self) -> None:
        from crosslingual_tts_lab.backends.spark_tts import SparkTTSBackend

        revision = "immutable"
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            for relative in (
                ".cache/huggingface/download/LLM/model.safetensors.metadata",
                ".cache/huggingface/download/BiCodec/model.safetensors.metadata",
            ):
                metadata = model_dir / relative
                metadata.parent.mkdir(parents=True, exist_ok=True)
                metadata.write_text(f"{revision}\nweight-hash\n", encoding="utf-8")
            backend = SparkTTSBackend(
                {"model_name": str(model_dir), "revision": revision}
            )
            backend._verify_local_checkpoint_revision()

        self.assertTrue(backend._checkpoint_revision_verified)

    def test_asr_adapters_normalize_correctly(self) -> None:
        from crosslingual_tts_lab.text_metrics import get_asr_adapter

        # English
        en_adapter = get_asr_adapter("en-US")
        self.assertEqual(en_adapter.normalize("Hello, World! It's nice."), "hello world it's nice")

        # Russian
        ru_adapter = get_asr_adapter("ru_RU")
        self.assertEqual(ru_adapter.normalize("Привет, Мир! Всё хорошо."), "привет мир все хорошо")

        # Chinese
        zh_adapter = get_asr_adapter("zh-CN")
        self.assertEqual(zh_adapter.normalize("亚马逊河 也是 地球 上！"), "亚马逊河也是地球上")

        # Chinese/Mandarin with cmn prefix (FLEURS)
        cmn_adapter = get_asr_adapter("cmn_hans_cn")
        self.assertEqual(cmn_adapter.normalize("报告警告 称，"), "报告警告称")

        # Default fallback
        default_adapter = get_asr_adapter("unknown_lang")
        self.assertEqual(default_adapter.normalize("Hello, World!"), "hello world")


if __name__ == "__main__":
    unittest.main()
