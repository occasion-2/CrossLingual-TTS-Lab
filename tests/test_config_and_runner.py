from __future__ import annotations

import json
import sys
import tarfile
import tempfile
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
            {"coqui_xtts", "f5_tts", "qwen_tts", "external_command"},
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
