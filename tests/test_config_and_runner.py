from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from crosslingual_tts_lab.backends import create_backend
from crosslingual_tts_lab.backends.external import ExternalCommandBackend
from crosslingual_tts_lab.config import load_config
from crosslingual_tts_lab.config import ModelSpec, TargetSpec, VoiceSpec
from crosslingual_tts_lab.open_datasets import (
    _fleurs_dataset_code,
    _normalize_text,
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
                )


if __name__ == "__main__":
    unittest.main()
