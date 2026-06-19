from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from crosslingual_tts_lab.backends.base import SynthesisResult
from crosslingual_tts_lab.planner import GenerationJob


@dataclass
class ExternalCommandBackend:
    """Run an installed model CLI using formatted job placeholders."""

    params: dict[str, Any] = field(default_factory=dict)
    name: str = "external_command"

    def synthesize(self, job: GenerationJob, output_dir: Path) -> SynthesisResult:
        audio_path = output_dir / f"{job.id}.wav"
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        values = _format_values(job, output_dir, audio_path)
        command = self._command(values)
        expected_output = self._expected_output(values, audio_path)
        env = os.environ.copy()
        env.update({str(key): str(value) for key, value in self.params.get("env", {}).items()})

        completed = subprocess.run(
            command,
            cwd=self.params.get("cwd"),
            env=env,
            text=True,
            capture_output=True,
            timeout=float(self.params.get("timeout_s", 1800)),
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "external TTS command failed with exit code "
                f"{completed.returncode}\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
            )

        if expected_output != audio_path and expected_output.exists():
            shutil.copyfile(expected_output, audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(
                f"external command completed but did not create expected audio: {audio_path}"
            )

        return SynthesisResult(
            audio_path=audio_path,
            metadata={
                "backend": self.name,
                "command": " ".join(shlex.quote(part) for part in command),
                "expected_output": str(expected_output),
                "stdout_tail": completed.stdout[-2000:],
                "stderr_tail": completed.stderr[-2000:],
                "synthetic_placeholder": False,
            },
        )

    def _command(self, values: dict[str, str]) -> list[str]:
        command = self.params.get("command")
        if not command:
            raise ValueError("external_command backend requires params.command")
        if isinstance(command, list):
            return [str(part).format(**values) for part in command]
        return [str(part).format(**values) for part in shlex.split(str(command))]

    def _expected_output(self, values: dict[str, str], audio_path: Path) -> Path:
        expected = self.params.get("expected_output")
        if expected:
            return Path(str(expected).format(**values)).resolve()
        return audio_path


def _format_values(job: GenerationJob, output_dir: Path, audio_path: Path) -> dict[str, str]:
    return {
        "audio_path": str(audio_path),
        "output_dir": str(output_dir),
        "job_id": job.id,
        "model_id": job.model.id,
        "source_language": job.voice.language,
        "target_language": job.target.language,
        "speaker_id": job.voice.speaker_id,
        "voice_id": job.voice.id,
        "voice_audio_path": str(job.voice.audio_path),
        "voice_transcript": job.voice.transcript or "",
        "target_id": job.target.id,
        "target_text": job.target.text,
    }
