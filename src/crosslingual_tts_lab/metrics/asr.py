from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from crosslingual_tts_lab.cuda_libs import prepare_ctranslate2_cuda_libraries
from crosslingual_tts_lab.device import DeviceProfile
from crosslingual_tts_lab.metrics.base import MetricResult
from crosslingual_tts_lab.runner_types import GeneratedSample
from crosslingual_tts_lab.text_metrics import choose_error_rate


@dataclass
class FasterWhisperASRMetric:
    name: str
    params: dict[str, Any]
    device_profile: DeviceProfile
    _model: Any = field(default=None, init=False, repr=False)
    _forced_device: str | None = field(default=None, init=False, repr=False)
    _forced_compute_type: str | None = field(default=None, init=False, repr=False)
    _forced_model_size: str | None = field(default=None, init=False, repr=False)
    _fallback_reason: str | None = field(default=None, init=False, repr=False)

    def evaluate(self, sample: GeneratedSample) -> MetricResult:
        try:
            segments, info = self._transcribe(sample)
            transcript = " ".join(segment.text.strip() for segment in segments).strip()
            error_name, error_value = choose_error_rate(
                sample.job.target.language,
                sample.job.target.text,
                transcript,
            )
            return MetricResult(
                name=self.name,
                status="ok",
                value=round(error_value, 6),
                details={
                    "error_type": error_name,
                    "target_language": sample.job.target.language,
                    "detected_language": getattr(info, "language", None),
                    "detected_language_probability": _round_or_none(
                        getattr(info, "language_probability", None)
                    ),
                    "transcript": transcript,
                    "model_size": self._model_size(),
                    "device": self._device(),
                    "compute_type": self._compute_type(),
                    "fallback_reason": self._fallback_reason,
                },
            )
        except ModuleNotFoundError as exc:
            return _missing_dependency(self.name, "faster-whisper", exc)
        except Exception as exc:
            return _metric_error(self.name, exc)

    def _transcribe(self, sample: GeneratedSample) -> tuple[Any, Any]:
        try:
            model = self._load_model()
            segments, info = model.transcribe(
                str(sample.audio_path),
                language=sample.job.target.language,
                beam_size=int(self.params.get("beam_size", 5)),
                vad_filter=bool(self.params.get("vad_filter", True)),
            )
            return list(segments), info
        except RuntimeError as exc:
            if not self._should_retry_on_cpu(exc):
                raise
            self._fallback_to_cpu(exc)
            model = self._load_model()
            segments, info = model.transcribe(
                str(sample.audio_path),
                language=sample.job.target.language,
                beam_size=int(self.params.get("beam_size", 5)),
                vad_filter=bool(self.params.get("vad_filter", True)),
            )
            return list(segments), info

    def _load_model(self) -> Any:
        if self._model is None:
            if self._device() == "cuda":
                prepare_ctranslate2_cuda_libraries()
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                self._model_size(),
                device=self._device(),
                compute_type=self._compute_type(),
                download_root=self.params.get("download_root"),
            )
        return self._model

    def _model_size(self) -> str:
        return str(
            self._forced_model_size
            or self.params.get("model_size")
            or self.device_profile.recommended_whisper_model
            or "small"
        )

    def _device(self) -> str:
        return str(self._forced_device or self.params.get("device") or self.device_profile.device)

    def _compute_type(self) -> str:
        return str(
            self._forced_compute_type
            or self.params.get("compute_type")
            or self.device_profile.recommended_compute_type
        )

    def _should_retry_on_cpu(self, exc: RuntimeError) -> bool:
        if not bool(self.params.get("allow_cpu_fallback", True)):
            return False
        if self._device() == "cpu":
            return False
        message = str(exc).casefold()
        return "libcublas.so.12" in message or "cublas" in message or "cuda" in message

    def _fallback_to_cpu(self, exc: RuntimeError) -> None:
        self._model = None
        self._forced_device = "cpu"
        self._forced_compute_type = str(self.params.get("cpu_compute_type", "int8"))
        self._forced_model_size = str(self.params.get("cpu_model_size", "small"))
        self._fallback_reason = f"{type(exc).__name__}: {exc}"


@dataclass
class FasterWhisperLIDMetric:
    name: str
    params: dict[str, Any]
    device_profile: DeviceProfile
    _model: Any = field(default=None, init=False, repr=False)
    _forced_device: str | None = field(default=None, init=False, repr=False)
    _forced_compute_type: str | None = field(default=None, init=False, repr=False)
    _forced_model_size: str | None = field(default=None, init=False, repr=False)
    _fallback_reason: str | None = field(default=None, init=False, repr=False)

    def evaluate(self, sample: GeneratedSample) -> MetricResult:
        try:
            segments, info = self._transcribe(sample)
            for _ in segments:
                break
            detected = getattr(info, "language", None)
            probability = getattr(info, "language_probability", None)
            target = sample.job.target.language
            return MetricResult(
                name=self.name,
                status="ok",
                value=_round_or_none(probability),
                details={
                    "target_language": target,
                    "detected_language": detected,
                    "matches_target": detected == target,
                    "model_size": self._model_size(),
                    "device": self._device(),
                    "compute_type": self._compute_type(),
                    "fallback_reason": self._fallback_reason,
                },
            )
        except ModuleNotFoundError as exc:
            return _missing_dependency(self.name, "faster-whisper", exc)
        except Exception as exc:
            return _metric_error(self.name, exc)

    def _transcribe(self, sample: GeneratedSample) -> tuple[Any, Any]:
        try:
            model = self._load_model()
            segments, info = model.transcribe(
                str(sample.audio_path),
                language=None,
                beam_size=1,
                vad_filter=bool(self.params.get("vad_filter", True)),
            )
            return list(segments), info
        except RuntimeError as exc:
            if not self._should_retry_on_cpu(exc):
                raise
            self._fallback_to_cpu(exc)
            model = self._load_model()
            segments, info = model.transcribe(
                str(sample.audio_path),
                language=None,
                beam_size=1,
                vad_filter=bool(self.params.get("vad_filter", True)),
            )
            return list(segments), info

    def _load_model(self) -> Any:
        if self._model is None:
            if self._device() == "cuda":
                prepare_ctranslate2_cuda_libraries()
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                self._model_size(),
                device=self._device(),
                compute_type=self._compute_type(),
                download_root=self.params.get("download_root"),
            )
        return self._model

    def _model_size(self) -> str:
        return str(
            self._forced_model_size
            or self.params.get("model_size")
            or self.device_profile.recommended_whisper_model
            or "small"
        )

    def _device(self) -> str:
        return str(self._forced_device or self.params.get("device") or self.device_profile.device)

    def _compute_type(self) -> str:
        return str(
            self._forced_compute_type
            or self.params.get("compute_type")
            or self.device_profile.recommended_compute_type
        )

    def _should_retry_on_cpu(self, exc: RuntimeError) -> bool:
        if not bool(self.params.get("allow_cpu_fallback", True)):
            return False
        if self._device() == "cpu":
            return False
        message = str(exc).casefold()
        return "libcublas.so.12" in message or "cublas" in message or "cuda" in message

    def _fallback_to_cpu(self, exc: RuntimeError) -> None:
        self._model = None
        self._forced_device = "cpu"
        self._forced_compute_type = str(self.params.get("cpu_compute_type", "int8"))
        self._forced_model_size = str(self.params.get("cpu_model_size", "small"))
        self._fallback_reason = f"{type(exc).__name__}: {exc}"


def _missing_dependency(name: str, package: str, exc: Exception) -> MetricResult:
    return MetricResult(
        name=name,
        status="missing_backend",
        value=None,
        details={
            "package": package,
            "reason": f"{type(exc).__name__}: {exc}",
        },
    )


def _metric_error(name: str, exc: Exception) -> MetricResult:
    return MetricResult(
        name=name,
        status="error",
        value=None,
        details={
            "error_type": type(exc).__name__,
            "error": str(exc),
        },
    )


def _round_or_none(value: Any) -> float | None:
    if value is None:
        return None
    return round(float(value), 6)
