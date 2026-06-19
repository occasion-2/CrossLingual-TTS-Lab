from __future__ import annotations

import warnings
from dataclasses import dataclass, asdict
from typing import Any


@dataclass(frozen=True)
class DeviceProfile:
    device: str
    cuda_available: bool
    torch_version: str | None = None
    torch_cuda_version: str | None = None
    cuda_device_count: int | None = None
    gpu_name: str | None = None
    total_vram_gb: float | None = None
    recommended_whisper_model: str = "small"
    recommended_compute_type: str = "int8"
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def detect_device_profile(prefer_cuda: bool = True) -> DeviceProfile:
    try:
        import torch
    except Exception as exc:
        return DeviceProfile(
            device="cpu",
            cuda_available=False,
            notes=(f"torch unavailable: {type(exc).__name__}: {exc}",),
        )

    torch_version = str(getattr(torch, "__version__", "unknown"))
    torch_cuda_version = getattr(torch.version, "cuda", None)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Can't initialize NVML")
        try:
            device_count = int(torch.cuda.device_count())
        except Exception:
            device_count = None
        cuda_available = bool(prefer_cuda and torch.cuda.is_available())
    if not cuda_available:
        notes = ["CUDA unavailable; using CPU-safe defaults."]
        if torch_cuda_version:
            notes.append(
                f"Torch was built with CUDA {torch_cuda_version}, but no CUDA device is visible."
            )
        return DeviceProfile(
            device="cpu",
            cuda_available=False,
            torch_version=torch_version,
            torch_cuda_version=torch_cuda_version,
            cuda_device_count=device_count,
            notes=tuple(notes),
        )

    index = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(index)
    total_gb = props.total_memory / (1024**3)
    model = "medium" if total_gb >= 11.0 else "small"
    notes = []
    if total_gb >= 11.0:
        notes.append("12GB-class GPU detected; medium Whisper-family ASR is a practical default.")
    else:
        notes.append("Sub-12GB GPU detected; small ASR model is the safer default.")
    return DeviceProfile(
        device="cuda",
        cuda_available=True,
        torch_version=torch_version,
        torch_cuda_version=torch_cuda_version,
        cuda_device_count=device_count,
        gpu_name=props.name,
        total_vram_gb=round(total_gb, 2),
        recommended_whisper_model=model,
        recommended_compute_type="float16",
        notes=tuple(notes),
    )
