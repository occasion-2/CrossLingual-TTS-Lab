from __future__ import annotations

import ctypes
import os
import site
import sys
from pathlib import Path


_PRELOADED: list[ctypes.CDLL] = []
_PREPARED = False


def prepare_ctranslate2_cuda_libraries() -> list[str]:
    """Make CUDA 12 NVIDIA wheel libraries visible to CTranslate2.

    CTranslate2 wheels used by faster-whisper are built against CUDA 12. The
    project can still use a CUDA 13 Torch build for F5-TTS/SpeechBrain, but
    faster-whisper needs CUDA 12 cuBLAS libraries available in the process
    before it initializes CUDA.
    """
    global _PREPARED
    lib_dirs = _nvidia_library_dirs()
    if _PREPARED:
        return [str(path) for path in lib_dirs]

    if sys.platform.startswith("linux"):
        existing = os.environ.get("LD_LIBRARY_PATH", "")
        merged = [str(path) for path in lib_dirs]
        if existing:
            merged.extend(existing.split(":"))
        os.environ["LD_LIBRARY_PATH"] = ":".join(dict.fromkeys(item for item in merged if item))
        _preload_linux_libraries(lib_dirs)
    elif sys.platform == "win32":
        for path in lib_dirs:
            os.add_dll_directory(str(path))

    _PREPARED = True
    return [str(path) for path in lib_dirs]


def _nvidia_library_dirs() -> list[Path]:
    roots = []
    for package_dir in site.getsitepackages():
        roots.append(Path(package_dir) / "nvidia")
    user_site = site.getusersitepackages()
    if user_site:
        roots.append(Path(user_site) / "nvidia")

    dirs: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.glob("*/lib"):
            if path.exists():
                dirs.append(path.resolve())
        for path in root.glob("cu*/lib"):
            if path.exists():
                dirs.append(path.resolve())
    return list(dict.fromkeys(dirs))


def _preload_linux_libraries(lib_dirs: list[Path]) -> None:
    names = [
        "libcublas.so.12",
        "libcublasLt.so.12",
    ]
    for name in names:
        path = _find_library(lib_dirs, name)
        if path is None:
            continue
        _PRELOADED.append(ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL))


def _find_library(lib_dirs: list[Path], name: str) -> Path | None:
    for directory in lib_dirs:
        path = directory / name
        if path.exists():
            return path
    return None
