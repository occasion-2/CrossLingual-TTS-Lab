from __future__ import annotations

import hashlib
import math
import wave
from pathlib import Path


def write_tone_wav(
    path: Path,
    key: str,
    duration_s: float = 1.2,
    sample_rate: int = 16_000,
) -> None:
    """Write a small deterministic tone WAV used by the dummy backend."""
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    base_freq = 180 + digest[0] % 260
    wobble_freq = 2 + digest[1] % 6
    amplitude = 0.18
    n_samples = int(duration_s * sample_rate)

    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        frames = bytearray()
        for index in range(n_samples):
            t = index / sample_rate
            envelope = min(1.0, index / (0.05 * sample_rate))
            envelope *= min(1.0, (n_samples - index) / (0.08 * sample_rate))
            freq = base_freq + 18 * math.sin(2 * math.pi * wobble_freq * t)
            sample = amplitude * envelope * math.sin(2 * math.pi * freq * t)
            frames.extend(int(sample * 32767).to_bytes(2, "little", signed=True))
        wav.writeframes(bytes(frames))
