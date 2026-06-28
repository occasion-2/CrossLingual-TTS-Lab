# CrossLingual TTS Lab

CrossLingual TTS Lab is a benchmark harness for evaluating cross-lingual zero-shot voice cloning, focusing on target-language intelligibility, target-language identification, speaker preservation, and source-language leakage.

Core question:

> When a reference voice is in one language and target text is in another, does
> the model preserve speaker identity without leaking source-language accent,
> phonetics, or prosody into the target language?

The harness started as a lightweight `uv` project with a config-driven run planner, a dummy synthesis backend, pluggable model/metric interfaces, and machine-readable plus Markdown reports. It now includes several real TTS backends, metric adapters, and a source-language leakage probe, while SER-based emotion preservation metrics remain a future extension.

## Quick Start

From this source checkout, use `xttslab.py`. It loads the package from `src/`
directly, so you do not need to install the package just to smoke-test the
pipeline.

```bash
uv run python xttslab.py plan --config configs/mini.toml
uv run python xttslab.py run --config configs/mini.toml --out runs/mini
uv run python xttslab.py report --run runs/mini
```

The default config uses the built-in `dummy` backend. It writes deterministic
WAV files and metric placeholders, which makes the whole pipeline testable before
large model dependencies are installed.

The run creates:

- `runs/mini/audio/*.wav` for generated audio
- `runs/mini/manifest.json` for machine-readable sample and metric records
- `runs/mini/report.md` for a readable summary

If synthesis succeeded and only metrics need to be recomputed, reuse the WAV
files instead of regenerating audio:

```bash
uv run python xttslab.py score --config configs/mini.toml --run runs/mini
```

To start a new editable config:

```bash
uv run python xttslab.py init configs/my-mini.toml
uv run python xttslab.py plan --config configs/my-mini.toml
```

In a normal writable Python environment you can also install the package and use
the shorter console script:

```bash
uv run xttslab plan --config configs/mini.toml
```

In restricted environments where `uv` cannot write to its default cache under
your home directory, point the cache at a writable directory:

```bash
UV_CACHE_DIR=.uv-cache uv run python xttslab.py plan --config configs/mini.toml
```

## Open Dataset Path

For a real open-data slice on Hugging Face, generate a config from Google
FLEURS. For Qwen-TTS, keep the benchmark id short and put the Hugging Face model
repo in `--model-param model=...`:

```bash
uv run python xttslab.py dataset fleurs \
  --languages ru:ru,en:en,zh-CN:zh \
  --voices-per-language 4 \
  --targets-per-language 8 \
  --target-languages en,zh \
  --max-voice-chars 120 \
  --max-target-chars 110 \
  --model-id qwen3_tts_1_7b_base \
  --model-backend qwen_tts \
  --model-param model=Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --model-param ref_text_mode=empty \
  --out configs/fleurs_ru_en_zh.toml
```

Do not combine `--model-backend qwen_tts` with
`--model-param model=F5TTS_v1_Base`: that asks the Qwen backend to download an
F5 model as if it were a Qwen Hugging Face repo.

If the Qwen stack prints `sox: command not found`, install the system SoX
binary before running synthesis. On Debian/Ubuntu:

```bash
sudo apt-get install sox libsox-fmt-all
```

Use `dataset_code:benchmark_code` for languages when the dataset code differs
from the TTS/ASR code you want in the benchmark. The FLEURS command accepts
short aliases for this benchmark: `ru`, `en`, and `zh-CN` map to FLEURS
`ru_ru`, `en_us`, and `cmn_hans_cn`.

Common Voice remains supported as a command target, but Mozilla Common Voice
datasets on Hugging Face are now placeholder/empty repos after the move to
Mozilla Data Collective in October 2025. Use FLEURS for the direct HF path, or
download Common Voice manually from Mozilla Data Collective and build a local
config.

For F5-TTS, avoid Russian target text for now: the base F5 model is much more
usable for English/Mandarin targets, while Russian references can still be used
as voice prompts. The FLEURS generator also removes spaces between Mandarin
characters before writing the config.

The equivalent F5 command uses the F5 backend and F5 model parameter:

```bash
uv run python xttslab.py dataset fleurs \
  --languages ru:ru,en:en,zh-CN:zh \
  --voices-per-language 2 \
  --targets-per-language 4 \
  --target-languages en,zh \
  --max-voice-chars 120 \
  --max-target-chars 110 \
  --model-id f5tts_v1_base \
  --model-backend f5_tts \
  --model-param model=F5TTS_v1_Base \
  --model-param ref_text_mode=empty \
  --out configs/fleurs_ru_en_zh_f5.toml
```

Then inspect and run:

```bash
uv run python xttslab.py plan --config configs/fleurs_ru_en_zh.toml
uv run python xttslab.py run \
  --config configs/fleurs_ru_en_zh.toml \
  --out runs/fleurs_ru_en_zh
```

The generated config uses open-source metric adapters:

- `faster_whisper_asr` for target-language WER/CER
- `faster_whisper_lid` for generated-audio language identification
- `speechbrain_speaker_similarity` for reference/generated speaker similarity

## GPU Profile

Check what the runner sees:

```bash
uv run python xttslab.py doctor
```

On CUDA-enabled cards (such as 12GB-class GPUs), the default execution uses float16 and a
`medium` Whisper-family ASR model (falling back to `small` on smaller VRAM cards). The project
pins the Linux GPU stack to `torch>=2.11,<2.12` and `torchaudio>=2.11,<2.12` from the PyTorch
`cu130` wheel index. If `doctor` reports a CUDA-built Torch but zero visible devices, the
Python package is correct and the issue is device visibility in the current container/session.

The ASR/LID metrics use faster-whisper/CTranslate2, which currently expects
CUDA 12 cuBLAS even when Torch itself is CUDA 13. The `metrics`
extra therefore installs `nvidia-cublas-cu12`, and the runner preloads cuBLAS
before creating a CUDA Whisper model. Do not install `nvidia-cudnn-cu12` into
the main environment: it shares the `nvidia/cudnn` path with Torch's CUDA 13 cuDNN
package and can break SpeechBrain/PyTorch speaker similarity. Note that the
CosyVoice isolated virtual environment (`overnight_runs/.venv_cosyvoice`) explicitly installs
`nvidia-cudnn-cu12` because the CosyVoice backend uses ONNX Runtime GPU, which requires it.
## Running the FLEURS Benchmark Experiment

To evaluate multiple model backends cleanly without PyTorch/CUDA dependency poisoning, run the automated experiment pipeline:

1. **Install system prerequisites** (e.g. `sox` is required by Qwen-TTS):
   ```bash
   sudo apt-get install sox libsox-fmt-all
   ```

2. **Install external repositories and weights**:
   ```bash
   ./install_dependencies.sh
   ```
    This will clone the `CosyVoice` and `Spark-TTS` repositories locally and download the 2GB Spark-TTS pre-trained weights.

3. **Install Python dependencies for specific backends**:
   All model backend dependencies are cleanly separated into optional-dependencies (extras) inside [pyproject.toml](pyproject.toml). You can install the dependencies of your choice directly into your active virtual environment:

   * **Using the Helper Script** (checks for `uv` or `pip` automatically):
     - For CosyVoice: `./install_dependencies.sh --cosyvoice`
     - For Spark-TTS: `./install_dependencies.sh --spark-tts`

   * **Installing Manually**:
     - **XTTS**: `uv pip install -e ".[open-data,metrics,tts]"`
     - **F5-TTS**: `uv pip install -e ".[open-data,metrics,f5]"`
     - **Qwen-TTS**: `uv pip install -e ".[open-data,metrics,qwen]"`
     - **Spark-TTS**: `uv pip install -e ".[open-data,metrics,spark-tts]"`
     - **CosyVoice** (requires pre-installing setuptools/wheel and bypassing build isolation for its legacy dependencies):
       ```bash
       uv pip install "setuptools<70" wheel
       uv pip install -e ".[open-data,metrics,cosyvoice]" --no-build-isolation-package openai-whisper --no-build-isolation-package deepspeed
       ```

4. **Run the experiment example**:
   ```bash
   ./run_fleurs_experiment_example.sh
   ```
   This script automatically configures isolated virtual environments for each model (under `overnight_runs/`), handles their incompatible CUDA/PyTorch package resolutions, plans the FLEURS slice, synthesizes the audio, and scores metrics.

5. **WAV-level Resumability**:
   If a run gets interrupted or fails for one model, running `./run_fleurs_experiment_example.sh` again will instantly skip completed models (based on `report.md`) and skip already-synthesized WAV files, resuming right from the point of failure.

To switch a generated config from the dummy backend to XTTS, set:

```toml
[[models]]
id = "xtts_v2"
backend = "coqui_xtts"
params = { model_name = "tts_models/multilingual/multi-dataset/xtts_v2" }
```

You can also use F5-TTS through its Python API:

```toml
[[models]]
id = "f5tts_v1_base"
backend = "f5_tts"
params = { model = "F5TTS_v1_Base", ref_text_mode = "transcript", nfe_step = 32 }
```

Or through an installed CLI with the generic command backend:

```toml
[[models]]
id = "f5tts_cli"
backend = "external_command"
params = { command = ["f5-tts_infer-cli", "--model", "F5TTS_v1_Base", "--ref_audio", "{voice_audio_path}", "--ref_text", "{voice_transcript}", "--gen_text", "{target_text}", "--output_dir", "{output_dir}"], expected_output = "{output_dir}/infer_cli_basic.wav" }
```

The command backend supports placeholders such as `{audio_path}`,
`{output_dir}`, `{voice_audio_path}`, `{voice_transcript}`, `{target_text}`,
`{source_language}`, and `{target_language}`. Use it for models that are easier
to run from their own CLI or a separate environment.

Backend names are resolved through aliases, so these are equivalent where
appropriate: `coqui_xtts`, `xtts`, `xtts_v2`; `f5_tts`, `f5`, `f5tts`;
`qwen_tts`, `qwen`, `qwentts`, `qwen3_tts`; and `external_command`, `command`,
`cli`.

Metric backends are configured in `[[metrics]]` blocks. Omit the section to use
the deterministic placeholder metrics, or set real adapters explicitly:

```toml
[[metrics]]
id = "asr_error"
backend = "faster_whisper_asr"
params = { vad_filter = true }

[[metrics]]
id = "target_language_id"
backend = "faster_whisper_lid"
params = { vad_filter = true }

[[metrics]]
id = "speaker_similarity"
backend = "speechbrain_speaker_similarity"
params = {}
```

## What The Report Tracks

- target-language intelligibility through ASR WER/CER when configured
- speaker preservation through speaker embeddings when configured
- language-ID confidence on generated audio when configured
- source-language similarity / leakage-proxy metrics when configured
- optional emotion-preservation placeholders until an SER backend is added

Each generated sample is associated with:

- model id and backend
- source voice language and speaker id
- target language and text
- output audio path
- metric records with explicit status: `ok`, `missing_backend`, or `error`

## Project Layout

```text
src/crosslingual_tts_lab/
  cli.py              # stdlib CLI entrypoint
  config.py           # TOML/JSON config loading and validation
  planner.py          # expands model x pair benchmark jobs
  runner.py           # generation + metric execution
  report.py           # JSON/Markdown report writer
  audio.py            # tiny deterministic WAV helper for dummy backend
  cuda_libs.py        # cuBLAS/CTranslate2 helper for CUDA metrics execution
  device.py           # device detection and profile generator (CPU vs GPU)
  open_datasets.py    # config builder for open datasets (FLEURS, Common Voice)
  runner_types.py     # common dataclasses used across runner and metrics
  text_metrics.py     # calculation of WER and CER error metrics
  backends/           # TTS backend interface and implementations
  metrics/            # metric interface and baseline placeholder metrics
configs/
  mini.toml           # example benchmark
tests/
  test_config_and_runner.py  # test suite for config, dataset, and runner logic
```

## Config Shape

```toml
name = "mini-ru-crosslingual"

[[models]]
id = "dummy_tts"
backend = "dummy"

[[voices]]
id = "ru_ref_001"
language = "ru"
speaker_id = "cv-ru-demo-001"
audio_path = "data/reference/ru_ref_001.wav"
transcript = "eto korotkaya russkaya referensnaya fraza"

[[targets]]
id = "en_weather"
language = "en"
text = "The weather changed quickly, but the speaker stayed calm."

[[pairs]]
voice = "ru_ref_001"
target = "en_weather"
```

## Reproducibility Snapshot

- **Dataset**: Google FLEURS
- **Config**: `configs/fleurs_tiny_all.toml`
- **Languages**: English, Russian, Mandarin Chinese
- **Jobs per full direction**: 100
- **ASR/LID backend**: faster-whisper (medium/small depending on VRAM)
- **Speaker similarity**: SpeechBrain ECAPA-TDNN (`speechbrain/spkrec-ecapa-voxceleb`)
- **Confidence intervals**: 95% bootstrap (1000 resamples)
- **Hardware**: CUDA-enabled GPUs (e.g., 12GB+ VRAM class)
- **Random seed**: System-default pseudo-random sampling during subset generation

## Benchmark Results on Google FLEURS

The ASR evaluation uses target-language specific text-normalization adapters (preprocessors) to clean reference and hypothesis transcriptions (handling lowercase, removing punctuation, and stripping spaces for CJK characters) before computing WER/CER. 

The benchmark harness is being evaluated on a cross-lingual subset of the Google FLEURS dataset (`configs/fleurs_tiny_all.toml`) across several state-of-the-art zero-shot voice cloning models. The real metrics stack includes `faster_whisper_asr` for ASR error (measuring target-language intelligibility), `faster_whisper_lid` for target language identification (acting as a proxy indicator for successful target-language rendering), `speechbrain_speaker_similarity` (ECAPA-TDNN) for speaker-similarity preservation, and `speechbrain_language_similarity` (VoxLingua107) to measure source-language leakage.

Below is the comparative summary of the cross-lingual generalization capabilities of the installed models.

*Note: Lower ASR error indicates better target-text intelligibility under the chosen ASR and normalization pipeline. Higher Target LID indicates the model successfully transitioned to the target language. Higher Speaker Sim indicates the target-language voice effectively cloned the source speaker's identity. Higher Leakage indicates the generated audio sounds more like the source language's accent/prosody.*

### Table 1: Common Target-Language Subset
*Only `en` and `zh` target conditions. Excludes target-Russian directions to avoid unsupported/degraded model conditions. F5-TTS target-Russian results are reported in Table 4 for transparency but excluded from this table because the base F5 model is not expected to handle Russian target synthesis reliably.*

| Model | n | ASR Error ↓ (95% CI) | Target LID ↑ (95% CI) | Speaker Sim ↑ (95% CI) |
|---|---|---|---|---|
| Qwen3-TTS 1.7B | 400 | 7.0% [5.9–8.2] | 96.1% [95.4–96.8] | 0.515 [0.503–0.527] |
| Qwen3-TTS 0.6B | 397 | 7.8% [6.6–9.1] | 94.6% [93.8–95.4] | 0.516 [0.503–0.529] |
| XTTS v2 | 400 | 9.6% [8.1–11.2] | 97.7% [97.1–98.1] | 0.468 [0.456–0.479] |
| Spark-TTS | 400 | 11.9% [10.5–13.5] | 96.4% [95.7–97.1] | 0.420 [0.409–0.431] |
| CosyVoice | 400 | 17.9% [15.5–20.5] | 82.1% [79.8–84.1] | 0.688 [0.673–0.701] |
| F5-TTS | 400 | 31.8% [28.2–35.1] | 90.3% [88.8–91.8] | 0.530 [0.509–0.551] |

### Table 2: Target-Language Aggregates
*Aggregated by target language across all sources.*

| Model | Target | n | ASR Error ↓ (95% CI) | Target LID ↑ (95% CI) | Speaker Sim ↑ (95% CI) |
|---|---|---|---|---|---|
| Qwen3-TTS 1.7B | en | 200 | 3.9% [3.0–4.8] | 93.8% [92.5–94.9] | 0.556 [0.541–0.573] |
| Qwen3-TTS 0.6B | en | 199 | 5.1% [3.8–6.5] | 91.0% [89.7–92.3] | 0.571 [0.557–0.585] |
| XTTS v2 | en | 200 | 3.7% [2.7–4.6] | 97.1% [96.7–97.5] | 0.497 [0.483–0.509] |
| Spark-TTS | en | 200 | 4.8% [3.6–6.3] | 93.9% [92.7–95.0] | 0.433 [0.419–0.445] |
| CosyVoice | en | 200 | 12.2% [9.7–14.9] | 77.8% [75.1–80.6] | 0.719 [0.704–0.734] |
| F5-TTS | en | 200 | 13.1% [10.7–15.7] | 94.6% [93.7–95.5] | 0.610 [0.596–0.622] |
| Qwen3-TTS 1.7B | ru | 200 | 1.5% [1.0–2.1] | 97.0% [96.5–97.5] | 0.479 [0.458–0.499] |
| Qwen3-TTS 0.6B | ru | 194 | 4.1% [3.1–5.1] | 98.3% [97.9–98.6] | 0.449 [0.429–0.470] |
| XTTS v2 | ru | 200 | 7.5% [6.0–9.5] | 98.1% [97.3–98.7] | 0.456 [0.435–0.477] |
| Spark-TTS | ru | 0 | - | - | - |
| CosyVoice | ru | 200 | 53.3% [47.0–60.2] | 70.3% [67.2–73.2] | 0.731 [0.713–0.748] |
| F5-TTS | ru | 200 | 131.9% [122.7–142.0] | 68.5% [65.8–71.2] | 0.526 [0.484–0.565] |
| Qwen3-TTS 1.7B | zh | 200 | 10.1% [8.1–12.0] | 98.5% [98.2–98.8] | 0.474 [0.457–0.493] |
| Qwen3-TTS 0.6B | zh | 198 | 10.6% [8.9–12.7] | 98.2% [97.8–98.6] | 0.460 [0.441–0.478] |
| XTTS v2 | zh | 200 | 15.5% [12.7–18.6] | 98.2% [97.2–99.0] | 0.439 [0.418–0.459] |
| Spark-TTS | zh | 200 | 19.1% [16.8–21.6] | 98.9% [98.2–99.4] | 0.408 [0.387–0.428] |
| CosyVoice | zh | 200 | 23.5% [19.8–27.6] | 86.3% [82.9–89.1] | 0.657 [0.634–0.678] |
| F5-TTS | zh | 200 | 50.5% [45.4–55.9] | 86.0% [83.3–88.7] | 0.451 [0.415–0.486] |

**Interpretation:** Target-language aggregation shows that target Chinese is more difficult for most systems than target English or Russian. Qwen3-TTS remains the most balanced system, while XTTS is highly competitive for target English but degrades on target Chinese. F5-TTS collapses on target Russian, supporting the decision to separate full-coverage and common-subset comparisons.

### Table 3: Source-Language Aggregates (Speaker Similarity)
*Aggregated by source language to show how well each model retains speaker identity across origin languages.*

| Model | Source | n | Speaker Sim ↑ (95% CI) |
|---|---|---|---|
| Qwen3-TTS 1.7B | en | 200 | 0.379 [0.365–0.394] |
| Qwen3-TTS 0.6B | en | 194 | 0.348 [0.334–0.364] |
| XTTS v2 | en | 200 | 0.352 [0.333–0.370] |
| Spark-TTS | en | 100 | 0.319 [0.296–0.342] |
| CosyVoice | en | 200 | 0.626 [0.604–0.647] |
| F5-TTS | en | 200 | 0.316 [0.275–0.359] |
| Qwen3-TTS 1.7B | ru | 200 | 0.550 [0.534–0.564] |
| Qwen3-TTS 0.6B | ru | 200 | 0.549 [0.534–0.564] |
| XTTS v2 | ru | 200 | 0.478 [0.466–0.489] |
| Spark-TTS | ru | 200 | 0.459 [0.445–0.474] |
| CosyVoice | ru | 200 | 0.722 [0.706–0.737] |
| F5-TTS | ru | 200 | 0.581 [0.566–0.595] |
| Qwen3-TTS 1.7B | zh | 200 | 0.580 [0.567–0.593] |
| Qwen3-TTS 0.6B | zh | 197 | 0.582 [0.571–0.593] |
| XTTS v2 | zh | 200 | 0.562 [0.552–0.572] |
| Spark-TTS | zh | 100 | 0.443 [0.426–0.461] |
| CosyVoice | zh | 200 | 0.759 [0.744–0.773] |
| F5-TTS | zh | 200 | 0.689 [0.679–0.699] |

**Interpretation:** Source-language aggregation reveals that ECAPA speaker similarity depends strongly on the reference language, with English references producing remarkably lower similarity across several models compared to Chinese or Russian references. This strongly motivates future calibration against real-real same-speaker and different-speaker baselines to isolate true identity preservation from source-language acoustics.

### Table 4: Per-Direction Breakdowns
*Provides full visibility into specific language pairs, exposing asymmetric performance.*

| Model | Direction | n | ASR Error ↓ (95% CI) | Target LID ↑ (95% CI) | Speaker Sim ↑ (95% CI) |
|---|---|---|---|---|---|
| Qwen3-TTS 1.7B | en->ru | 100 | 1.4% [0.8–2.2] | 95.5% [94.5–96.4] | 0.365 [0.345–0.389] |
| Qwen3-TTS 1.7B | en->zh | 100 | 6.6% [4.2–9.1] | 97.6% [97.1–98.0] | 0.394 [0.373–0.413] |
| Qwen3-TTS 1.7B | ru->en | 100 | 4.4% [3.1–5.8] | 92.8% [90.7–94.7] | 0.545 [0.523–0.570] |
| Qwen3-TTS 1.7B | ru->zh | 100 | 13.6% [10.7–16.6] | 99.4% [99.3–99.5] | 0.555 [0.535–0.574] |
| Qwen3-TTS 1.7B | zh->en | 100 | 3.4% [2.2–4.8] | 94.7% [93.4–95.9] | 0.567 [0.546–0.589] |
| Qwen3-TTS 1.7B | zh->ru | 100 | 1.6% [0.9–2.4] | 98.5% [98.2–98.7] | 0.592 [0.578–0.607] |
| Qwen3-TTS 0.6B | en->ru | 96 | 5.8% [4.1–7.7] | 97.2% [96.7–97.8] | 0.327 [0.304–0.349] |
| Qwen3-TTS 0.6B | en->zh | 98 | 7.5% [5.0–9.9] | 97.1% [96.4–97.7] | 0.369 [0.349–0.391] |
| Qwen3-TTS 0.6B | ru->en | 100 | 5.1% [3.4–6.7] | 93.0% [91.2–94.5] | 0.548 [0.526–0.570] |
| Qwen3-TTS 0.6B | ru->zh | 100 | 13.7% [10.9–16.6] | 99.3% [99.2–99.4] | 0.550 [0.532–0.568] |
| Qwen3-TTS 0.6B | zh->en | 99 | 5.0% [3.4–6.9] | 89.0% [86.8–91.0] | 0.595 [0.580–0.611] |
| Qwen3-TTS 0.6B | zh->ru | 98 | 2.5% [1.5–3.6] | 99.2% [99.1–99.4] | 0.569 [0.553–0.586] |
| XTTS v2 | en->ru | 100 | 8.9% [6.2–12.8] | 97.2% [95.7–98.5] | 0.336 [0.311–0.362] |
| XTTS v2 | en->zh | 100 | 17.9% [12.7–23.4] | 96.9% [95.0–98.5] | 0.368 [0.342–0.394] |
| XTTS v2 | ru->en | 100 | 3.5% [2.3–4.8] | 97.0% [96.5–97.5] | 0.447 [0.434–0.460] |
| XTTS v2 | ru->zh | 100 | 13.1% [10.5–15.8] | 99.5% [99.4–99.5] | 0.509 [0.492–0.526] |
| XTTS v2 | zh->en | 100 | 3.8% [2.3–5.3] | 97.2% [96.6–97.8] | 0.547 [0.533–0.562] |
| XTTS v2 | zh->ru | 100 | 6.2% [4.8–7.6] | 98.9% [98.8–99.1] | 0.577 [0.564–0.588] |
| Spark-TTS | en->ru | 0 | - | - | - |
| Spark-TTS | en->zh | 100 | 25.5% [22.0–29.1] | 98.4% [97.1–99.4] | 0.319 [0.295–0.343] |
| Spark-TTS | ru->en | 100 | 5.4% [3.4–7.8] | 93.2% [91.1–94.9] | 0.422 [0.402–0.441] |
| Spark-TTS | ru->zh | 100 | 12.7% [10.2–15.6] | 99.4% [99.3–99.5] | 0.497 [0.479–0.515] |
| Spark-TTS | zh->en | 100 | 4.3% [2.9–5.8] | 94.6% [93.2–95.9] | 0.443 [0.427–0.461] |
| Spark-TTS | zh->ru | 0 | - | - | - |
| CosyVoice | en->ru | 100 | 39.0% [32.9–45.1] | 68.6% [64.3–72.7] | 0.675 [0.645–0.701] |
| CosyVoice | en->zh | 100 | 14.3% [9.8–18.8] | 95.2% [93.1–96.9] | 0.577 [0.545–0.604] |
| CosyVoice | ru->en | 100 | 11.1% [8.1–14.5] | 72.3% [67.9–76.4] | 0.708 [0.685–0.728] |
| CosyVoice | ru->zh | 100 | 32.7% [26.6–38.5] | 77.4% [72.0–82.4] | 0.737 [0.713–0.758] |
| CosyVoice | zh->en | 100 | 13.2% [9.4–17.2] | 83.4% [80.9–86.1] | 0.731 [0.708–0.749] |
| CosyVoice | zh->ru | 100 | 67.7% [56.1–78.2] | 71.9% [67.5–76.4] | 0.787 [0.767–0.802] |
| F5-TTS | en->ru | 100 | 117.5% [108.0–128.2] | 64.9% [62.0–67.8] | 0.331 [0.269–0.395] |
| F5-TTS | en->zh | 100 | 55.4% [46.4–64.9] | 76.3% [71.5–80.4] | 0.301 [0.242–0.360] |
| F5-TTS | ru->en | 100 | 22.5% [18.7–26.9] | 93.8% [92.6–94.8] | 0.562 [0.542–0.580] |
| F5-TTS | ru->zh | 100 | 45.6% [39.8–53.0] | 95.8% [93.6–97.5] | 0.600 [0.582–0.618] |
| F5-TTS | zh->en | 100 | 3.7% [2.5–5.2] | 95.5% [94.1–96.7] | 0.658 [0.643–0.673] |
| F5-TTS | zh->ru | 100 | 146.3% [130.9–163.4] | 72.1% [67.9–76.5] | 0.721 [0.712–0.731] |

**Interpretation:** Aggregate averages hide severe model-specific and direction-specific failures. Cross-lingual zero-shot voice cloning is highly direction-dependent. For example, while F5-TTS achieves an impressive 3.7% ASR Error on `zh->en`, it completely fails on `*->ru`. CosyVoice struggles with intelligibility in most cross-lingual pairs (e.g., 67.7% ASR Error for `zh->ru`), despite scoring the highest speaker similarity.

### Table 5: Pareto Ranking
Better intelligibility / target-language transfer does **not** imply better speaker preservation. This tradeoff is evident across the models:

- **Best intelligibility**: Qwen3-TTS 1.7B
- **Best target LID**: XTTS v2
- **Best speaker similarity**: CosyVoice
- **Best small model tradeoff**: Qwen3-TTS 0.6B

### Table 6: Source-Language Similarity (Leakage Proxy)
*Cosine similarity of generated audio language embeddings to the source language reference.*

| Model | Direction | n | Source-language similarity ↑ (95% CI) |
|---|---|---|---|
| F5-TTS | en->ru | 100 | 0.649 [0.599–0.697] |
| F5-TTS | en->zh | 100 | 0.642 [0.599–0.686] |
| F5-TTS | ru->en | 100 | 0.825 [0.818–0.832] |
| F5-TTS | ru->zh | 100 | 0.829 [0.823–0.835] |
| F5-TTS | zh->en | 100 | 0.846 [0.840–0.853] |
| F5-TTS | zh->ru | 100 | 0.910 [0.906–0.913] |
| CosyVoice | en->ru | 100 | 0.899 [0.888–0.906] |
| CosyVoice | en->zh | 100 | 0.874 [0.860–0.886] |
| CosyVoice | ru->en | 100 | 0.883 [0.874–0.891] |
| CosyVoice | ru->zh | 100 | 0.894 [0.882–0.902] |
| CosyVoice | zh->en | 100 | 0.874 [0.866–0.881] |
| CosyVoice | zh->ru | 100 | 0.909 [0.904–0.913] |
| Qwen3-TTS 0.6B | en->ru | 96 | 0.805 [0.798–0.812] |
| Qwen3-TTS 0.6B | en->zh | 98 | 0.845 [0.841–0.849] |
| Qwen3-TTS 0.6B | ru->en | 100 | 0.827 [0.819–0.834] |
| Qwen3-TTS 0.6B | ru->zh | 100 | 0.828 [0.824–0.833] |
| Qwen3-TTS 0.6B | zh->en | 99 | 0.856 [0.850–0.861] |
| Qwen3-TTS 0.6B | zh->ru | 98 | 0.820 [0.815–0.825] |
| Qwen3-TTS 1.7B | en->ru | 100 | 0.808 [0.801–0.816] |
| Qwen3-TTS 1.7B | en->zh | 100 | 0.852 [0.847–0.856] |
| Qwen3-TTS 1.7B | ru->en | 100 | 0.815 [0.809–0.822] |
| Qwen3-TTS 1.7B | ru->zh | 100 | 0.829 [0.824–0.834] |
| Qwen3-TTS 1.7B | zh->en | 100 | 0.841 [0.835–0.847] |
| Qwen3-TTS 1.7B | zh->ru | 100 | 0.814 [0.810–0.818] |
| Spark-TTS | en->zh | 100 | 0.848 [0.826–0.865] |
| Spark-TTS | ru->en | 100 | 0.791 [0.773–0.804] |
| Spark-TTS | ru->zh | 100 | 0.830 [0.825–0.835] |
| Spark-TTS | zh->en | 100 | 0.854 [0.850–0.858] |
| XTTS v2 | en->ru | 100 | 0.798 [0.787–0.809] |
| XTTS v2 | en->zh | 100 | 0.835 [0.824–0.844] |
| XTTS v2 | ru->en | 100 | 0.813 [0.806–0.821] |
| XTTS v2 | ru->zh | 100 | 0.828 [0.823–0.833] |
| XTTS v2 | zh->en | 100 | 0.858 [0.854–0.863] |
| XTTS v2 | zh->ru | 100 | 0.836 [0.832–0.841] |

**Interpretation:** The leakage probe reveals a critical tradeoff in CosyVoice: its consistently high "Speaker Similarity" (from Table 1/2) is directly correlated with high source-language similarity (~0.87–0.90 across the board). It achieves high speaker embedding scores by refusing to fully adapt to target-language phonetics, explaining its poor intelligibility. In contrast, Qwen3-TTS shows lower source-language similarity than CosyVoice while maintaining stronger intelligibility, suggesting better separation between voice identity and source-language acoustic cues under this probe.

### Leakage Metric Caveat
The current leakage score is an embedding-based proxy. It measures source-language similarity in VoxLingua107 embedding space and should be interpreted relatively across models and directions, not as a calibrated perceptual accent-leakage score. Future work will validate it against target-language similarity, real-language centroids, and human accent/prosody judgments.

### Future Work: TASLP Methodological Improvements
To rigorously validate the leakage metric and speaker similarity for peer-reviewed publication, the following calibrations will be added:

| Needed | Why |
|---|---|
| source vs target similarity delta | makes leakage relative, not absolute |
| real FLEURS language centroids | avoids single-reference noise |
| generated-vs-target similarity | checks whether output adapted |
| human accent/nativeness labels | validates the proxy |
| speaker-sim positive/negative calibration | prevents ECAPA overinterpretation |

### Future Work: Speaker-Similarity Calibration
Speaker similarity currently requires calibration against ground-truth positive/negative bounds to fully disentangle voice preservation from channel or language artifacts. Without these, it is difficult to determine if CosyVoice’s high speaker similarity is true voice preservation or an artifact of failed linguistic transfer. Future versions will include the following calibration baselines:

| Pair type | Speaker Sim |
|---|---:|
| same speaker real-real | upper bound |
| different speaker same language | lower bound |
| different speaker cross-language | lower bound |
| generated vs wrong reference | false-positive check |

## Completed Integrations

1. **Real TTS Model Backends**:
   - **F5-TTS** (`F5TTSBackend` / `f5_tts`): Official Python API integration.
   - **XTTS/Coqui** (`CoquiXTTSBackend` / `coqui_xtts` / `xtts`): Multi-lingual clone support through the `TTS` API (patched for PyTorch 2.6).
   - **CosyVoice** (`CosyVoiceBackend` / `cosyvoice`): Zero-shot cloning through FunASR/CosyVoice python API.
   - **Spark-TTS** (`SparkTTSBackend` / `spark_tts`): Zero-shot cloning through the `SparkTTS` python API.
   *(Note: Because these cutting-edge models have deeply conflicting CUDA and PyTorch dependencies, they cannot coexist in a single environment. The `./run_fleurs_experiment_example.sh` script automatically constructs perfectly isolated virtual environments for each model to safely run them without dependency crashes).*
2. **ASR adapters per target language** to compute WER/CER:
   - English, Russian, and Chinese (`zh`/`cmn`) specific normalizers in [text_metrics.py](src/crosslingual_tts_lab/text_metrics.py).

## Next Integrations

The intended next pieces are:

1. Add speaker-verification embeddings for speaker similarity. (Completed - uses SpeechBrain ECAPA-TDNN)
2. Add LID inference on generated audio. (Completed - uses Faster Whisper LID)
3. Add a source-language leakage probe trained on generated audio embeddings while controlling for target language. (Completed - uses SpeechBrain VoxLingua107 embeddings)
4. Add optional emotion preservation metrics from SER models and emotion-labeled subsets.
