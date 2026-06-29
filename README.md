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

### Common Voice Speaker Calibration Run

FLEURS is still the main direction-aware benchmark used in the paper tables, but its speaker labels are not strong enough for ground-truth same-speaker calibration. To get stronger speaker-similarity bounds, run the Common Voice companion script:

```bash
./run_common_voice_calibration.sh
```

This script builds Common Voice configs with repeated reference utterances per known `client_id`/`speaker_id`, runs the same isolated model stack under `overnight_runs_cv/`, and writes `calibration.md` for each model. The calibration command now prefers known same-speaker pairs when repeated speaker IDs are present, while retaining the older inferred FLEURS fallback for legacy runs.

Current `overnight_runs_cv/` snapshot: each full model run contains 600 cross-lingual jobs from 30 Common Voice prompts and 30 targets. F5-TTS, Qwen3-TTS 0.6B, Qwen3-TTS 1.7B, XTTS v2, and CosyVoice all completed 600 scored samples. Spark-TTS completed the 400 supported English/Chinese-target samples and records 200 expected placeholders for target-Russian directions.

The speaker-calibration bounds now come from real Common Voice repeated-speaker IDs, not from inferred FLEURS pseudo-pairs:

| Pair type | Speaker Sim |
|---|---|
| same speaker real-real (known speaker ID) | 0.635 ± 0.129 (n=15) |
| different speaker same language | 0.104 ± 0.104 (n=120) |
| different speaker cross-language | 0.081 ± 0.090 (n=300) |

Generated-vs-wrong-reference sanity checks from the same run stay near the negative bounds: F5-TTS 0.042 ± 0.075 (n=600), Qwen3-TTS 0.6B 0.059 ± 0.068 (n=600), Qwen3-TTS 1.7B 0.052 ± 0.063 (n=600), XTTS v2 0.064 ± 0.070 (n=600), CosyVoice 0.078 ± 0.082 (n=600), and Spark-TTS 0.058 ± 0.079 (n=400).

Common Voice is no longer usable through the old Hugging Face placeholder repos. The companion script now uses the official Mozilla Data Collective API to fetch the requested locale archives, then extracts only the selected `validated.tsv` rows and clips into `overnight_runs_cv/common_voice/`. Put your Mozilla Data Collective key in `.env` as `COMMONVOICE_APIKEY=...` or set that environment variable before running the script. The parser also accepts spaced `.env` assignments such as `COMMONVOICE_APIKEY = ...`.

Mozilla Data Collective requires accepting the terms for each dataset before the API will issue a download URL. If the script reports a terms error, open the dataset URL in the message while signed in, accept the terms, and rerun the script.

The official API currently returns full locale `.tar.gz` archives. The script caches those archives under `overnight_runs_cv/common_voice_archives/` and resumes interrupted downloads with HTTP range requests. If the connection drops, rerun the same command; it resumes the `.part` archive instead of restarting from byte zero. Set `CV_ARCHIVE_CACHE=/path/with/space` if the default run directory is not large enough for the official archives.

The default API dataset IDs cover the scripted-speech 26.0 `ru`, `en`, and `zh-CN` archives used by the calibration script. For other languages or releases, pass explicit MDC IDs:

```bash
CV_DATASET_IDS=ru=...,en=...,zh-CN=... ./run_common_voice_calibration.sh
```

For the English slice, the overnight script defaults to native-labeled Common Voice accents only: `United States English`, `England English`, `Canadian English`, `Australian English`, `New Zealand English`, `Scottish English`, `Irish English`, and `Welsh English`. This excludes blank and non-native English accent labels such as `Nepalese`. Override it with `CV_ACCENT_FILTERS='en=Label|Label'`, or disable accent filtering with `CV_ACCENT_FILTERS=`.

If you already downloaded a local slice before changing the filter, rebuild it with:

```bash
CV_FORCE_COMMON_VOICE_DOWNLOAD=1 ./run_common_voice_calibration.sh
```

The Common Voice config generator also filters out very short targets by default in the overnight script (`CV_MIN_TARGET_CHARS=4`). This avoids backend crashes on one-token targets such as `six` or `六`. If a run already produced configs with shorter targets, regenerate configs before rerunning:

```bash
CV_FORCE_CONFIG=1 ./run_common_voice_calibration.sh
```

If you need to recompute completed model outputs after changing dataset filters, also set:

```bash
CV_FORCE_CONFIG=1 CV_FORCE_RUN=1 ./run_common_voice_calibration.sh
```

Manual local corpora remain supported. To skip the API downloader, point `CV_LOCAL_ROOT` at a directory containing locale folders such as `en/validated.tsv` and `en/clips/`, then set `CV_DOWNLOAD_COMMON_VOICE=0`.

The underlying config generator can also be used directly:

```bash
uv run python xttslab.py dataset common-voice \
  --local-root /data/cv-corpus-21.0-2025-03-14 \
  --languages ru:ru,en:en,zh-CN:zh \
  --split validated \
  --voices-per-language 5 \
  --utterances-per-speaker 2 \
  --targets-per-language 10 \
  --model-id dummy_tts \
  --model-backend dummy \
  --out overnight_runs_cv/config_dummy.toml
```

Useful knobs:

```bash
CV_SPEAKERS_PER_LANGUAGE=8 \
CV_UTTERANCES_PER_SPEAKER=3 \
CV_TARGETS_PER_LANGUAGE=10 \
CV_LANGUAGES=ru:ru,en:en,zh-CN:zh \
./run_common_voice_calibration.sh
```

If your downloaded corpus uses `dev.tsv` or `train.tsv` instead of `validated.tsv`, set `CV_SPLIT=dev` or `CV_SPLIT=train`.

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
- **Config generation**: `run_fleurs_experiment_example.sh`, producing per-model `overnight_runs/config_*.toml` files
- **Languages**: English, Russian, Mandarin Chinese
- **Jobs per full direction**: 100
- **ASR/LID backend**: faster-whisper (medium/small depending on VRAM)
- **Speaker similarity**: SpeechBrain ECAPA-TDNN (`speechbrain/spkrec-ecapa-voxceleb`)
- **Confidence intervals**: 95% bootstrap (1000 resamples, seed 20260628)
- **Hardware**: CUDA-enabled GPUs (e.g., 12GB+ VRAM class)
- **Subset construction**: deterministic first rows after language and length filtering

## Benchmark Results on Google FLEURS

The ASR evaluation uses target-language specific text-normalization adapters (preprocessors) to clean reference and hypothesis transcriptions (handling lowercase, removing punctuation, and stripping spaces for CJK characters) before computing WER/CER. 

The benchmark harness is being evaluated on a cross-lingual subset of the Google FLEURS dataset generated by `run_fleurs_experiment_example.sh` across several state-of-the-art zero-shot voice cloning models. The real metrics stack includes `faster_whisper_asr` for ASR error (measuring target-language intelligibility), `faster_whisper_lid` for a conservative target-language identification score (detected-language confidence when the detected language matches the target, otherwise 0), `speechbrain_speaker_similarity` (ECAPA-TDNN) for speaker-similarity preservation, and `speechbrain_language_similarity` (VoxLingua107) to measure source-language leakage.

Below is the comparative summary of the cross-lingual generalization capabilities of the installed models.

*Note: Lower ASR error indicates better target-text intelligibility under the chosen ASR and normalization pipeline. Higher Target LID score indicates the model was detected as the target language with high confidence. Higher Speaker Sim indicates stronger speaker-embedding similarity to the reference. Higher Leakage indicates the generated audio sounds more like the source language's accent/prosody.*

### Table 1: Common Target-Language Subset
*Only `en` and `zh` target conditions. Excludes target-Russian directions to avoid unsupported/degraded model conditions. F5-TTS target-Russian results are reported in Table 4 for transparency but excluded from this table because the base F5 model is not expected to handle Russian target synthesis reliably.*

| Model | n | ASR Error ↓ (95% CI) | Target LID score ↑ (95% CI) | Speaker Sim ↑ (95% CI) |
|---|---|---|---|---|
| Qwen3-TTS 1.7B | 400 | 7.0% [5.9–8.1] | 96.1% [95.4–96.7] | 0.515 [0.503–0.528] |
| Qwen3-TTS 0.6B | 397 | 7.8% [6.6–9.0] | 94.5% [93.5–95.3] | 0.516 [0.504–0.529] |
| XTTS v2 | 400 | 9.6% [8.1–11.3] | 97.0% [95.8–98.0] | 0.468 [0.455–0.479] |
| Spark-TTS | 400 | 11.9% [10.3–13.5] | 96.0% [94.9–96.9] | 0.420 [0.408–0.432] |
| CosyVoice | 400 | 17.9% [15.4–20.5] | 74.7% [71.5–78.0] | 0.688 [0.674–0.701] |
| F5-TTS | 400 | 31.8% [28.1–35.5] | 83.1% [79.9–86.3] | 0.530 [0.508–0.550] |

### Table 2: Target-Language Aggregates
*Aggregated by target language across all sources.*

| Model | Target | n | ASR Error ↓ (95% CI) | Target LID score ↑ (95% CI) | Speaker Sim ↑ (95% CI) |
|---|---|---|---|---|---|
| Qwen3-TTS 1.7B | en | 200 | 3.9% [3.0–4.9] | 93.8% [92.6–94.9] | 0.556 [0.540–0.571] |
| Qwen3-TTS 0.6B | en | 199 | 5.1% [3.8–6.2] | 90.8% [89.0–92.3] | 0.571 [0.557–0.584] |
| XTTS v2 | en | 200 | 3.7% [2.7–4.6] | 97.1% [96.7–97.5] | 0.497 [0.485–0.509] |
| Spark-TTS | en | 200 | 4.8% [3.6–6.3] | 93.6% [92.2–95.0] | 0.433 [0.420–0.445] |
| CosyVoice | en | 200 | 12.2% [9.7–14.8] | 72.7% [68.9–76.6] | 0.719 [0.704–0.734] |
| F5-TTS | en | 200 | 13.1% [10.8–15.7] | 94.6% [93.6–95.5] | 0.610 [0.596–0.622] |
| Qwen3-TTS 1.7B | ru | 200 | 1.5% [1.0–2.1] | 97.0% [96.4–97.5] | 0.479 [0.459–0.499] |
| Qwen3-TTS 0.6B | ru | 194 | 4.1% [3.1–5.3] | 98.3% [97.9–98.6] | 0.449 [0.428–0.473] |
| XTTS v2 | ru | 200 | 7.5% [6.0–9.4] | 97.5% [96.0–98.7] | 0.456 [0.435–0.479] |
| Spark-TTS | ru | 0 | - | - | - |
| CosyVoice | ru | 200 | 53.3% [47.1–59.9] | 31.4% [25.9–37.1] | 0.731 [0.713–0.748] |
| F5-TTS | ru | 200 | 131.9% [122.3–142.3] | 0.0% [0.0–0.0] | 0.526 [0.482–0.567] |
| Qwen3-TTS 1.7B | zh | 200 | 10.1% [8.2–11.9] | 98.5% [98.2–98.7] | 0.474 [0.456–0.491] |
| Qwen3-TTS 0.6B | zh | 198 | 10.6% [8.9–12.7] | 98.2% [97.8–98.6] | 0.460 [0.441–0.479] |
| XTTS v2 | zh | 200 | 15.5% [12.7–18.5] | 96.8% [94.4–98.7] | 0.439 [0.421–0.456] |
| Spark-TTS | zh | 200 | 19.1% [16.7–21.4] | 98.3% [96.8–99.4] | 0.408 [0.387–0.427] |
| CosyVoice | zh | 200 | 23.5% [19.7–27.4] | 76.8% [71.8–81.9] | 0.657 [0.634–0.678] |
| F5-TTS | zh | 200 | 50.5% [45.0–55.8] | 71.6% [65.8–77.6] | 0.451 [0.412–0.486] |

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

**Interpretation:** Source-language aggregation reveals that ECAPA speaker similarity depends strongly on the reference language, with English references producing remarkably lower similarity across several models compared to Chinese or Russian references. The Common Voice calibration table below now provides real-real same-speaker and different-speaker bounds for interpreting those scores instead of relying on FLEURS-only proxies.

### Table 4: Per-Direction Breakdowns
*Provides full visibility into specific language pairs, exposing asymmetric performance.*

| Model | Direction | n | ASR Error ↓ (95% CI) | Target LID score ↑ (95% CI) | Speaker Sim ↑ (95% CI) |
|---|---|---|---|---|---|
| Qwen3-TTS 1.7B | en->ru | 100 | 1.4% [0.8–2.2] | 95.5% [94.6–96.4] | 0.365 [0.343–0.386] |
| Qwen3-TTS 1.7B | en->zh | 100 | 6.6% [4.4–9.1] | 97.6% [97.1–98.0] | 0.394 [0.373–0.413] |
| Qwen3-TTS 1.7B | ru->en | 100 | 4.4% [3.1–5.7] | 92.8% [90.7–94.5] | 0.545 [0.523–0.567] |
| Qwen3-TTS 1.7B | ru->zh | 100 | 13.6% [10.5–16.5] | 99.4% [99.3–99.4] | 0.555 [0.536–0.574] |
| Qwen3-TTS 1.7B | zh->en | 100 | 3.4% [2.3–4.8] | 94.7% [93.2–95.9] | 0.567 [0.545–0.587] |
| Qwen3-TTS 1.7B | zh->ru | 100 | 1.6% [0.9–2.3] | 98.5% [98.2–98.8] | 0.592 [0.578–0.606] |
| Qwen3-TTS 0.6B | en->ru | 96 | 5.8% [3.9–7.8] | 97.2% [96.6–97.8] | 0.327 [0.305–0.350] |
| Qwen3-TTS 0.6B | en->zh | 98 | 7.5% [5.1–9.9] | 97.1% [96.3–97.7] | 0.369 [0.348–0.390] |
| Qwen3-TTS 0.6B | ru->en | 100 | 5.1% [3.5–7.1] | 93.0% [91.3–94.6] | 0.548 [0.524–0.570] |
| Qwen3-TTS 0.6B | ru->zh | 100 | 13.7% [11.0–16.4] | 99.3% [99.2–99.4] | 0.550 [0.532–0.568] |
| Qwen3-TTS 0.6B | zh->en | 99 | 5.0% [3.3–6.9] | 88.5% [85.4–90.9] | 0.595 [0.579–0.611] |
| Qwen3-TTS 0.6B | zh->ru | 98 | 2.5% [1.5–3.5] | 99.2% [99.1–99.4] | 0.569 [0.554–0.584] |
| XTTS v2 | en->ru | 100 | 8.9% [5.8–12.2] | 96.2% [93.1–98.5] | 0.336 [0.313–0.361] |
| XTTS v2 | en->zh | 100 | 17.9% [13.1–23.0] | 94.1% [89.3–98.0] | 0.368 [0.341–0.392] |
| XTTS v2 | ru->en | 100 | 3.5% [2.4–4.8] | 97.0% [96.5–97.4] | 0.447 [0.434–0.460] |
| XTTS v2 | ru->zh | 100 | 13.1% [10.3–15.7] | 99.5% [99.4–99.5] | 0.509 [0.492–0.527] |
| XTTS v2 | zh->en | 100 | 3.8% [2.3–5.4] | 97.2% [96.5–97.8] | 0.547 [0.533–0.562] |
| XTTS v2 | zh->ru | 100 | 6.2% [4.9–7.6] | 98.9% [98.8–99.1] | 0.577 [0.565–0.588] |
| Spark-TTS | en->ru | 0 | - | - | - |
| Spark-TTS | en->zh | 100 | 25.5% [21.8–29.0] | 97.3% [94.3–99.4] | 0.319 [0.296–0.342] |
| Spark-TTS | ru->en | 100 | 5.4% [3.6–7.9] | 93.2% [91.2–94.7] | 0.422 [0.402–0.441] |
| Spark-TTS | ru->zh | 100 | 12.7% [10.1–15.3] | 99.4% [99.3–99.5] | 0.497 [0.478–0.516] |
| Spark-TTS | zh->en | 100 | 4.3% [3.0–6.0] | 94.1% [91.6–96.0] | 0.443 [0.426–0.462] |
| Spark-TTS | zh->ru | 0 | - | - | - |
| CosyVoice | en->ru | 100 | 39.0% [33.0–45.1] | 41.3% [33.1–48.8] | 0.675 [0.645–0.701] |
| CosyVoice | en->zh | 100 | 14.3% [9.8–19.5] | 92.5% [87.6–96.1] | 0.577 [0.547–0.604] |
| CosyVoice | ru->en | 100 | 11.1% [8.2–14.6] | 62.9% [56.0–69.3] | 0.708 [0.686–0.730] |
| CosyVoice | ru->zh | 100 | 32.7% [27.1–38.8] | 61.2% [52.5–69.8] | 0.737 [0.714–0.757] |
| CosyVoice | zh->en | 100 | 13.2% [9.6–17.5] | 82.5% [79.0–85.5] | 0.731 [0.708–0.749] |
| CosyVoice | zh->ru | 100 | 67.7% [57.3–78.7] | 21.5% [14.3–29.8] | 0.787 [0.767–0.803] |
| F5-TTS | en->ru | 100 | 117.5% [107.8–128.7] | 0.0% [0.0–0.0] | 0.331 [0.270–0.393] |
| F5-TTS | en->zh | 100 | 55.4% [46.8–64.6] | 49.3% [39.4–58.2] | 0.301 [0.246–0.357] |
| F5-TTS | ru->en | 100 | 22.5% [18.5–26.7] | 93.8% [92.5–94.8] | 0.562 [0.543–0.581] |
| F5-TTS | ru->zh | 100 | 45.6% [39.8–52.5] | 93.9% [90.1–97.0] | 0.600 [0.583–0.617] |
| F5-TTS | zh->en | 100 | 3.7% [2.3–5.2] | 95.5% [94.0–96.8] | 0.658 [0.643–0.673] |
| F5-TTS | zh->ru | 100 | 146.3% [131.4–162.4] | 0.0% [0.0–0.0] | 0.721 [0.710–0.730] |

**Interpretation:** Aggregate averages hide severe model-specific and direction-specific failures. Cross-lingual zero-shot voice cloning is highly direction-dependent. For example, while F5-TTS achieves an impressive 3.7% ASR Error on `zh->en`, it completely fails on `*->ru`. CosyVoice struggles with intelligibility in most cross-lingual pairs (e.g., 67.7% ASR Error for `zh->ru`), despite scoring the highest speaker similarity.

### Table 5: Pareto Ranking
Better intelligibility / target-language transfer does **not** imply better speaker preservation. This tradeoff is evident across the models:

- **Best intelligibility**: Qwen3-TTS 1.7B
- **Best target LID**: XTTS v2
- **Best speaker similarity**: CosyVoice
- **Best small model tradeoff**: Qwen3-TTS 0.6B

### Table 6: Normalized Source-Language Leakage (Delta)
*Difference between generated audio's cosine similarity to the source-language centroid vs the target-language centroid. Higher delta (> 0) means the audio sounds more like the source language than the target language.*

| Model | Direction | n | Leakage Delta ↓ (95% CI) |
|---|---|---|---|
| F5-TTS | en->ru | 100 | 0.070 [0.062–0.079] |
| F5-TTS | en->zh | 100 | -0.023 [-0.032–-0.015] |
| F5-TTS | ru->en | 100 | -0.061 [-0.070–-0.052] |
| F5-TTS | ru->zh | 100 | -0.054 [-0.060–-0.047] |
| F5-TTS | zh->en | 100 | -0.051 [-0.057–-0.045] |
| F5-TTS | zh->ru | 100 | 0.109 [0.105–0.113] |
| CosyVoice | en->ru | 100 | 0.045 [0.035–0.056] |
| CosyVoice | en->zh | 100 | -0.034 [-0.039–-0.028] |
| CosyVoice | ru->en | 100 | 0.001 [-0.008–0.009] |
| CosyVoice | ru->zh | 100 | 0.018 [0.011–0.024] |
| CosyVoice | zh->en | 100 | -0.008 [-0.013–-0.003] |
| CosyVoice | zh->ru | 100 | 0.053 [0.045–0.061] |
| Qwen3-TTS 0.6B | en->ru | 96 | -0.127 [-0.132–-0.121] |
| Qwen3-TTS 0.6B | en->zh | 98 | -0.075 [-0.079–-0.071] |
| Qwen3-TTS 0.6B | ru->en | 100 | -0.076 [-0.084–-0.070] |
| Qwen3-TTS 0.6B | ru->zh | 100 | -0.086 [-0.091–-0.081] |
| Qwen3-TTS 0.6B | zh->en | 99 | -0.032 [-0.037–-0.027] |
| Qwen3-TTS 0.6B | zh->ru | 98 | -0.109 [-0.114–-0.104] |
| Qwen3-TTS 1.7B | en->ru | 100 | -0.130 [-0.137–-0.124] |
| Qwen3-TTS 1.7B | en->zh | 100 | -0.071 [-0.075–-0.067] |
| Qwen3-TTS 1.7B | ru->en | 100 | -0.094 [-0.103–-0.086] |
| Qwen3-TTS 1.7B | ru->zh | 100 | -0.088 [-0.093–-0.083] |
| Qwen3-TTS 1.7B | zh->en | 100 | -0.053 [-0.058–-0.048] |
| Qwen3-TTS 1.7B | zh->ru | 100 | -0.116 [-0.121–-0.111] |
| Spark-TTS | en->zh | 100 | -0.050 [-0.054–-0.045] |
| Spark-TTS | ru->en | 100 | -0.107 [-0.114–-0.101] |
| Spark-TTS | ru->zh | 100 | -0.078 [-0.082–-0.073] |
| Spark-TTS | zh->en | 100 | -0.035 [-0.039–-0.030] |
| XTTS v2 | en->ru | 100 | -0.107 [-0.116–-0.100] |
| XTTS v2 | en->zh | 100 | -0.063 [-0.067–-0.059] |
| XTTS v2 | ru->en | 100 | -0.099 [-0.105–-0.092] |
| XTTS v2 | ru->zh | 100 | -0.085 [-0.089–-0.080] |
| XTTS v2 | zh->en | 100 | -0.032 [-0.038–-0.025] |
| XTTS v2 | zh->ru | 100 | -0.085 [-0.092–-0.078] |

**Interpretation:** The relative leakage probe reveals a critical tradeoff in CosyVoice: its consistently high "Speaker Similarity" (from Table 1/2) is directly correlated with high source-language leakage (often delta > 0, meaning it sounds closer to the source language than the target language). It achieves high speaker embedding scores by refusing to fully adapt to target-language phonetics, explaining its poor intelligibility. In contrast, Qwen3-TTS successfully shifts its audio distribution toward the target language (delta < 0) while maintaining strong intelligibility, suggesting better separation between voice identity and source-language acoustic cues under this probe.

### Leakage Metric Caveat
The current leakage score is an embedding-based proxy using VoxLingua107 space normalized against FLEURS language centroids. While directional trends are clear, future work will validate it against human accent/prosody judgments.

### Table 7: Speaker-Similarity Calibration
*Speaker similarity requires calibration against ground-truth positive/negative bounds to fully disentangle voice preservation from channel or language artifacts. The following bounds are extracted from `overnight_runs_cv/`, using known repeated Common Voice `client_id`/`speaker_id` values rather than inferred FLEURS pseudo-pairs:*

| Pair type | Speaker Sim |
|---|---|
| same speaker real-real (known speaker ID) | 0.635 ± 0.129 (n=15) |
| different speaker same language | 0.104 ± 0.104 (n=120) |
| different speaker cross-language | 0.081 ± 0.090 (n=300) |
| generated vs wrong reference, F5-TTS | 0.042 ± 0.075 (n=600) |
| generated vs wrong reference, Qwen3-TTS 0.6B | 0.059 ± 0.068 (n=600) |
| generated vs wrong reference, Qwen3-TTS 1.7B | 0.052 ± 0.063 (n=600) |
| generated vs wrong reference, XTTS v2 | 0.064 ± 0.070 (n=600) |
| generated vs wrong reference, CosyVoice | 0.078 ± 0.082 (n=600) |
| generated vs wrong reference, Spark-TTS | 0.058 ± 0.079 (n=400) |

Note: same-speaker cross-language calibration is still `N/A` in this Common Voice slice because the available repeated speaker IDs are within locale, not across languages.

**Interpretation:** The calibration matrix now gives a real-real same-speaker bound from known Common Voice IDs: ECAPA-TDNN places repeated same-speaker utterances around ~0.64, while different-speaker pairs sit near ~0.10 or below. The `generated vs wrong reference` checks remain close to those negative bounds across all models, which is a useful sanity check against trivial score inflation. CosyVoice’s high FLEURS speaker similarity should therefore be treated as a plausible voice-preservation signal relative to calibrated bounds, not as proof of ground-truth identity preservation.

### Future Work: TASLP Methodological Improvements
To rigorously validate speaker similarity and phonetic disentanglement for peer-reviewed publication, the following validation remains:

| Needed | Why |
|---|---|
| human accent/nativeness labels | validates the leakage proxy |

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
