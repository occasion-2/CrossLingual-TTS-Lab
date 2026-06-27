# CrossLingual TTS Lab

CrossLingual TTS Lab is a small benchmark harness for testing voice-language
disentanglement in multilingual text-to-speech systems.

Core question:

> When a reference voice is in one language and target text is in another, does
> the model preserve speaker identity without leaking source-language accent,
> phonetics, or prosody into the target language?

The first implementation is intentionally lightweight: it gives you a working
`uv` project, a config-driven run planner, a dummy synthesis backend for smoke
tests, pluggable model/metric interfaces, and machine-readable plus Markdown
reports. Heavy integrations such as F5-TTS, CosyVoice, XTTS, speaker embedding
models, ASR, LID, SER, and source-language leakage probes can be added behind the
same interfaces.

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
   All model backend dependencies are cleanly separated into optional-dependencies (extras) inside [pyproject.toml](file:///srv/code/Pet/vleak_inspect/pyproject.toml). You can install the dependencies of your choice directly into your active virtual environment:

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
- source-language leakage placeholder metrics until the probe is added
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

## Benchmark Results on Google FLEURS

The ASR evaluation uses target-language specific text-normalization adapters (preprocessors) to clean reference and hypothesis transcriptions (handling lowercase, removing punctuation, and stripping spaces for CJK characters) before computing WER/CER. 

The benchmark harness was evaluated on a cross-lingual subset of the Google FLEURS dataset (`configs/fleurs_tiny_all.toml`) across several state-of-the-art zero-shot voice cloning models. The real metrics stack includes `faster_whisper_asr` for ASR error (measuring intelligibility and accent leakage), `faster_whisper_lid` for target language identification, and `speechbrain_speaker_similarity` (ECAPA-TDNN) for speaker verification between the source reference and the generated target-language output.

Below is the comparative summary of the cross-lingual generalization capabilities of the installed models:

| Model | Size | ASR WER (Intelligibility) | Target Lang ID Confidence | Speaker Similarity |
|---|---|---|---|---|
| F5-TTS | 385M | 27.6% | 94.8% | 0.595 |
| Qwen3-TTS 0.6B | 600M | 6.8% | 93.0% | 0.615 |
| Qwen3-TTS 1.7B | 1.7B | 6.8% | 97.6% | 0.621 |

*Note: Lower ASR WER indicates better intelligibility and pronunciation in the target language. Higher Target Lang ID indicates the model successfully transitioned to the target language without heavy source-language accent leakage. Higher Speaker Similarity indicates the target-language voice effectively cloned the source speaker's identity.*

## Completed Integrations

1. **Real TTS Model Backends**:
   - **F5-TTS** (`F5TTSBackend` / `f5_tts`): Official Python API integration.
   - **XTTS/Coqui** (`CoquiXTTSBackend` / `coqui_xtts` / `xtts`): Multi-lingual clone support through the `TTS` API (patched for PyTorch 2.6).
   - **CosyVoice** (`CosyVoiceBackend` / `cosyvoice`): Zero-shot cloning through FunASR/CosyVoice python API.
   - **Spark-TTS** (`SparkTTSBackend` / `spark_tts`): Zero-shot cloning through the `SparkTTS` python API.
   *(Note: Because these cutting-edge models have deeply conflicting CUDA and PyTorch dependencies, they cannot coexist in a single environment. The `./run_fleurs_experiment_example.sh` script automatically constructs perfectly isolated virtual environments for each model to safely run them without dependency crashes).*
2. **ASR adapters per target language** to compute WER/CER:
   - English, Russian, and Chinese (`zh`/`cmn`) specific normalizers in [text_metrics.py](file:///srv/code/Pet/vleak_inspect/src/crosslingual_tts_lab/text_metrics.py).

## Next Integrations

The intended next pieces are:

1. Add speaker-verification embeddings for speaker similarity. (Completed - uses SpeechBrain ECAPA-TDNN)
2. Add LID inference on generated audio. (Completed - uses Faster Whisper LID)
3. Add a source-language leakage probe trained on generated audio embeddings while controlling for target language.
4. Add optional emotion preservation metrics from SER models and emotion-labeled subsets.
