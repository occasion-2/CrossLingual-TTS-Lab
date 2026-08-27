# CrossLingual TTS Lab

CrossLingual TTS Lab is a benchmark harness for cross-lingual zero-shot voice cloning. It tracks target-text intelligibility, target-language identification, speaker similarity, and a source-language leakage proxy.

Core question:

> When a reference voice is in one language and target text is in another, does
> the model preserve speaker identity without leaking source-language accent,
> phonetics, or prosody into the target language?

The project is a `uv` workspace with a config-driven planner, model and metric backends, resumable synthesis, and JSON/Markdown reports. It includes dummy and real TTS backends, ASR/LID/speaker metrics, and an embedding-space leakage probe. SER-based emotion metrics are not implemented yet.

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
  --model-param revision=fd4b254389122332181a7c3db7f27e918eec64e3 \
  --model-param ref_text_mode=empty \
  --model-param x_vector_only_mode=true \
  --model-param dtype=bfloat16 \
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
- `speechbrain_language_similarity` for the language-centroid leakage proxy

## GPU Profile

Check what the runner sees:

```bash
uv run python xttslab.py doctor
```

The generated scientific configs pin a float16 `medium` Whisper-family ASR model on CUDA
and an immutable `small`/int8 checkpoint for initial CPU execution or CUDA fallback.
Hand-written configs that omit `model_size` still use the device profile's VRAM-based
recommendation. The project
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

To run the multi-model FLEURS experiment, use the wrapper script. It keeps each model in its own virtual environment because the model stacks require different CUDA/PyTorch packages.

1. **Install system prerequisites** (e.g. `sox` is required by Qwen-TTS):
   ```bash
   sudo apt-get install sox libsox-fmt-all
   ```

2. **Install external repositories and weights**:
   ```bash
   ./install_dependencies.sh
   ```
    This clones the `CosyVoice` and `Spark-TTS` repositories at the paper-snapshot commits and downloads the pinned Spark-TTS 0.5B checkpoint. Existing checkouts are preserved; installation fails with a corrective message when their HEAD differs from the snapshot.

3. **Install Python dependencies for specific backends**:
   Backend dependencies are split into optional extras in [pyproject.toml](pyproject.toml). Install only the stack you need if you are running a single model:

   * **Using the helper script**:
     - For CosyVoice: `./install_dependencies.sh --cosyvoice`
     - For Spark-TTS: `./install_dependencies.sh --spark-tts`

   * **Installing manually**:
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
   This creates model-specific virtual environments under `overnight_runs/`, generates FLEURS configs with the checkpoint revisions and inference settings recorded in [`configs/paper_model_snapshot.toml`](configs/paper_model_snapshot.toml), synthesizes audio, and scores the metrics.

5. **WAV-level resumability**:
   If a run stops, rerun `./run_fleurs_experiment_example.sh`. Completed models are skipped when `report.md` exists, and existing WAV files are reused.

### Common Voice Speaker Calibration Run

FLEURS is the main direction-aware benchmark used in the paper tables. Its speaker labels are not enough for a ground-truth same-speaker calibration, so the speaker-similarity bounds come from a Common Voice companion run:

```bash
./run_common_voice_calibration.sh
```

This builds Common Voice configs with repeated reference utterances per known `client_id`/`speaker_id`, runs the model stack under `overnight_runs_cv/`, and writes `calibration.md` for each model. The calibration command uses known repeated speaker IDs when they are present. The older inferred FLEURS fallback remains for legacy runs.

Calibration pins `speechbrain/spkrec-ecapa-voxceleb@0f99f2d0ebe89ac095bcc5903c4dd8f72b367286` by default and records the checkpoint/device in each report. The `calibrate` command also accepts `--model-id`, `--model-revision`, and `--device` for an explicitly different setup.

Current `overnight_runs_cv/` snapshot: each full model run contains 600 cross-lingual jobs from 30 Common Voice prompts and 30 targets. F5-TTS, Qwen3-TTS 0.6B, Qwen3-TTS 1.7B, XTTS v2, and CosyVoice all completed 600 scored samples. Spark-TTS completed the 400 supported English/Chinese-target samples and records 200 expected placeholders for target-Russian directions.

The calibration bounds now come from repeated Common Voice speaker IDs rather than inferred FLEURS pseudo-pairs:

| Pair type | Speaker Sim (mean ± population SD) |
|---|---|
| same speaker real-real (known speaker ID) | 0.635 ± 0.129 (n=15) |
| different speaker same language | 0.104 ± 0.104 (n=120) |
| different speaker cross-language | 0.081 ± 0.090 (n=300) |

The generated-vs-wrong-reference checks from the same run (reported as mean ± population SD, not confidence intervals) are close to the different-speaker bounds: F5-TTS 0.042 ± 0.075 (n=600), Qwen3-TTS 0.6B 0.059 ± 0.068 (n=600), Qwen3-TTS 1.7B 0.052 ± 0.063 (n=600), XTTS v2 0.064 ± 0.070 (n=600), CosyVoice 0.078 ± 0.082 (n=600), and Spark-TTS 0.058 ± 0.079 (n=400).

Common Voice is no longer usable through the old Hugging Face placeholder repos. The companion script now uses the official Mozilla Data Collective API to fetch the requested locale archives, then extracts only the selected `validated.tsv` rows and clips into `overnight_runs_cv/common_voice/`. Put your Mozilla Data Collective key in `.env` as `COMMONVOICE_APIKEY=...` or set that environment variable before running the script. The parser also accepts spaced `.env` assignments such as `COMMONVOICE_APIKEY = ...`.

Mozilla Data Collective requires accepting the terms for each dataset before the API will issue a download URL. If the script reports a terms error, open the dataset URL in the message while signed in, accept the terms, and rerun the script.

The official API currently returns full locale `.tar.gz` archives. The script caches them under `overnight_runs_cv/common_voice_archives/` and resumes interrupted downloads with HTTP range requests. If the connection drops, rerun the same command; it continues from the `.part` archive. Set `CV_ARCHIVE_CACHE=/path/with/space` if the default run directory is too small.

The default API dataset IDs cover the scripted-speech 26.0 `ru`, `en`, and `zh-CN` archives used by the calibration script. For other languages or releases, pass explicit MDC IDs:

```bash
CV_DATASET_IDS=ru=...,en=...,zh-CN=... ./run_common_voice_calibration.sh
```

For the English slice, the overnight script defaults to native-labeled Common Voice accents only: `United States English`, `England English`, `Canadian English`, `Australian English`, `New Zealand English`, `Scottish English`, `Irish English`, and `Welsh English`. This excludes blank and non-native English accent labels such as `Nepalese`. Override it with `CV_ACCENT_FILTERS='en=Label|Label'`, or disable accent filtering with `CV_ACCENT_FILTERS=`.

If you already downloaded a local slice before changing the filter, rebuild it with:

```bash
CV_FORCE_COMMON_VOICE_DOWNLOAD=1 ./run_common_voice_calibration.sh
```

The Common Voice config generator filters out very short targets by default in the overnight script (`CV_MIN_TARGET_CHARS=4`). This avoids known backend failures on one-token targets such as `six` or `六`. If a run already produced configs with shorter targets, regenerate configs before rerunning:

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
params = { model_size = "medium", model_revision = "08e178d48790749d25932bbc082711ddcfdfbc4f", vad_filter = true, beam_size = 5, cpu_model_size = "small", cpu_model_revision = "536b0662742c02347bc0e980a01041f333bce120", cpu_compute_type = "int8" }

[[metrics]]
id = "target_language_id"
backend = "faster_whisper_lid"
params = { model_size = "medium", model_revision = "08e178d48790749d25932bbc082711ddcfdfbc4f", vad_filter = true, cpu_model_size = "small", cpu_model_revision = "536b0662742c02347bc0e980a01041f333bce120", cpu_compute_type = "int8" }

[[metrics]]
id = "speaker_similarity"
backend = "speechbrain_speaker_similarity"
params = { model_id = "speechbrain/spkrec-ecapa-voxceleb", model_revision = "0f99f2d0ebe89ac095bcc5903c4dd8f72b367286" }

[[metrics]]
id = "source_language_similarity"
backend = "speechbrain_language_similarity"
params = { model_id = "speechbrain/lang-id-voxlingua107-ecapa", model_revision = "0253049ae131d6a4be1c4f0d8b0ff483a0f8c8e9" }
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
  mini.toml                  # example benchmark
  paper_model_snapshot.toml  # best-available paper-run artifact/configuration provenance
compute_stats.py             # crossed-bootstrap paper statistics and table generator
generated_tables.md          # canonical generated paper-result tables
tests/
  test_config_and_runner.py  # test suite for config, dataset, and runner logic
  test_compute_stats.py      # statistics and clustered-bootstrap regression tests
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
- **Design per full direction**: 10 reference utterances crossed with 10 target texts, producing 100 generations
- **ASR/LID backend**: `Systran/faster-whisper-medium@08e178d4` (CUDA/float16), with pinned `Systran/faster-whisper-small@536b0662` CPU/int8 fallback
- **Speaker similarity**: SpeechBrain ECAPA-TDNN (`speechbrain/spkrec-ecapa-voxceleb@0f99f2d0`)
- **Leakage encoder**: SpeechBrain VoxLingua107 (`speechbrain/lang-id-voxlingua107-ecapa@0253049a`); checked-in centroid SHA-256 `9adca9f8...4373d91`
- **Confidence intervals**: 95% stratified crossed reference/text bootstrap intervals (1000 resamples, seed 20260628)
- **Successful sample**: correct target-language identification and ASR error strictly below 10%
- **Model snapshot**: [`configs/paper_model_snapshot.toml`](configs/paper_model_snapshot.toml)
- **Hardware snapshot**: Python 3.11, Torch 2.11.0+cu130, NVIDIA GeForce RTX 4080 Laptop GPU; see the model snapshot for backend-specific provenance caveats
- **Subset construction**: deterministic first rows after language and length filtering

The model snapshot is the best-available provenance record for the paper run's synthesis and evaluator checkpoint IDs, reconstructed immutable hub revisions or surviving local hashes, package or source-repository versions, inference settings, sample rates, and seed status. Every synthesis model and evaluator has explicit `provenance_status`, `provenance_source`, and `provenance_note` fields so reconstructed values are not presented as run-attested facts. The current FLEURS and Common Voice wrappers pin or verify those artifacts for new runs. Several historical systems used stochastic inference without a fixed seed, and the legacy manifests did not preserve a complete environment lock; the historical waveforms are therefore not bit-reproducible and reconstructed revisions are not original-run attestations. XTTS also has mixed device metadata from a resumed run, so the snapshot does not claim one run-attested device for every XTTS sample.

The manifests attest the evaluator aliases and actual per-sample execution modes. Most analyzed ASR/LID rows used faster-whisper `medium` on CUDA/float16, but 358/600 Qwen3-TTS 0.6B LID rows and 200/399 analyzed Spark-TTS ASR rows fell back after CUDA errors to `small` on CPU/int8. The immutable evaluator revisions and package versions in the snapshot were reconstructed afterward. This scoring heterogeneity is a limitation of the historical results; a new comparison should rescore every WAV with one pinned evaluator configuration.

### Evaluator-only paper rescore

First validate the complete paper subset and both generated/reference WAV inventories without loading evaluator models:

```bash
.venv/bin/python -u rescore_paper_evaluators.py \
  --source-root overnight_runs \
  --output-root paper_evaluator_rescore \
  --dry-run \
  --skip-runtime-checks
```

Then omit the two dry-run flags and add `--resume` to score on CUDA. The profile requires faster-whisper `medium` at the pinned revision with CUDA/float16 (ASR beam 5, LID beam 1), disables CPU fallback, and pins both SpeechBrain evaluator revisions. It writes new manifests and reports under `paper_evaluator_rescore/`; historical `overnight_runs/results_*/manifest.json` files and WAVs remain read-only inputs. Existing destination manifests are refused unless `--overwrite` is explicitly supplied. The completed manifests record the evaluator profile, exact package versions, source-manifest hashes, and separate generated/reference WAV inventory hashes.

Each of the four metrics runs in its own spawned process, so the two faster-whisper models and two SpeechBrain models never remain resident on the GPU together. A completed pass is strictly validated and attested under `paper_evaluator_rescore/results_*/.metric-passes/` before the next process starts; process exit provides a hard CUDA-memory teardown. The four pass results are merged by job ID in canonical metric order, fully validated, and atomically installed as the model manifest. Audio is never truncated and evaluator errors do not trigger CPU fallback.

If scoring is interrupted, rerun the same command with `--resume`. The script reuses completed model manifests and exact, validated per-metric pass manifests; an invalid or failed pass is rerun without discarding earlier valid passes. `--resume` and `--overwrite` are mutually exclusive. Do not use plain `uv run` for this command because it may attempt to resolve unrelated synthesis extras such as FlashAttention; if the existing environment must be launched through uv, use `uv run --no-sync rescore_paper_evaluators.py ...`.

## Benchmark Results on Google FLEURS

ASR evaluation uses target-language text normalizers before computing WER/CER. The normalizers lowercase where appropriate, remove punctuation, and strip spaces for CJK text.

The canonical paper tables are generated from the FLEURS manifests under `overnight_runs/`. The metrics are:

- `faster_whisper_asr` for target-text ASR error
- `faster_whisper_lid` for a conservative target-language score: detected-language confidence if the detected language matches the target, otherwise 0
- `speechbrain_speaker_similarity` for ECAPA-TDNN speaker similarity
- `speechbrain_language_similarity` for the VoxLingua107 leakage proxy

The numbers are automatic metrics, not human preference or naturalness judgments.

Lower ASR error is better. Higher Target LID score means the output was more confidently detected as the target language. Higher Speaker Sim means higher embedding similarity to the reference. Higher leakage delta means the generated audio is closer to the source-language centroid than to the target-language centroid under this proxy.

### Regenerate the canonical tables

Run the statistics script against the FLEURS result root:

```bash
uv run --extra metrics python compute_stats.py \
  --run-root overnight_runs \
  --bootstrap-resamples 1000 \
  --bootstrap-seed 20260628 \
  --success-asr-threshold 0.10 \
  --output generated_tables.md
```

[`generated_tables.md`](generated_tables.md) is the canonical generated result artifact. It contains the common supported-target subset, target/source aggregates, per-direction metrics, direction-level leakage, successful-only leakage, and Pearson/Spearman leakage correlations with LID and ASR. The README intentionally does not duplicate its confidence-interval tables, so regenerated results cannot silently drift away from a second static copy.

The common subset is defined from documented target-language support before looking at model quality. Metrics use their own complete cases: a missing speaker-similarity value does not remove an otherwise valid ASR, LID, or leakage observation.

### Crossed reference/text bootstrap

The 100 samples in a full direction are not independent: each reference utterance is reused across ten texts, and each text is reused across ten references. `compute_stats.py` therefore uses a language-stratified crossed (pigeonhole) bootstrap rather than an iid row bootstrap. Within each replicate it independently resamples reference IDs inside source-language strata and target-text IDs inside target-language strata; an observation receives the product of its two cluster multiplicities. The same procedure supplies the intervals for means and correlations.

### Leakage on strictly successful samples

A sample is successful only when faster-whisper detects the intended target language and its target-language ASR error is strictly below 0.10. The threshold applies to WER for English/Russian and CER for Mandarin. Leakage analysis then requires a valid leakage value as a metric-specific complete case.

Current point estimates from the canonical artifact are summarized below; use [`generated_tables.md`](generated_tables.md) for the clustered confidence intervals and direction-level breakdowns.

| Model | Successful / eligible | Successful leakage delta ↓ | Spearman ρ(Δ, LID) | Spearman ρ(Δ, ASR) |
|---|---:|---:|---:|---:|
| F5-TTS | 144 / 600 | -0.058 | -0.800 | 0.775 |
| CosyVoice | 226 / 600 | -0.013 | -0.648 | 0.485 |
| Qwen3-TTS 0.6B | 456 / 600 | -0.085 | -0.469 | 0.044 |
| Qwen3-TTS 1.7B | 482 / 600 | -0.094 | -0.265 | 0.083 |
| Spark-TTS | 234 / 399 | -0.069 | -0.021 | 0.082 |
| XTTS v2 | 420 / 600 | -0.081 | -0.206 | 0.044 |

The correlations show substantial overlap between the leakage proxy and ASR/LID for failure-prone systems, especially F5-TTS and CosyVoice. The proxy is therefore not claimed to be statistically independent of those metrics. Its model- and direction-level variation remains after conditioning on correct LID and low ASR error, showing that the binary success screen alone does not make the proxy constant. The mixed fallback above makes the Qwen3-TTS 0.6B success and LID-correlation results, and the Spark-TTS success and ASR-correlation results, provisional pending homogeneous rescoring.

### Leakage metric caveat

Leakage delta is a VoxLingua107 language-embedding centroid margin, not a calibrated perceptual accent, phonetics, or prosody score. Human validation is still required for those claims.

## Current Status

Implemented model backends:

- **Qwen3-TTS** (`QwenTTSBackend` / `qwen_tts`)
- **F5-TTS** (`F5TTSBackend` / `f5_tts`)
- **XTTS/Coqui** (`CoquiXTTSBackend` / `coqui_xtts` / `xtts`)
- **CosyVoice** (`CosyVoiceBackend` / `cosyvoice`)
- **Spark-TTS** (`SparkTTSBackend` / `spark_tts`)

Implemented metrics:

- target-text ASR error with faster-whisper
- generated-audio language identification with faster-whisper
- speaker similarity with SpeechBrain ECAPA-TDNN
- source-language leakage proxy with SpeechBrain VoxLingua107 embeddings
- English, Russian, and Chinese (`zh`/`cmn`) text normalizers in [text_metrics.py](src/crosslingual_tts_lab/text_metrics.py)

The model stacks use conflicting CUDA and PyTorch packages, so the FLEURS wrapper script creates one virtual environment per model under `overnight_runs/`.

Not implemented:

- human listening labels for accent, nativeness, and perceived speaker similarity
- optional emotion-preservation metrics from SER models and emotion-labeled subsets
