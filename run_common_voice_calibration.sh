#!/usr/bin/env bash
set -euo pipefail

# Run a Common Voice benchmark slice with repeated utterances per speaker.
# This complements the FLEURS overnight run: FLEURS remains the main
# direction-aware benchmark, while this script produces stronger speaker
# calibration bounds from known Common Voice speaker IDs.
#
# Tunable environment variables:
#   CV_LANGUAGES=ru:ru,en:en,zh-CN:zh
#   CV_SPLIT=validated
#   CV_SPEAKERS_PER_LANGUAGE=5
#   CV_UTTERANCES_PER_SPEAKER=2
#   CV_TARGETS_PER_LANGUAGE=10
#   CV_MAX_VOICE_CHARS=120
#   CV_MAX_TARGET_CHARS=110
#   CV_MIN_TARGET_CHARS=4
#   CV_RUN_ROOT=overnight_runs_cv
#   CV_LOCAL_ROOT=overnight_runs_cv/common_voice
#   CV_ARCHIVE_CACHE=overnight_runs_cv/common_voice_archives
#   CV_DOWNLOAD_COMMON_VOICE=1
#   CV_FORCE_COMMON_VOICE_DOWNLOAD=0
#   CV_FORCE_CONFIG=0
#   CV_FORCE_RUN=0
#   CV_DATASET_IDS=ru=...,en=...,zh-CN=...
#   CV_ACCENT_FILTERS="en=United States English|England English|Canadian English|Australian English|New Zealand English|Scottish English|Irish English|Welsh English"
#   CV_MDC_API_KEY_ENV=COMMONVOICE_APIKEY
#   CV_ENV_FILE=.env

export COQUI_TOS_AGREED=1
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/.uv-cache}"
export UV_VENV_CLEAR="${UV_VENV_CLEAR:-1}"

CV_LANGUAGES="${CV_LANGUAGES:-ru:ru,en:en,zh-CN:zh}"
CV_SPLIT="${CV_SPLIT:-validated}"
CV_SPEAKERS_PER_LANGUAGE="${CV_SPEAKERS_PER_LANGUAGE:-5}"
CV_UTTERANCES_PER_SPEAKER="${CV_UTTERANCES_PER_SPEAKER:-2}"
CV_TARGETS_PER_LANGUAGE="${CV_TARGETS_PER_LANGUAGE:-10}"
CV_MAX_VOICE_CHARS="${CV_MAX_VOICE_CHARS:-120}"
CV_MAX_TARGET_CHARS="${CV_MAX_TARGET_CHARS:-110}"
CV_MIN_TARGET_CHARS="${CV_MIN_TARGET_CHARS:-4}"
CV_RUN_ROOT="${CV_RUN_ROOT:-overnight_runs_cv}"
CV_LOCAL_ROOT="${CV_LOCAL_ROOT:-$CV_RUN_ROOT/common_voice}"
CV_ARCHIVE_CACHE="${CV_ARCHIVE_CACHE:-$CV_RUN_ROOT/common_voice_archives}"
CV_DOWNLOAD_COMMON_VOICE="${CV_DOWNLOAD_COMMON_VOICE:-1}"
CV_FORCE_COMMON_VOICE_DOWNLOAD="${CV_FORCE_COMMON_VOICE_DOWNLOAD:-0}"
CV_FORCE_CONFIG="${CV_FORCE_CONFIG:-0}"
CV_FORCE_RUN="${CV_FORCE_RUN:-0}"
CV_DATASET_IDS="${CV_DATASET_IDS:-}"
CV_ACCENT_FILTERS="${CV_ACCENT_FILTERS-en=United States English|England English|Canadian English|Australian English|New Zealand English|Scottish English|Irish English|Welsh English}"
CV_MDC_API_BASE="${CV_MDC_API_BASE:-https://mozilladatacollective.com/api}"
CV_MDC_API_KEY_ENV="${CV_MDC_API_KEY_ENV:-COMMONVOICE_APIKEY}"
CV_ENV_FILE="${CV_ENV_FILE:-.env}"

declare -A models
models["f5tts"]="f5tts_v1_base f5_tts model=F5TTS_v1_Base ref_text_mode=empty"
models["qwen0_6b"]="qwen3_tts_0_6b_base qwen_tts model=Qwen/Qwen3-TTS-12Hz-0.6B-Base ref_text_mode=empty"
models["qwen1_7b"]="qwen3_tts_1_7b_base qwen_tts model=Qwen/Qwen3-TTS-12Hz-1.7B-Base ref_text_mode=empty"
models["xtts"]="xtts_v2 coqui_xtts model_name=tts_models/multilingual/multi-dataset/xtts_v2"
models["cosyvoice"]="cosyvoice cosyvoice model_name=FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
models["spark_tts"]="spark_tts spark_tts model_name=pretrained_models/Spark-TTS-0.5B ref_text_mode=empty"

mkdir -p "$CV_RUN_ROOT"
failed_runs=()

if [ "$CV_DOWNLOAD_COMMON_VOICE" != "0" ]; then
    echo "Downloading Common Voice calibration slice via Mozilla Data Collective API..."
    download_args=(
        dataset common-voice-download
        --out-root "$CV_LOCAL_ROOT"
        --split "$CV_SPLIT"
        --languages "$CV_LANGUAGES"
        --speakers-per-language "$CV_SPEAKERS_PER_LANGUAGE"
        --utterances-per-speaker "$CV_UTTERANCES_PER_SPEAKER"
        --targets-per-language "$CV_TARGETS_PER_LANGUAGE"
        --max-voice-chars "$CV_MAX_VOICE_CHARS"
        --max-target-chars "$CV_MAX_TARGET_CHARS"
        --min-target-chars "$CV_MIN_TARGET_CHARS"
        --api-base "$CV_MDC_API_BASE"
        --api-key-env "$CV_MDC_API_KEY_ENV"
        --env-file "$CV_ENV_FILE"
        --archive-cache "$CV_ARCHIVE_CACHE"
    )
    if [ -n "$CV_DATASET_IDS" ]; then
        download_args+=(--dataset-ids "$CV_DATASET_IDS")
    fi
    if [ -n "$CV_ACCENT_FILTERS" ]; then
        download_args+=(--accent-filters "$CV_ACCENT_FILTERS")
    fi
    if [ "$CV_FORCE_COMMON_VOICE_DOWNLOAD" != "0" ]; then
        download_args+=(--force)
    fi
    uv run python xttslab.py "${download_args[@]}"
else
    echo "Skipping Common Voice download; using existing local root."
fi

echo "Common Voice speaker-calibration run"
echo "local root: $CV_LOCAL_ROOT"
echo "archive cache: $CV_ARCHIVE_CACHE"
echo "languages: $CV_LANGUAGES"
echo "split: $CV_SPLIT"
echo "speakers/language: $CV_SPEAKERS_PER_LANGUAGE"
echo "utterances/speaker: $CV_UTTERANCES_PER_SPEAKER"
echo "targets/language: $CV_TARGETS_PER_LANGUAGE"
echo "min target chars: $CV_MIN_TARGET_CHARS"
echo "accent filters: ${CV_ACCENT_FILTERS:-none}"
echo "run root: $CV_RUN_ROOT"

for key in "${!models[@]}"; do
    IFS=' ' read -r id backend params <<< "${models[$key]}"
    echo "======================================"
    echo "Model: $key"

    venv_dir="$CV_RUN_ROOT/.venv_${key}"
    config_path="$CV_RUN_ROOT/config_${key}.toml"
    result_dir="$CV_RUN_ROOT/results_${key}"

    if [ ! -d "$venv_dir" ]; then
        echo "Preparing isolated environment for $key..."
        uv venv "$venv_dir"
    fi

    extras="open-data,metrics"
    if [[ "$key" == *"qwen"* ]]; then
        extras="$extras,qwen"
    elif [[ "$key" == *"f5"* ]]; then
        extras="$extras,f5"
    elif [[ "$key" == "xtts" ]]; then
        extras="$extras,tts"
    elif [[ "$key" == "cosyvoice" ]]; then
        extras="$extras,cosyvoice"
    elif [[ "$key" == "spark_tts" ]]; then
        extras="$extras,spark-tts"
    fi

    no_isolation=""
    if [[ "$key" == "cosyvoice" ]]; then
        echo "Installing build prerequisites and CUDA 12 support libraries for CosyVoice..."
        VIRTUAL_ENV="$venv_dir" uv pip install "setuptools<70" wheel
        VIRTUAL_ENV="$venv_dir" uv pip install "nvidia-cudnn-cu12>=8.9,<9.0.0" "nvidia-cuda-runtime-cu12>=12.9" "nvidia-cufft-cu12>=11.4" "nvidia-curand-cu12>=10.3" "nvidia-cusolver-cu12>=11.7" "nvidia-cusparse-cu12>=12.5"
        no_isolation="--no-build-isolation-package openai-whisper --no-build-isolation-package deepspeed"
    fi

    echo "Installing dependencies [$extras] into $venv_dir..."
    VIRTUAL_ENV="$venv_dir" uv pip install -e ".[$extras]" $no_isolation

    param_args=()
    for param in $params; do
        param_args+=(--model-param "$param")
    done

    nvidia_libs="$(find "$venv_dir" -type d -path "*/site-packages/nvidia/*/lib" | paste -sd : -)"

    if [ "$CV_FORCE_CONFIG" != "0" ] || [ ! -f "$config_path" ]; then
        echo "Generating Common Voice config for $key..."
        LD_LIBRARY_PATH="$nvidia_libs:${LD_LIBRARY_PATH:-}" "$venv_dir/bin/xttslab" dataset common-voice \
            --local-root "$CV_LOCAL_ROOT" \
            --split "$CV_SPLIT" \
            --languages "$CV_LANGUAGES" \
            --voices-per-language "$CV_SPEAKERS_PER_LANGUAGE" \
            --utterances-per-speaker "$CV_UTTERANCES_PER_SPEAKER" \
            --targets-per-language "$CV_TARGETS_PER_LANGUAGE" \
            --max-voice-chars "$CV_MAX_VOICE_CHARS" \
            --max-target-chars "$CV_MAX_TARGET_CHARS" \
            --min-target-chars "$CV_MIN_TARGET_CHARS" \
            --model-id "$id" \
            --model-backend "$backend" \
            "${param_args[@]}" \
            --out "$config_path"
    else
        echo "Reusing existing config: $config_path"
    fi

    if [ "$CV_FORCE_RUN" = "0" ] && [ -f "$result_dir/report.md" ]; then
        echo "Skipping synthesis/scoring for $key: $result_dir/report.md exists"
    else
        if [ "$CV_FORCE_RUN" != "0" ]; then
            echo "Forcing synthesis/scoring rerun for $key..."
            rm -f "$result_dir/manifest.json" "$result_dir/report.md" "$result_dir/calibration.md"
        fi
        if ! LD_LIBRARY_PATH="$nvidia_libs:${LD_LIBRARY_PATH:-}" "$venv_dir/bin/xttslab" run \
        --config "$config_path" \
        --out "$result_dir"; then
            echo "Error: Common Voice benchmark run for $key failed."
            failed_runs+=("$key")
            continue
        fi
    fi

    if [ -f "$result_dir/manifest.json" ] && [ ! -f "$result_dir/calibration.md" ]; then
        echo "Computing known-speaker calibration for $key..."
        LD_LIBRARY_PATH="$nvidia_libs:${LD_LIBRARY_PATH:-}" "$venv_dir/bin/xttslab" calibrate --run "$result_dir"
    fi
done

if [ ${#failed_runs[@]} -ne 0 ]; then
    echo "----------------------------------------------------------"
    echo "ERROR: The following Common Voice benchmark runs failed:"
    for failed in "${failed_runs[@]}"; do
        echo "  - $failed"
    done
    echo "Check the logs above for details."
    echo "----------------------------------------------------------"
    exit 1
fi

echo "Common Voice calibration runs completed. Check $CV_RUN_ROOT/results_*/calibration.md."
