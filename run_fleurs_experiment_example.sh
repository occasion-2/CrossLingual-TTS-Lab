#!/bin/bash
set -e

# Run overnight benchmark for 6 models: F5-TTS, Qwen3-TTS 0.6B, Qwen3-TTS 1.7B, XTTS, CosyVoice, Spark-TTS
# Requirements: at least 10 voices per language, at least 10 targets per voice per target language.
# This gives ~10 * 3 languages = 30 voices.
# Target languages: en, zh, ru (so 3 * 10 = 30 targets per voice).
# Total jobs per model: 30 voices * 30 targets = 900 jobs.

export COQUI_TOS_AGREED=1
export UV_CACHE_DIR="$PWD/.uv-cache"
export UV_VENV_CLEAR=1

echo "Generating configs for each model to avoid GPU OOM..."

declare -A models
models["f5tts"]="f5tts_v1_base f5_tts model=F5TTS_v1_Base ref_text_mode=empty"
models["qwen0_6b"]="qwen3_tts_0_6b_base qwen_tts model=Qwen/Qwen3-TTS-12Hz-0.6B-Base ref_text_mode=empty"
models["qwen1_7b"]="qwen3_tts_1_7b_base qwen_tts model=Qwen/Qwen3-TTS-12Hz-1.7B-Base ref_text_mode=empty"
models["xtts"]="xtts_v2 coqui_xtts model_name=tts_models/multilingual/multi-dataset/xtts_v2"
models["cosyvoice"]="cosyvoice cosyvoice model_name=FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
models["spark_tts"]="spark_tts spark_tts model_name=pretrained_models/Spark-TTS-0.5B ref_text_mode=empty"

mkdir -p overnight_runs
failed_runs=()

for key in "${!models[@]}"; do
    IFS=' ' read -r id backend params <<< "${models[$key]}"
    
    echo "======================================"
    
    if [ -f "overnight_runs/results_${key}/report.md" ]; then
        echo "Skipping $key: report already exists at overnight_runs/results_${key}/report.md"
        continue
    fi
    
    echo "Preparing isolated environment for $key..."
    
    VENV_DIR="overnight_runs/.venv_${key}"
    uv venv "$VENV_DIR"
    
    EXTRAS="open-data,metrics"
    if [[ "$key" == *"qwen"* ]]; then
        EXTRAS="$EXTRAS,qwen"
    elif [[ "$key" == *"f5"* ]]; then
        EXTRAS="$EXTRAS,f5"
    elif [[ "$key" == "xtts" ]]; then
        EXTRAS="$EXTRAS,tts"
    elif [[ "$key" == "cosyvoice" ]]; then
        EXTRAS="$EXTRAS,cosyvoice"
    elif [[ "$key" == "spark_tts" ]]; then
        EXTRAS="$EXTRAS,spark-tts"
    fi
    NO_ISOLATION=""
    if [[ "$key" == "cosyvoice" ]]; then
        echo "Installing build prerequisites and CUDA 12 support libraries for CosyVoice..."
        VIRTUAL_ENV="$VENV_DIR" uv pip install "setuptools<70" wheel
        VIRTUAL_ENV="$VENV_DIR" uv pip install "nvidia-cudnn-cu12>=8.9,<9.0.0" "nvidia-cuda-runtime-cu12>=12.9" "nvidia-cufft-cu12>=11.4" "nvidia-curand-cu12>=10.3" "nvidia-cusolver-cu12>=11.7" "nvidia-cusparse-cu12>=12.5"
        NO_ISOLATION="--no-build-isolation-package openai-whisper --no-build-isolation-package deepspeed"
    fi
    
    echo "Installing dependencies [$EXTRAS] into $VENV_DIR..."
    # Install dependencies mapped to this environment
    VIRTUAL_ENV="$VENV_DIR" uv pip install -e ".[$EXTRAS]" $NO_ISOLATION
    
    echo "Generating config for $key..."
    
    param_args=""
    for param in $params; do
        param_args="$param_args --model-param $param"
    done

    # Locate nvidia package libraries for this venv to prevent PyTorch/NVRTC loading issues
    NVIDIA_LIBS=$(find "$VENV_DIR" -type d -path "*/site-packages/nvidia/*/lib" | paste -sd : -)

    # Run from the isolated environment
    LD_LIBRARY_PATH="$NVIDIA_LIBS:$LD_LIBRARY_PATH" "$VENV_DIR/bin/xttslab" dataset fleurs \
      --languages ru:ru,en:en,zh-CN:zh \
      --voices-per-language 10 \
      --targets-per-language 10 \
      --target-languages en,zh,ru \
      --max-voice-chars 120 \
      --max-target-chars 110 \
      --model-id "$id" \
      --model-backend "$backend" \
      $param_args \
      --out "overnight_runs/config_${key}.toml"

    echo "Running benchmark for $key..."
    if ! LD_LIBRARY_PATH="$NVIDIA_LIBS:$LD_LIBRARY_PATH" "$VENV_DIR/bin/xttslab" run \
      --config "overnight_runs/config_${key}.toml" \
      --out "overnight_runs/results_${key}"; then
        echo "Error: Benchmark run for $key failed."
        failed_runs+=("$key")
    fi
done

if [ ${#failed_runs[@]} -ne 0 ]; then
    echo "----------------------------------------------------------"
    echo "ERROR: The following benchmark runs failed:"
    for failed in "${failed_runs[@]}"; do
        echo "  - $failed"
    done
    echo "Check the logs above for details."
    echo "----------------------------------------------------------"
    exit 1
fi

echo "Overnight benchmark runs completed. Check overnight_runs/ directory for manifests and reports."
