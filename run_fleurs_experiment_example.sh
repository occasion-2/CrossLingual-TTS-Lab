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
models["f5tts"]="f5tts_v1_base f5_tts model=F5TTS_v1_Base ref_text_mode=empty nfe_step=32 cfg_strength=2.0 sway_sampling_coef=-1.0 speed=1.0 remove_silence=false ode_method=euler use_ema=true checkpoint_revision=84e5a410d9cead4de2f847e7c9369a6440bdfaca vocoder_revision=0feb3fdd929bcd6649e0e7c5a688cf7dd012ef21"
models["qwen0_6b"]="qwen3_tts_0_6b_base qwen_tts model=Qwen/Qwen3-TTS-12Hz-0.6B-Base revision=5d83992436eae1d760afd27aff78a71d676296fc ref_text_mode=empty x_vector_only_mode=true dtype=bfloat16"
models["qwen1_7b"]="qwen3_tts_1_7b_base qwen_tts model=Qwen/Qwen3-TTS-12Hz-1.7B-Base revision=fd4b254389122332181a7c3db7f27e918eec64e3 ref_text_mode=empty x_vector_only_mode=true dtype=bfloat16"
models["xtts"]="xtts_v2 coqui_xtts model_name=tts_models/multilingual/multi-dataset/xtts_v2 model_sha256=c7ea20001c6a0a841c77e252d8409f6a74fb423e79b3206a0771ba5989776187 config_sha256=ef262b1454dd2a77e1461b0b2cd53e19b8a7624cc131b837d36df67356bc75e8 vocab_sha256=928260878a59da8a72a2a5b7687fea29d5106137669d90945430fe17e415304a speakers_sha256=f0f6137c19a4eab0cbbe4c99b5babacf68b1746e50da90807708c10e645b943b"
models["cosyvoice"]="cosyvoice cosyvoice model_name=FunAudioLLM/Fun-CosyVoice3-0.5B-2512 artifact_tree_sha256=3f05b0236c11518e035c73996e60a2055d31d48fd2b52f3514444fc6157ada46 source_revision=074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc ref_text_mode=transcript stream=false speed=1.0 text_frontend=true fp16=false load_jit=false load_trt=false load_vllm=false"
models["spark_tts"]="spark_tts spark_tts model_name=pretrained_models/Spark-TTS-0.5B revision=642071559bfc6346c2359d19dcb6be3f9dd8a05d source_revision=2f1ea9082400547242641f5271b6f941c9f439d1 ref_text_mode=empty temperature=0.8 top_k=50 top_p=0.95"

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
