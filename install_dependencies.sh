#!/bin/bash
set -e

# Parse command line arguments
INSTALL_COSY=false
INSTALL_SPARK=false

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --cosyvoice) INSTALL_COSY=true ;;
        --spark-tts) INSTALL_SPARK=true ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  (no options)     Clone CosyVoice & Spark-TTS repositories and download weights."
            echo "  --cosyvoice      Install CosyVoice repo/weights AND its Python dependencies into the active environment."
            echo "  --spark-tts      Install Spark-TTS repo/weights AND its Python dependencies into the active environment."
            exit 0
            ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

echo "=========================================================="
echo "Installing External Dependencies (CosyVoice & Spark-TTS)"
echo "=========================================================="

# Ensure git is available
if ! command -v git &> /dev/null; then
    echo "Error: git is not installed. Please install git and try again."
    exit 1
fi

# Clone CosyVoice if missing and requested
if [ "$INSTALL_COSY" = true ] || { [ "$INSTALL_COSY" = false ] && [ "$INSTALL_SPARK" = false ]; }; then
    if [ ! -d "CosyVoice" ]; then
        echo "Cloning CosyVoice repository..."
        git clone https://github.com/FunASR/CosyVoice.git
    else
        echo "CosyVoice repository already present."
    fi

    # Always ensure third-party submodules are initialized
    echo "Ensuring CosyVoice submodules are initialized..."
    cd CosyVoice
    git submodule update --init --recursive
    cd ..
fi

# Clone Spark-TTS if missing and requested
if [ "$INSTALL_SPARK" = true ] || { [ "$INSTALL_COSY" = false ] && [ "$INSTALL_SPARK" = false ]; }; then
    if [ ! -d "Spark-TTS" ]; then
        echo "Cloning Spark-TTS repository..."
        git clone https://github.com/SparkAudio/Spark-TTS.git
    else
        echo "Spark-TTS repository already present."
    fi

    # Download Spark-TTS pretrained weights
    if [ ! -d "pretrained_models/Spark-TTS-0.5B" ]; then
        echo "Downloading Spark-TTS 0.5B model weights from Hugging Face..."
        mkdir -p pretrained_models
        
        if command -v uv &> /dev/null; then
            uv run --with huggingface_hub huggingface-cli download SparkAudio/Spark-TTS-0.5B --local-dir pretrained_models/Spark-TTS-0.5B
        else
            pip install huggingface_hub
            huggingface-cli download SparkAudio/Spark-TTS-0.5B --local-dir pretrained_models/Spark-TTS-0.5B
        fi
    else
        echo "Spark-TTS model weights already downloaded."
    fi
fi

# Install Python dependencies if requested
if [ "$INSTALL_COSY" = true ]; then
    echo "----------------------------------------------------------"
    echo "Installing Python dependencies for CosyVoice..."
    echo "----------------------------------------------------------"
    if [[ "$VIRTUAL_ENV" != *".venv_cosyvoice"* ]]; then
        echo "Warning: The active environment is not the isolated CosyVoice environment (.venv_cosyvoice)."
        echo "Skipping automatic installation of nvidia-*-cu12 helper packages to prevent poisoning your environment (e.g. main project .venv)."
        echo "If you explicitly need them, please install them manually, or run within the isolated CosyVoice environment."
    else
        echo "Installing into virtual environment: $VIRTUAL_ENV"
        echo "Installing CUDA 12 support packages for CosyVoice ONNX Runtime GPU..."
        if command -v uv &> /dev/null; then
            uv pip install "nvidia-cudnn-cu12>=8.9,<9.0.0" "nvidia-cuda-runtime-cu12>=12.9" "nvidia-cufft-cu12>=11.4" "nvidia-curand-cu12>=10.3" "nvidia-cusolver-cu12>=11.7" "nvidia-cusparse-cu12>=12.5"
        else
            pip install "nvidia-cudnn-cu12>=8.9,<9.0.0" "nvidia-cuda-runtime-cu12>=12.9" "nvidia-cufft-cu12>=11.4" "nvidia-curand-cu12>=10.3" "nvidia-cusolver-cu12>=11.7" "nvidia-cusparse-cu12>=12.5"
        fi
    fi
    
    if command -v uv &> /dev/null; then
        echo "Using uv to install..."
        uv pip install "setuptools<70" wheel
        uv pip install -e ".[open-data,metrics,cosyvoice]" --no-build-isolation-package openai-whisper --no-build-isolation-package deepspeed
    else
        echo "Using pip to install..."
        pip install "setuptools<70" wheel
        pip install -e ".[open-data,metrics,cosyvoice]" --no-build-isolation-package openai-whisper --no-build-isolation-package deepspeed
    fi
fi

if [ "$INSTALL_SPARK" = true ]; then
    echo "----------------------------------------------------------"
    echo "Installing Python dependencies for Spark-TTS..."
    echo "----------------------------------------------------------"
    if [ -z "$VIRTUAL_ENV" ]; then
        echo "Warning: No active virtual environment detected (VIRTUAL_ENV is empty)."
        echo "Installing into your active Python environment."
    else
        echo "Installing into virtual environment: $VIRTUAL_ENV"
    fi
    
    if command -v uv &> /dev/null; then
        echo "Using uv to install..."
        uv pip install -e ".[open-data,metrics,spark-tts]"
    else
        echo "Using pip to install..."
        pip install -e ".[open-data,metrics,spark-tts]"
    fi
fi

echo "=========================================================="
echo "External dependencies successfully installed and configured!"
echo "=========================================================="
