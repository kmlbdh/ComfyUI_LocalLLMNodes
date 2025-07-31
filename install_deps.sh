#!/bin/bash

# install_deps.sh - Installs dependencies for ComfyUI_LocalLLMNodes

# Exit immediately if a command exits with a non-zero status.
set -e

echo "===== Installing ComfyUI_LocalLLMNodes Dependencies ====="

# --- Install core dependencies from requirements.txt ---
# This installs transformers, torch, bitsandbytes (if listed), and llama-cpp-python (CPU version potentially)
echo "1/3: Installing core dependencies from requirements.txt..."
pip install -r requirements.txt

# --- Install llama-cpp-python with CUDA support ---
# This overwrites/ensures the GPU-accelerated version is installed
echo "2/3: Installing llama-cpp-python with CUDA support..."
echo "    (This might take a while and download/build components...)"
# Set environment variable and install llama-cpp-python
# Adjust cu118/cu121/cu124 based on your CUDA version (check `nvcc --version`)
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir

# --- Completion ---
echo "3/3: Installation process completed."
echo "===== Installation Finished ====="
echo ""
echo "Please ensure you have the NVIDIA CUDA toolkit installed"
echo "and that your environment is set up correctly for ComfyUI."
echo "You can now start ComfyUI."
