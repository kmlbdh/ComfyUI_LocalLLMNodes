# install_deps.ps1 - Installs dependencies for ComfyUI_LocalLLMNodes (PowerShell)

Write-Host "===== Installing ComfyUI_LocalLLMNodes Dependencies ====="

# --- Install core dependencies from requirements.txt ---
# This installs transformers, torch, bitsandbytes (if listed), and llama-cpp-python (CPU version potentially)
Write-Host "1/3: Installing core dependencies from requirements.txt..."
pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Error "Error occurred during pip install -r requirements.txt"
    exit 1
}

# --- Install llama-cpp-python with CUDA support ---
# This overwrites/ensures the GPU-accelerated version is installed
Write-Host "2/3: Installing llama-cpp-python with CUDA support..."
Write-Host "    (This might take a while and download/build components...)"
# Set environment variable and install llama-cpp-python
# Adjust cu118/cu121/cu124 based on your CUDA version
$env:CMAKE_ARGS = "-DGGML_CUDA=on"
pip install llama-cpp-python --force-reinstall --no-cache-dir
if ($LASTEXITCODE -ne 0) {
    Write-Error "Error occurred during pip install llama-cpp-python"
    exit 1
}

# --- Completion ---
Write-Host "3/3: Installation process completed."
Write-Host "===== Installation Finished ====="
Write-Host ""
Write-Host "Please ensure you have the NVIDIA CUDA toolkit installed"
Write-Host "and that your environment is set up correctly for ComfyUI."
Write-Host "You can now start ComfyUI."
