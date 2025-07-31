@echo off
REM install_deps.bat - Installs dependencies for ComfyUI_LocalLLMNodes (Command Prompt)

REM Exit immediately if a command exits with a non-zero status.
if "%~1"=="-CI" set CI=1
setlocal enabledelayedexpansion

echo ===== Installing ComfyUI_LocalLLMNodes Dependencies =====

REM --- Install core dependencies from requirements.txt ---
REM This installs transformers, torch, bitsandbytes (if listed), and llama-cpp-python (CPU version potentially)
echo 1/3: Installing core dependencies from requirements.txt...
pip install -r requirements.txt
if errorlevel 1 (
    echo Error occurred during pip install -r requirements.txt
    exit /b 1
)

REM --- Install llama-cpp-python with CUDA support ---
REM This overwrites/ensures the GPU-accelerated version is installed
echo 2/3: Installing llama-cpp-python with CUDA support...
echo     (This might take a while and download/build components...)
REM Set environment variable and install llama-cpp-python
REM Adjust cu118/cu121/cu124 based on your CUDA version
set CMAKE_ARGS=-DGGML_CUDA=on
pip install llama-cpp-python --force-reinstall --no-cache-dir
if errorlevel 1 (
    echo Error occurred during pip install llama-cpp-python
    exit /b 1
)

REM --- Completion ---
echo 3/3: Installation process completed.
echo ===== Installation Finished =====
echo.
echo Please ensure you have the NVIDIA CUDA toolkit installed
echo and that your environment is set up correctly for ComfyUI.
echo You can now start ComfyUI.

endlocal