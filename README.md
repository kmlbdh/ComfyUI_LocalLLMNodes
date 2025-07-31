# ComfyUI_LocalLLMNodes

A custom node pack for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) that allows you to run Large Language Models (LLMs) locally and use them for prompt generation and other text tasks directly within your ComfyUI workflows.

This pack provides nodes to connect to and utilize local LLMs (like Llama, Phi, Gemma, Hermes in Hugging Face PyTorch format, or GGUF models) without needing external API calls. It's designed to integrate seamlessly with prompt generation workflows, such as those involving image description nodes like Florence-2, and to simplify the creation of complex prompts for models like Flux Kontext Dev.

## Features

*   **Local LLM Execution (Hugging Face & GGUF):** Run powerful LLMs directly on your machine (CPU or GPU) using either standard Hugging Face models or efficient GGUF models.
*   **Set Local LLM Service Connector Node (HuggingFace):** Select and configure your local Hugging Face format LLM model (models must be placed in `ComfyUI/models/LLM/`).
*   **Set Local GGUF LLM Service Connector Node:** Select and configure your local GGUF format LLM model file (`.gguf` files must be placed in `ComfyUI/models/LLM/`). **Includes dropdown for device selection (CPU/GPU) and `n_gpu_layers` slider for fine-grained control.**
*   **Local Kontext Prompt Generator Node:** **(Key Feature)** Generates detailed image prompts by intelligently combining image descriptions (e.g., from Florence-2) with simple user instructions. Designed to work with local LLM connectors to produce high-quality prompts for advanced models like Flux Kontext Dev, simplifying the user's task.
*   **User Preset Management:** Add and remove custom prompt generation presets using dedicated nodes.
*   **VRAM Optimization Ready:** Includes commented code examples for integrating quantization (4-bit/8-bit using `bitsandbytes` for Hugging Face models, or controlling `n_gpu_layers` for GGUF) to reduce memory footprint for running alongside large image models like Flux.
*   **Simplified User Experience:** Allows users to provide simple, natural language requests (e.g., "Make it look like it's being used in a luxury spa") and translates them into complex, Flux-ready prompts using the connected local LLM.

## Installation

1.  **Navigate** to your ComfyUI installation directory.
2.  **Go to** the `custom_nodes` folder.
3.  **Clone** this repository:
    ```bash
    git clone https://github.com/your_username/ComfyUI_LocalLLMNodes.git
    # Or download the zip and extract it into a folder named ComfyUI_LocalLLMNodes
    ```
4.  **Install Dependencies:** You can install dependencies using `pip` and the provided scripts or `requirements.txt`.
    *   **Option 1: Using Installation Scripts (Recommended for GPU Support):**
        *   Navigate to the `ComfyUI_LocalLLMNodes` directory:
            ```bash
            cd ComfyUI_LocalLLMNodes
            ```
        *   **Linux/macOS:**
            ```bash
            # Make the script executable
            chmod +x install_deps.sh
            # Run the script
            ./install_deps.sh
            ```
        *   **Windows (Command Prompt):**
            ```cmd
            install_deps.bat
            ```
        *   **Windows (PowerShell):**
            ```powershell
            # You might need to adjust execution policy first:
            # Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
            .\install_deps.ps1
            ```
        *   These scripts will install core dependencies and specifically `llama-cpp-python` with CUDA support (essential for GPU acceleration with GGUF models).
    *   **Option 2: Standard `pip install`:**
        ```bash
        cd ComfyUI_LocalLLMNodes
        pip install -r requirements.txt
        ```
        *Note: Ensure you are installing these packages in the same Python environment that you use to run ComfyUI.*
        *Note on `llama-cpp-python`: The standard `pip install llama-cpp-python` often installs a CPU-only version. For GPU acceleration (highly recommended for GGUF models), use the installation scripts above or follow the manual steps below.*

### Installing `llama-cpp-python` with GPU Support (CUDA) - Important for GGUF Nodes

To leverage your GPU for running GGUF models via the `Set Local GGUF LLM Service Connector 🐑` node, you need to install `llama-cpp-python` with CUDA support compiled in. The standard `pip install llama-cpp-python` often installs a CPU-only version.

**Manual Installation (if not using scripts):**

1.  **Ensure CUDA Toolkit is Installed:** You need the NVIDIA CUDA toolkit installed on your system, matching the version compatible with your GPU drivers. Check NVIDIA's website for instructions.
2.  **Set Environment Variables and Install:**
    *   **Linux/macOS:**
        ```bash
        # Replace cu118/cu121/cu124 with the CUDA version you have installed (e.g., cu118 for CUDA 11.8, cu121 for CUDA 12.1)
        CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
        ```
    *   **Windows (Command Prompt):**
        ```cmd
        set CMAKE_ARGS=-DGGML_CUDA=on
        pip install llama-cpp-python --force-reinstall --no-cache-dir
        ```
    *   **Windows (PowerShell):**
        ```powershell
        $env:CMAKE_ARGS = "-DGGML_CUDA=on"
        pip install llama-cpp-python --force-reinstall --no-cache-dir
        ```
3.  **Verify Installation:** After installation, you can check if CUDA support is enabled by running Python and trying to import:
    ```bash
    python -c "import llama_cpp; print('llama_cpp imported successfully')"
    # A successful import without errors related to CUDA libraries usually indicates it's compiled correctly.
    # Detailed logs during model loading (like `load_tensors: layer X assigned to device CUDA0`) will confirm GPU usage.
    ```

## Usage

1.  **Download a Local LLM:**
    *   **Hugging Face Format:**
        *   Obtain a Hugging Face format LLM (e.g., `TinyLlama/TinyLlama-1.1B-Chat-v1.0`, `microsoft/Phi-3-mini-4k-instruct`, `NousResearch/Hermes-2-Pro-Llama-3-8B`, `mistralai/Mistral-Nemo-Instruct-2407`).
        *   Download the model files into a subdirectory within your `ComfyUI/models/LLM/` folder.
            *   Example: `ComfyUI/models/LLM/Phi-3-mini-4k-instruct/` should contain `config.json`, `pytorch_model.bin` (or `.safetensors`), `tokenizer_config.json`, etc.
    *   **GGUF Format:**
        *   Obtain a GGUF format LLM file (e.g., `mistral-nemo-instruct-2407.Q8_0.gguf`, `Hermes-2-Pro-Llama-3-8B-Q8_0.gguf`).
        *   Place the `.gguf` file directly in your `ComfyUI/models/LLM/` folder, or within a subfolder (e.g., `ComfyUI/models/LLM/Mistral-Nemo-Instruct-2407/` containing `mistral-nemo-instruct-2407.Q8_0.gguf`).
2.  **Restart ComfyUI** to load the new nodes.
3.  **Find the Nodes:** Look for the new nodes in the ComfyUI node library under the category:
    *   `Local LLM Nodes/LLM Connectors`
4.  **Use the Nodes:**
    *   **For Hugging Face Models:**
        *   Add the **"Set Local LLM Service Connector 🐑 (HuggingFace)"** node to your graph.
        *   Select your downloaded Hugging Face model directory from the dropdown menu.
    *   **For GGUF Models:**
        *   Add the **"Set Local GGUF LLM Service Connector 🐑"** node to your graph.
        *   **Select your downloaded GGUF model file (or its containing directory) from the dropdown menu.**
        *   **Use the `device` dropdown:** Choose `GPU` to attempt GPU acceleration (requires `llama-cpp-python` installed with CUDA support as described above). Choose `CPU` to force CPU execution.
        *   **Adjust `n_gpu_layers` slider:**
            *   If `device` is `GPU`, the slider defaults to `-1`, meaning "offload as many layers as possible to the GPU".
            *   If `device` is `CPU`, the slider defaults to `-1`, but the node internally sets `n_gpu_layers=0`.
            *   You can manually adjust the slider to offload a specific number of layers (e.g., `30` out of `32`) if desired, overriding the default behavior.
    *   **Common Steps for Prompt Generation:**
        *   Add the **"Local Kontext Prompt Generator 🐑"** node.
        *   Connect the output of the chosen "Set Local ... LLM Service Connector 🐑" node to the `llm_service_connector` input of the "Local Kontext Prompt Generator 🐑" node.
        *   **Provide Inputs:**
            *   Connect the output of an image description node (like Florence-2) to the `image1_description` input.
            *   Provide a simple, natural language `edit_instruction` in the node's text field (e.g., "Make it look like it's being used in a luxury spa", "Change the background to a beach").
            *   Select a suitable `preset` from the dropdown (e.g., "User Intent -> Flux Prompt", "Product - 产品摄影").
        *   Connect the `kontext_prompt` output to your desired node (e.g., an image generator like Flux Kontext Dev).

## Memory Optimization

Running large LLMs alongside large image models (like SDXL or Flux) can strain system resources (RAM/VRAM).

*   **Hugging Face Models:**
    *   **Quantization:** The `local_llm_connector.py` file includes commented code examples for integrating Hugging Face model quantization (4-bit/8-bit) using the `bitsandbytes` library. This can significantly reduce the LLM's VRAM usage.
        *   To use quantization:
            1.  Ensure `bitsandbytes` is installed (`pip install bitsandbytes`).
            2.  Uncomment and adjust the quantization configuration section in the `_load_model` method within `local_llm_connector.py`.
            3.  Restart ComfyUI.
*   **GGUF Models:**
    *   GGUF models are inherently quantized (e.g., Q4_K_M, Q5_K, Q8_0). Choosing a more quantized version (like Q8_0 vs. f16) inherently uses less memory.
    *   For GPU acceleration with GGUF models, configure the `n_gpu_layers` parameter during loading (if supported by your `llama-cpp-python` build and setup). The `Set Local GGUF LLM Service Connector 🐑` node provides explicit controls for this.

## Nodes Included

*   `SetLocalLLMServiceConnector`: Selects and prepares a connection to a local Hugging Face format LLM model.
*   `SetLocalGGUFLLMServiceConnector`: **(Updated)** Selects and prepares a connection to a local GGUF format LLM model file. Includes `device` dropdown (`CPU`/`GPU`) and `n_gpu_layers` slider for controlling offloading.
*   `LocalKontextPromptGenerator`: **(Key Node)** Generates prompts using a connected local LLM by combining image descriptions and simple user instructions, optimized for advanced image models like Flux Kontext Dev.
*   `AddUserLocalKontextPreset`: Adds a custom preset for prompt generation.
*   `RemoveUserLocalKontextPreset`: Removes a custom preset.

## Requirements

*   [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
*   Python Libraries (see `requirements.txt` for versions):
    *   `transformers`
    *   `torch`
    *   `llama-cpp-python` (Optional, for GGUF model support - **install with GPU flags as described for GPU acceleration**)
    *   `bitsandbytes` (Optional, for Hugging Face model quantization)
    *   Other dependencies as listed in `requirements.txt`

## Acknowledgements

This node pack builds upon concepts and structures found in the excellent [ComfyUI-MieNodes](https://github.com/MieMieeeee/ComfyUI-MieNodes) extension, particularly the `KontextPromptGenerator` and LLM service connector patterns.
