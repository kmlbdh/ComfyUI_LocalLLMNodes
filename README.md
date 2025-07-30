# ComfyUI_LocalLLMNodes

A custom node pack for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) that allows you to run Large Language Models (LLMs) locally and use them for prompt generation and other text tasks directly within your ComfyUI workflows.

This pack provides nodes to connect to and utilize local LLMs in Hugging Face (PyTorch) or GGUF format, eliminating the need for external API calls. It's designed to integrate seamlessly with prompt generation workflows, such as those involving image description nodes like Florence-2.

## Features

*   **Local LLM Execution (Hugging Face & GGUF):** Run powerful LLMs directly on your machine (CPU or GPU) using either standard Hugging Face models or efficient GGUF models.
*   **Set Local LLM Service Connector Node (Hugging Face):** Select and configure your local Hugging Face format LLM model (models must be placed in `ComfyUI/models/LLM/`).
*   **Set Local GGUF LLM Service Connector Node:** Select and configure your local GGUF format LLM model file (`.gguf` files must be placed in `ComfyUI/models/LLM/`).
*   **Local Kontext Prompt Generator Node:** Generate detailed image prompts by combining descriptions and edit instructions, leveraging your connected local LLM.
*   **User Preset Management:** Add and remove custom prompt generation presets using dedicated nodes.
*   **Compatibility:** Designed to work with the `LLMServiceConnector` type identifier, ensuring compatibility with standard MieNodes prompt generators (e.g., `KontextPromptGenerator`) if needed.
*   **VRAM Optimization (Hugging Face):** Includes commented code examples for integrating Hugging Face model quantization (4-bit/8-bit) using `bitsandbytes` to reduce memory footprint.
*   **Efficient GGUF Models:** GGUF models are inherently quantized, offering lower memory usage and often good performance, especially on CPU.

## Installation

1.  **Navigate** to your ComfyUI installation directory.
2.  **Go to** the `custom_nodes` folder.
3.  **Clone** this repository:
    ```bash
    git clone https://github.com/your_username/ComfyUI_LocalLLMNodes.git
    # Or download the zip and extract it into a folder named ComfyUI_LocalLLMNodes
    ```
4.  **Install Dependencies:** Navigate into the `ComfyUI_LocalLLMNodes` directory and install the required Python packages.
    ```bash
    cd ComfyUI_LocalLLMNodes
    pip install -r requirements.txt
    ```
    *Note: Ensure you are installing these packages in the same Python environment that you use to run ComfyUI.*
    *Note on `llama-cpp-python`: Installing with GPU support (CUDA) requires specific environment variables during installation (e.g., `CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python --force-reinstall --no-cache-dir`). See the `llama-cpp-python` documentation for details. The CPU-only installation (`pip install llama-cpp-python`) is simpler and sufficient for CPU inference.

## Usage

1.  **Download a Local LLM:**
    *   **Hugging Face Format:**
        *   Obtain a Hugging Face format LLM (e.g., `TinyLlama/TinyLlama-1.1B-Chat-v1.0`, `microsoft/Phi-3-mini-4k-instruct`, `NousResearch/Hermes-2-Pro-Llama-3-8B`).
        *   Download the model files into a subdirectory within your `ComfyUI/models/LLM/` folder.
            *   Example: `ComfyUI/models/LLM/Phi-3-mini-4k-instruct/` should contain `config.json`, `pytorch_model.bin` (or `.safetensors`), `tokenizer_config.json`, etc.
    *   **GGUF Format:**
        *   Obtain a GGUF format LLM file (e.g., `mistral-7b-instruct-v0.3.Q8_0.gguf`).
        *   Place the `.gguf` file directly in your `ComfyUI/models/LLM/` folder, or within a subfolder (e.g., `ComfyUI/models/LLM/Mistral-7B-Instruct/` containing `mistral-7b-instruct-v0.3.Q8_0.gguf`).
2.  **Restart ComfyUI** to load the new nodes.
3.  **Find the Nodes:** Look for the new nodes in the ComfyUI node library under the category:
    *   `Local LLM Nodes/LLM Connectors`
4.  **Use the Nodes:**
    *   **For Hugging Face Models:**
        *   Add the **"Set Local LLM Service Connector 🐑 (HuggingFace)"** node to your graph.
        *   Select your downloaded Hugging Face model directory from the dropdown menu.
    *   **For GGUF Models:**
        *   Add the **"Set Local GGUF LLM Service Connector 🐑"** node to your graph.
        *   Select your downloaded GGUF model file (or its containing directory) from the dropdown menu.
    *   **Common Steps:**
        *   Add the **"Local Kontext Prompt Generator 🐑"** node.
        *   Connect the output of the chosen "Set Local ... LLM Service Connector 🐑" node to the `llm_service_connector` input of the "Local Kontext Prompt Generator 🐑" node.
        *   Provide inputs like `image1_description` (e.g., from Florence-2), `edit_instruction`, and select a `preset`.
        *   Connect the `kontext_prompt` output to your desired node (e.g., an image generator).

## Memory Optimization

Running large LLMs alongside large image models (like SDXL or Flux) can strain system resources (RAM/VRAM).

*   **Hugging Face Models:**
    *   **Quantization:** The `local_llm_connector.py` file includes commented code examples showing how to implement 4-bit or 8-bit quantization using the `bitsandbytes` library. This can significantly reduce the LLM's VRAM usage.
        *   To use quantization:
            1.  Ensure `bitsandbytes` is installed (`pip install bitsandbytes`).
            2.  Uncomment and adjust the quantization configuration section in the `_load_model` method within `local_llm_connector.py`.
            3.  Restart ComfyUI.
*   **GGUF Models:**
    *   GGUF models are pre-quantized (e.g., Q4, Q5, Q8). Choosing a more quantized version (like Q8_0 vs. f16) inherently uses less memory. For GPU acceleration with GGUF models, configure the `n_gpu_layers` parameter during loading (if supported by your `llama-cpp-python` build).

## Nodes Included

*   `SetLocalLLMServiceConnector`: Selects and prepares a connection to a local Hugging Face format LLM model.
*   `SetLocalGGUFLLMServiceConnector`: Selects and prepares a connection to a local GGUF format LLM model file.
*   `LocalKontextPromptGenerator`: Generates prompts using a connected local LLM based on descriptions and instructions.
*   `AddUserLocalKontextPreset`: Adds a custom preset for prompt generation.
*   `RemoveUserLocalKontextPreset`: Removes a custom preset.

## Requirements

*   [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
*   Python Libraries (see `requirements.txt` for versions):
    *   `transformers`
    *   `torch`
    *   `bitsandbytes` (Optional, for Hugging Face model quantization)
    *   `llama-cpp-python` (Optional, for GGUF model support)
    *   Other dependencies as listed in `requirements.txt`

## Acknowledgements

This node pack builds upon concepts and structures found in the excellent [ComfyUI-MieNodes](https://github.com/MieMieeeee/ComfyUI-MieNodes) extension, particularly the `KontextPromptGenerator` and LLM service connector patterns.