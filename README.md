# ComfyUI_LocalLLMNodes

A custom node pack for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) that allows you to run Large Language Models (LLMs) locally and use them for prompt generation and other text tasks directly within your ComfyUI workflows.

This pack provides nodes to connect to and utilize local LLMs (like Llama, Phi, Gemma, etc., in Hugging Face format) without needing external API calls. It's designed to integrate seamlessly with prompt generation workflows, such as those involving image description nodes like Florence-2.

## Features

*   **Local LLM Execution:** Run powerful LLMs directly on your machine (CPU or GPU).
*   **Set Local LLM Service Connector Node:** Select and configure your local LLM model (models must be placed in `ComfyUI/models/LLM/`).
*   **Local Kontext Prompt Generator Node:** Generate detailed image prompts by combining descriptions and edit instructions, leveraging your local LLM.
*   **User Preset Management:** Add and remove custom prompt generation presets using dedicated nodes.
*   **Compatibility:** Designed to work with standard MieNodes prompt generators (e.g., `KontextPromptGenerator`) if needed, using the `LLMServiceConnector` type identifier.
*   **VRAM Optimization Ready:** Includes commented code examples for integrating quantization (4-bit/8-bit) using `bitsandbytes` to reduce memory footprint for running alongside large image models like Flux.

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
    # Or install directly:
    # pip install transformers torch
    # Optional for quantization: pip install bitsandbytes
    ```
    *Note: Ensure you are installing these packages in the same Python environment that you use to run ComfyUI.*

## Usage

1.  **Download a Local LLM:**
    *   Obtain a Hugging Face format LLM (e.g., `TinyLlama/TinyLlama-1.1B-Chat-v1.0`, `microsoft/Phi-3-mini-4k-instruct`, `google/gemma-2b-it`).
    *   Download the model files into a subdirectory within your `ComfyUI/models/LLM/` folder.
        *   Example: `ComfyUI/models/LLM/Phi-3-mini-4k-instruct/` should contain `config.json`, `pytorch_model.bin` (or `.safetensors`), `tokenizer_config.json`, etc.
2.  **Restart ComfyUI** to load the new nodes.
3.  **Find the Nodes:** Look for the new nodes in the ComfyUI node library under the categories:
    *   `Local LLM Nodes/LLM Connectors`
    *   `Local LLM Nodes/Prompt Generators`
4.  **Use the Nodes:**
    *   Add the **"Set Local LLM Service Connector 🐑"** node to your graph.
    *   Select your downloaded local LLM model from the dropdown menu.
    *   Add the **"Local Kontext Prompt Generator 🐑"** node.
    *   Connect the output of the "Set Local LLM Service Connector 🐑" node to the `llm_service_connector` input of the "Local Kontext Prompt Generator 🐑" node.
    *   Provide inputs like `image1_description` (e.g., from Florence-2), `edit_instruction`, and select a `preset`.
    *   Connect the `kontext_prompt` output to your desired node (e.g., an image generator).

## Memory Optimization (VRAM)

Running large LLMs alongside large image models (like SDXL or Flux) can strain GPU memory (VRAM).

*   **Quantization:** The `local_llm_connector.py` file includes commented code examples showing how to implement 4-bit or 8-bit quantization using the `bitsandbytes` library. This can significantly reduce the LLM's VRAM usage.
    *   To use quantization:
        1.  Ensure `bitsandbytes` is installed (`pip install bitsandbytes`).
        2.  Uncomment and adjust the quantization configuration section in the `_load_model` method within `local_llm_connector.py`.
        3.  Restart ComfyUI.

## Nodes Included

*   `SetLocalLLMServiceConnector`: Selects and prepares a connection to a local LLM model.
*   `LocalKontextPromptGenerator`: Generates prompts using a connected local LLM based on descriptions and instructions.
*   `AddUserLocalKontextPreset`: Adds a custom preset for prompt generation.
*   `RemoveUserLocalKontextPreset`: Removes a custom preset.

## Requirements

*   [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
*   Python Libraries:
    *   `transformers`
    *   `torch`
    *   `bitsandbytes` (Optional, for quantization)
    *   (See `requirements.txt`)

## Acknowledgements

This node pack builds upon concepts and structures found in the excellent [ComfyUI-MieNodes](https://github.com/MieMieeeee/ComfyUI-MieNodes) extension, particularly the `KontextPromptGenerator` and LLM service connector patterns.
