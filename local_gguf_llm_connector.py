# ComfyUI/custom_nodes/ComfyUI_LocalLLMNodes/local_gguf_llm_connector.py
import os
import folder_paths
import subprocess
import sys

# --- Library for GGUF local LLM inference ---
# We're now using ctransformers instead of llama_cpp
try:
    from ctransformers import AutoModelForCausalLM
    CTRANSFORMERS_AVAILABLE = True
except ImportError:
    from .local_llm_connector import log
    log("[LocalGGUFLLMConnector] Warning: ctransformers library not found. Attempting auto-install...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ctransformers"])
        from ctransformers import AutoModelForCausalLM
        CTRANSFORMERS_AVAILABLE = True
        log("[LocalGGUFLLMConnector] ctransformers installed successfully.")
    except Exception as e:
        log(f"[LocalGGUFLLMConnector] Failed to install ctransformers. Please install it manually: pip install ctransformers. Error: {e}")
        CTRANSFORMERS_AVAILABLE = False
        AutoModelForCausalLM = None

# Reuse the logging function from local_llm_connector
from .local_llm_connector import log

LOCAL_LLM_CATEGORY = "Local LLM Nodes/LLM Connectors"

def get_local_gguf_model_names():
    """Discovers .gguf files or directories containing .gguf files within models/LLM."""
    llm_models_dir = os.path.join(folder_paths.models_dir, "LLM")
    model_names = []

    if os.path.exists(llm_models_dir):
        try:
            for item in os.listdir(llm_models_dir):
                item_path = os.path.join(llm_models_dir, item)
                if os.path.isfile(item_path) and item.endswith('.gguf'):
                    # Add the filename without .gguf extension
                    model_names.append(os.path.splitext(item)[0])
                elif os.path.isdir(item_path):
                     # Check if directory contains a .gguf file
                     for subitem in os.listdir(item_path):
                         if subitem.endswith('.gguf'):
                             model_names.append(item)
                             break
        except Exception as e:
            log(f"[LocalGGUFLLMConnector] Error scanning models/LLM directory: {e}")
    else:
        log(f"[LocalGGUFLLMConnector] models/LLM directory not found: {llm_models_dir}")

    if not model_names:
        model_names = ["No_Local_GGUF_Models_Found"]
    return model_names

class SetLocalGGUFLLMServiceConnector:
    """
    A node to select and prepare a connection to a local GGUF LLM model.
    Now uses the ctransformers library.
    """
    @classmethod
    def INPUT_TYPES(cls):
        model_names = get_local_gguf_model_names()
        return {
            "required": {
                "local_gguf_model_name": (model_names, {"default": model_names[0] if model_names else "No_Local_GGUF_Models_Found"}),
                "device": (["GPU", "CPU"], {"default": "GPU"}),
                "n_gpu_layers": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 100,
                    "step": 1,
                    "display": "slider"
                }),
                # --- Optional: Expose other common parameters ---
                # "n_threads": ("INT", {"default": 8, "min": 1, "max": 64}),
                # "n_ctx": ("INT", {"default": 4096, "min": 1, "max": 32768}), # Max depends on model
            },
        }

    RETURN_TYPES = ("LLMServiceConnector",)
    FUNCTION = "get_connector"
    CATEGORY = LOCAL_LLM_CATEGORY
    OUTPUT_NODE = False

    def get_connector(self, local_gguf_model_name, device, n_gpu_layers):
        """Returns a connector object for the selected GGUF model."""
        if not CTRANSFORMERS_AVAILABLE:
            raise Exception("The 'ctransformers' library is required for the Local GGUF LLM node but is not installed.")

        if local_gguf_model_name == "No_Local_GGUF_Models_Found":
             raise Exception("No local GGUF models found in models/LLM directory. Please place your .gguf files there.")

        base_models_dir = os.path.join(folder_paths.models_dir, "LLM")
        model_path = os.path.join(base_models_dir, local_gguf_model_name)

        gguf_file_path = None
        if os.path.isfile(model_path) and model_path.endswith('.gguf'):
            gguf_file_path = model_path
        elif os.path.isdir(model_path):
            for file in os.listdir(model_path):
                if file.endswith('.gguf'):
                    gguf_file_path = os.path.join(model_path, file)
                    break

        if not gguf_file_path or not os.path.exists(gguf_file_path):
             raise FileNotFoundError(f"[LocalGGUFLLMConnector] GGUF model file not found for selection: {local_gguf_model_name}")

        final_n_gpu_layers = n_gpu_layers
        if device == "CPU" and n_gpu_layers == -1:
            final_n_gpu_layers = 0
            log(f"[LocalGGUFLLMConnector] Device set to CPU, overriding n_gpu_layers to 0.")
        elif device == "GPU" and n_gpu_layers == -1:
            final_n_gpu_layers = -1
            log(f"[LocalGGUFLLMConnector] Device set to GPU, keeping n_gpu_layers=-1 (max offload).")
        else:
            log(f"[LocalGGUFLLMConnector] Using user-provided n_gpu_layers={n_gpu_layers} (device={device}).")

        connector = LocalGGUFLLMServiceConnector(
            gguf_file_path,
            n_gpu_layers=final_n_gpu_layers
        )
        return (connector,)


class LocalGGUFLLMServiceConnector:
    """
    Represents the connection to a specific local GGUF LLM using ctransformers.
    The `gguf_file_path` should point directly to the .gguf file.
    """
    def __init__(self, gguf_file_path, n_gpu_layers=-1):
        self.gguf_file_path = gguf_file_path
        self.n_gpu_layers = n_gpu_layers
        self.model = None
        self.is_loaded = False

    def _load_model(self):
        """Loads the GGUF model using ctransformers."""
        if self.is_loaded:
            return

        if not CTRANSFORMERS_AVAILABLE:
            raise Exception("The 'ctransformers' library is not available.")

        try:
            log(f"[LocalGGUFLLMConnector] Loading GGUF model from: {self.gguf_file_path}")
            model_kwargs = {
                # We need to extract the model file name from the full path
                "model_file": os.path.basename(self.gguf_file_path),
                # The path to the directory containing the model file
                "model_path": os.path.dirname(self.gguf_file_path),
                "model_type": "llama", # Assuming a Llama-like format
                "gpu_layers": self.n_gpu_layers,
                # Optional: configure other parameters here if needed
                "verbose": True
            }
            self.model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
            self.is_loaded = True
            log(f"[LocalGGUFLLMConnector] GGUF model loaded successfully.")
        except Exception as e:
            error_msg = f"[LocalGGUFLLMConnector] Failed to load GGUF model: {e}"
            log(error_msg)
            raise Exception(error_msg) from e

    def invoke(self, messages: list, generation_kwargs: dict = None) -> str:
        """
        Generates text using the local GGUF LLM based on the messages list.
        This method formats the messages into a single prompt string suitable for ctransformers.
        """
        try:
            if not self.is_loaded:
                self._load_model()

            if not self.model:
                raise Exception("[LocalGGUFLLMConnector] Local GGUF LLM model failed to load.")
            
            # Make sure generation_kwargs is a dictionary
            if generation_kwargs is None:
                generation_kwargs = {}

            # Filter out unsupported keywords for ctransformers, like 'seed'
            # This is the key fix to prevent the error
            gen_kwargs = {
                k: v for k, v in generation_kwargs.items() if k != 'seed'
            }

            # ctransformers does not have a native create_chat_completion method.
            # We must format the messages list into a single prompt string.
            prompt_parts = []
            for message in messages:
                role = message.get('role', 'user')
                content = message.get('content', '')
                if role == "system":
                    prompt_parts.append(f"### System:\n{content}")
                elif role == "user":
                    prompt_parts.append(f"### User:\n{content}")
                elif role == "assistant":
                    prompt_parts.append(f"### Assistant:\n{content}")
            
            full_prompt = "\n\n".join(prompt_parts) + "\n\n### Assistant:\n"
            
            final_gen_kwargs = {
                'temperature': gen_kwargs.get('temperature', 0.7),
                'max_new_tokens': gen_kwargs.get('max_new_tokens', 256),
                'repetition_penalty': gen_kwargs.get('repetition_penalty', 1.1),
                'top_p': gen_kwargs.get('top_p', 0.9),
                'stop': gen_kwargs.get('stop', []),
            }

            generated_text = self.model(full_prompt, **final_gen_kwargs)
            
            return generated_text.strip() if generated_text else ""

        except Exception as e:
            error_msg = f"[LocalGGUFLLMConnector] Error in invoke method: {str(e)}"
            log(error_msg)
            raise e
