# ComfyUI/custom_nodes/ComfyUI_LocalLLMNodes/local_gguf_llm_connector.py
import os
import subprocess
import sys
import platform

try:
    import llama_cpp
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    from .local_llm_connector import log

    def check_nvidia_gpu():
        try:
            result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return result.returncode == 0
        except (FileNotFoundError, OSError):
            return False

    def check_apple_metal():
        return platform.system() == "Darwin" and "arm" in platform.machine().lower()

    log("[LocalGGUFLLMConnector] llama-cpp-python not found. Attempting auto-install...")

    # Detect hardware
    cmake_args = []
    if check_nvidia_gpu():
        log("NVIDIA GPU detected: Enabling CUDA acceleration.")
        cmake_args.append("-DLLAMA_CUBLAS=on")
    elif check_apple_metal():
        log("Apple Silicon (M1/M2/M3) detected: Enabling Metal acceleration.")
        cmake_args.append("-DLLAMA_METAL=on")
    else:
        log("💻 No compatible GPU found: Installing CPU-only version.")

    # Build pip install command
    cmd = [sys.executable, "-m", "pip", "install", "llama-cpp-python"]
    if cmake_args:
        cmd += [f"--config-settings=cmake_args={';'.join(cmake_args)}"]
    cmd += ["--force-reinstall", "--no-cache-dir"]

    try:
        log(f"📦 Running: {' '.join(cmd)}")
        subprocess.check_call(cmd)
        log("llama-cpp-python installed successfully!")

        # Force reload after install
        import importlib
        import llama_cpp
        importlib.reload(llama_cpp)
        LLAMA_CPP_AVAILABLE = True
    except subprocess.CalledProcessError as e:
        log(f"Installation failed: {e}")
        LLAMA_CPP_AVAILABLE = False
    except Exception as e:
        log(f"Unexpected error during install: {e}")
        LLAMA_CPP_AVAILABLE = False

# Reuse the logging function from local_llm_connector
from .local_llm_connector import log # Ensure this is the final import for log

LOCAL_LLM_CATEGORY = "Local LLM Nodes/LLM Connectors" # Reuse or define new sub-category like "Local LLM Nodes/GGUF Connectors"

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
                             break # Found one, add the directory name and move to next item
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
    Models should be placed in ComfyUI/models/LLM/your_model.gguf or ComfyUI/models/LLM/your_model_folder/your_model.gguf.
    Requires 'llama-cpp-python': pip install llama-cpp-python (consider CUDA flags for GPU support)
    """
    @classmethod
    def INPUT_TYPES(cls):
        model_names = get_local_gguf_model_names()
        return {
            "required": {
                "local_gguf_model_name": (model_names, {"default": model_names[0] if model_names else "No_Local_GGUF_Models_Found"}),
                # --- Add device selection dropdown ---
                "device": (["GPU", "CPU"], {"default": "GPU"}), # Default to GPU
                # --- n_gpu_layers slider ---
                "n_gpu_layers": ("INT", {
                    "default": -1, # This default will be overridden based on 'device'
                    "min": -1, # -1 for 'all possible'
                    "max": 100, # Adjust based on typical model layer counts if needed
                    "step": 1,
                    "display": "slider"
                }),
                # --- Optional: Expose other common parameters ---
                # "n_threads": ("INT", {"default": 8, "min": 1, "max": 64}),
                # "n_ctx": ("INT", {"default": 4096, "min": 1, "max": 32768}), # Max depends on model
            },
        }

    RETURN_TYPES = ("LLMServiceConnector",) # Use the same type for compatibility
    FUNCTION = "get_connector"
    CATEGORY = LOCAL_LLM_CATEGORY # Should be defined earlier, e.g., "Local LLM Nodes/LLM Connectors"
    OUTPUT_NODE = False

    # --- Update get_connector to accept 'device' and set n_gpu_layers default dynamically ---
    def get_connector(self, local_gguf_model_name, device, n_gpu_layers): # , n_threads=8, n_ctx=4096): # Add other params if exposed
        """Returns a connector object for the selected GGUF model."""
        if not LLAMA_CPP_AVAILABLE:
            raise Exception("The 'llama-cpp-python' library is required for the Local GGUF LLM node but is not installed.")

        if local_gguf_model_name == "No_Local_GGUF_Models_Found":
             raise Exception("No local GGUF models found in models/LLM directory. Please place your .gguf files there.")

        # Determine the full path to the .gguf file
        base_models_dir = os.path.join(folder_paths.models_dir, "LLM")
        model_path = os.path.join(base_models_dir, local_gguf_model_name)

        gguf_file_path = None
        if os.path.isfile(model_path) and model_path.endswith('.gguf'):
            gguf_file_path = model_path
        elif os.path.isdir(model_path):
            # Search for .gguf file inside the directory
            for file in os.listdir(model_path):
                if file.endswith('.gguf'):
                    gguf_file_path = os.path.join(model_path, file)
                    break

        if not gguf_file_path or not os.path.exists(gguf_file_path):
             raise FileNotFoundError(f"[LocalGGUFLLMConnector] GGUF model file not found for selection: {local_gguf_model_name}")

        # --- Key Change: Set n_gpu_layers default based on device selection ---
        # Determine the final n_gpu_layers value to pass to the connector
        final_n_gpu_layers = n_gpu_layers

        # Apply default logic based on device and slider state
        # If device is CPU and n_gpu_layers is the slider's default (-1), assume user wants CPU mode
        if device == "CPU" and n_gpu_layers == -1:
            final_n_gpu_layers = 0
            log(f"[LocalGGUFLLMConnector] Device set to CPU, overriding n_gpu_layers to 0.")
        # If device is GPU and n_gpu_layers is the slider's default (-1), keep -1 for max offload
        elif device == "GPU" and n_gpu_layers == -1:
            final_n_gpu_layers = -1
            log(f"[LocalGGUFLLMConnector] Device set to GPU, keeping n_gpu_layers=-1 (max offload).")
        else:
            # If user explicitly set n_gpu_layers (slider moved), respect that value regardless of device dropdown
            log(f"[LocalGGUFLLMConnector] Using user-provided n_gpu_layers={n_gpu_layers} (device={device}).")
        # --- End of Key Change ---

        # --- Pass the potentially adjusted n_gpu_layers (and others if exposed) to the connector ---
        connector = LocalGGUFLLMServiceConnector(
            gguf_file_path,
            n_gpu_layers=final_n_gpu_layers
            # n_threads=n_threads, # Pass if exposed
            # n_ctx=n_ctx # Pass if exposed
        )
        # --- End of Passing Parameters ---
        return (connector,)


class LocalGGUFLLMServiceConnector:
    """
    Represents the connection to a specific local GGUF LLM using llama-cpp-python.
    The `gguf_file_path` should point directly to the .gguf file.
    """
    # --- Update __init__ to accept and store parameters ---
    def __init__(self, gguf_file_path, n_gpu_layers=-1): # , n_ctx=4096, n_threads=8):
        self.gguf_file_path = gguf_file_path
        self.n_gpu_layers = n_gpu_layers # Store n_gpu_layers
        # self.n_ctx = n_ctx # Store n_ctx if exposed
        # self.n_threads = n_threads # Store n_threads if exposed
        self.model = None
        self.is_loaded = False
    # --- End of Update ---

    def _load_model(self):
        """Loads the GGUF model using llama-cpp-python."""
        if self.is_loaded:
            return # Already loaded

        try:
            log(f"[LocalGGUFLLMConnector] Loading GGUF model from: {self.gguf_file_path}")
            # --- Model Loading Configuration for llama-cpp-python ---
            # Use the parameters passed from the node
            model_kwargs = {
                "n_ctx": getattr(self, 'n_ctx', 4096), # Use default if not passed/store, or just hardcode if not exposed
                "n_threads": getattr(self, 'n_threads', 8), # Use default if not passed/store, or just hardcode if not exposed
                # --- Key Change: Add n_gpu_layers ---
                "n_gpu_layers": self.n_gpu_layers, # Use the value set by the user/node (default adjusted by SetLocalGGUFLLMServiceConnector)
                # --- Optional: Adjust verbosity ---
                # "verbose": False, # Set to False to reduce llama.cpp logging if needed
            }
            # --- End of Key Change ---

            # --- Load Model ---
            # Crucially, pass the full path to the .gguf file and the kwargs
            self.model = llama_cpp.Llama(model_path=self.gguf_file_path, **model_kwargs)

            self.is_loaded = True
            log(f"[LocalGGUFLLMConnector] GGUF model loaded successfully.")
        except Exception as e:
            error_msg = f"[LocalGGUFLLMConnector] Failed to load GGUF model: {e}"
            log(error_msg)
            raise Exception(error_msg) from e # Chain the exception

    # ... (rest of the class: invoke method remains largely the same) ...
    def invoke(self, messages, **generation_kwargs):
        """
        Generates text using the local GGUF LLM based on the messages list.
        :param messages: List of message dictionaries (like OpenAI format).
                         Example: [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Hello!"}]
        :param generation_kwargs: Additional arguments for text generation (e.g., max_tokens, temperature).
        :return: The generated text string.
        """
        try:
            if not self.is_loaded:
                self._load_model() # Load model on first invocation

            if not self.model:
                raise Exception("[LocalGGUFLLMConnector] Local GGUF LLM model failed to load.")

            # Filter kwargs for llama-cpp-python chat completion
            chat_kwargs = {
                k: v for k, v in generation_kwargs.items()
                if k in ['temperature', 'top_p', 'top_k', 'max_tokens', 'presence_penalty', 'frequency_penalty', 'repeat_penalty', 'seed']
            }
            if 'max_new_tokens' in generation_kwargs and 'max_tokens' not in chat_kwargs:
                chat_kwargs['max_tokens'] = generation_kwargs['max_new_tokens']

            response = self.model.create_chat_completion(messages=messages, **chat_kwargs)
            generated_text = response['choices'][0]['message']['content']
            return generated_text.strip() if generated_text else ""

        except Exception as e:
            error_msg = f"[LocalGGUFLLMConnector] Error in invoke method: {str(e)}"
            log(error_msg)
            raise e # Re-raise

# ... (rest of the file) ...