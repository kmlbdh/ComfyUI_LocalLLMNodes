# ComfyUI/custom_nodes/ComfyUI_LocalLLMNodes/local_gguf_llm_connector.py
import os
import folder_paths
# --- Library for GGUF local LLM inference ---
try:
    import llama_cpp
    # import llama_cpp.llama_tokenizer # Optional, remove if not used
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    # --- Simple logging utility for this package (reuse existing one) ---
    # Import here to avoid potential issues if log isn't defined yet if imported at top in case of circular import risks,
    # though usually safe. Ensure local_llm_connector.py defines 'log' early.
    from .local_llm_connector import log
    log("[LocalGGUFLLMConnector] Warning: llama-cpp-python library not found.")
    LLAMA_CPP_AVAILABLE = False
    llama_cpp = None

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
                # Optional parameters for llama-cpp-python loading
                # "n_ctx": ("INT", {"default": 4096, "min": 1, "max": 100000}),
                # "n_gpu_layers": ("INT", {"default": 0, "min": -1, "max": 100}), # -1 = all
            },
        }

    RETURN_TYPES = ("LLMServiceConnector",) # Use the same type for compatibility
    FUNCTION = "get_connector"
    CATEGORY = LOCAL_LLM_CATEGORY
    OUTPUT_NODE = False

    def get_connector(self, local_gguf_model_name): # , n_ctx=4096, n_gpu_layers=0):
        """Returns a connector object for the selected GGUF model."""
        if not LLAMA_CPP_AVAILABLE:
            raise Exception("The 'llama-cpp-python' library is required for the Local GGUF LLM node but is not installed.")

        if local_gguf_model_name == "No_Local_GGUF_Models_Found":
             raise Exception("No local GGUF models found in models/LLM directory. Please place your .gguf files there.")

        # Determine the base path selected by the user
        base_models_dir = os.path.join(folder_paths.models_dir, "LLM")
        model_path = os.path.join(base_models_dir, local_gguf_model_name)

        gguf_file_path = None
        # Case 1: User selected a name derived from a file directly in models/LLM (e.g., 'model' for 'models/LLM/model.gguf')
        # model_path would be 'models/LLM/model' (not a .gguf file itself)
        if os.path.isfile(model_path) and model_path.endswith('.gguf'):
            gguf_file_path = model_path
        # Case 2: User selected a name derived from a directory (e.g., 'ModelDir' for 'models/LLM/ModelDir/')
        # model_path would be 'models/LLM/ModelDir' (a directory)
        elif os.path.isdir(model_path):
            # Search for the .gguf file inside the directory
            for file in os.listdir(model_path):
                if file.endswith('.gguf'):
                    gguf_file_path = os.path.join(model_path, file)
                    break # Use the first .gguf file found

        # Final check: Ensure we found a valid .gguf file path
        if not gguf_file_path or not os.path.exists(gguf_file_path):
             raise FileNotFoundError(f"[LocalGGUFLLMConnector] GGUF model file not found for selection: '{local_gguf_model_name}'. Expected file: '{gguf_file_path}'")

        # Create and return the GGUF connector instance, passing the resolved .gguf file path
        connector = LocalGGUFLLMServiceConnector(gguf_file_path) # Pass kwargs like n_ctx, n_gpu_layers if added
        return (connector,)


class LocalGGUFLLMServiceConnector:
    """
    Represents the connection to a specific local GGUF LLM using llama-cpp-python.
    The `gguf_file_path` should point directly to the .gguf file.
    """
    def __init__(self, gguf_file_path): # , n_ctx=4096, n_gpu_layers=0):
        self.gguf_file_path = gguf_file_path
        self.model = None
        self.is_loaded = False
        # self.n_ctx = n_ctx
        # self.n_gpu_layers = n_gpu_layers

    def _load_model(self):
        """Loads the GGUF model using llama-cpp-python."""
        if self.is_loaded:
            return # Already loaded

        try:
            log(f"[LocalGGUFLLMConnector] Loading GGUF model from: {self.gguf_file_path}")
            # --- Model Loading Configuration for llama-cpp-python ---
            # Adjust these parameters based on your needs and system capabilities.
            model_kwargs = {
                "n_ctx": 4096,      # Context window size (adjust if needed, larger uses more memory)
                "n_threads": 8,     # Number of CPU threads to use
                # "n_threads_batch": 8, # For batch processing (if applicable)
                # --- GPU Acceleration (if llama-cpp-python was built with CUDA support) ---
                # "n_gpu_layers": self.n_gpu_layers, # Number of layers to offload to GPU (e.g., 33 for 7B models)
                # --- Verbosity ---
                # "verbose": False, # Set to False to reduce llama.cpp logging
            }

            # --- Load Model ---
            # Crucially, pass the full path to the .gguf file
            self.model = llama_cpp.Llama(model_path=self.gguf_file_path, **model_kwargs)

            self.is_loaded = True
            log(f"[LocalGGUFLLMConnector] GGUF model loaded successfully from: {self.gguf_file_path}")
        except Exception as e:
            error_msg = f"[LocalGGUFLLMConnector] Failed to load GGUF model from '{self.gguf_file_path}': {e}"
            log(error_msg)
            # Re-raise the exception to halt the node execution
            raise Exception(error_msg) from e # Chain the exception

    def invoke(self, messages, **generation_kwargs):
        """
        Generates text using the local GGUF LLM based on the messages list.
        :param messages: List of message dictionaries (like OpenAI format).
                         Example: [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Hello!"}]
        :param generation_kwargs: Additional arguments for text generation (e.g., max_tokens, temperature).
                                  These will be filtered for compatibility with llama-cpp-python.
        :return: The generated text string.
        """
        try:
            if not self.is_loaded:
                self._load_model() # Load model on first invocation

            if not self.model:
                raise Exception("[LocalGGUFLLMConnector] Local GGUF LLM model failed to load or is not initialized.")

            # --- Prepare generation parameters for llama-cpp-python ---
            # Filter the provided kwargs to only include ones accepted by create_chat_completion
            # Commonly accepted kwargs for llama-cpp-python (check its documentation for the latest)
            accepted_kwargs = [
                'temperature', 'top_p', 'top_k', 'max_tokens', 'presence_penalty',
                'frequency_penalty', 'repeat_penalty', 'seed', 'stop', 'stream',
                'mirostat_mode', 'mirostat_tau', 'mirostat_eta'
                # Add others as needed/allowed by llama-cpp-python's API
            ]
            filtered_kwargs = {k: v for k, v in generation_kwargs.items() if k in accepted_kwargs}

            # Handle 'max_new_tokens' if passed (common in Transformers, convert to 'max_tokens' for llama-cpp)
            # Note: The logic here prioritizes 'max_tokens' if both are somehow passed.
            if 'max_new_tokens' in generation_kwargs and 'max_tokens' not in filtered_kwargs:
                filtered_kwargs['max_tokens'] = generation_kwargs['max_new_tokens']
            # Example of handling 'seed' if passed directly (llama-cpp might use it differently internally)
            # The `seed` is often handled by setting the global random state before generation
            # if 'seed' in generation_kwargs and generation_kwargs['seed'] is not None:
            #     # llama_cpp.Llama.sample_seed might be relevant, or rely on global state
            #     pass # llama-cpp-python often handles seed within the generation call if passed

            # --- Call the LLM ---
            # Use the chat completion interface which is generally preferred and handles templates
            response = self.model.create_chat_completion(
                messages=messages,
                **filtered_kwargs # Pass the filtered and potentially adjusted kwargs
            )
            # --- Extract the generated text ---
            # llama-cpp-python's create_chat_completion returns a dict
            # response = {'id': '...', 'object': 'chat.completion', 'created': ..., 'model': '...', 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': '...'}, 'finish_reason': 'stop'}], 'usage': {...}}
            generated_text = response['choices'][0]['message']['content']
            if generated_text is None:
                generated_text = "" # Handle potential None if model produces no content

            # Ensure the output is a clean string
            final_text = generated_text.strip()
            return final_text # <-- Return ONLY the generated string

        except Exception as e:
            # Catch any error that occurred within the try block and log it
            error_msg = f"[LocalGGUFLLMConnector] Error in invoke method: {str(e)}"
            log(error_msg)
            # Re-raise the exception so the calling node (e.g., LocalKontextPromptGenerator) knows it failed
            raise e # Or raise Exception(error_msg) from e
