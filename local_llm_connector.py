# ComfyUI/custom_nodes/ComfyUI_LocalLLMNodes/local_llm_connector.py
import os
import folder_paths

# --- Simple logging utility for this package ---
def log(message):
    print(f"[LocalLLMNodes] {message}")

# --- Library for local LLM inference ---
# Requires 'transformers' and 'torch'. These are checked in __init__.py.
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    # Optional: For quantization (requires 'bitsandbytes')
    # from transformers import BitsAndBytesConfig
    import torch
    HF_AVAILABLE = True
except ImportError:
    log("Warning: transformers or torch library not found. Local LLM node will not work.")
    HF_AVAILABLE = False
    AutoTokenizer = None
    AutoModelForCausalLM = None
    torch = None

# Define category directly for this package
LOCAL_LLM_CATEGORY = "Local LLM Nodes/LLM Connectors"

def get_local_llm_model_names():
    """Discovers subdirectories within models/LLM that could be local LLM models."""
    llm_models_dir = os.path.join(folder_paths.models_dir, "LLM")
    model_names = []

    if os.path.exists(llm_models_dir):
        try:
            # List directories inside models/LLM
            for item in os.listdir(llm_models_dir):
                item_path = os.path.join(llm_models_dir, item)
                if os.path.isdir(item_path):
                    # Basic check: assume any directory is a potential model
                    model_names.append(item)
        except Exception as e:
            log(f"Error scanning models/LLM directory: {e}")
    else:
        log(f"models/LLM directory not found: {llm_models_dir}")

    if not model_names:
        model_names = ["No_Local_Models_Found"] # Placeholder if no models found

    return model_names

class SetLocalLLMServiceConnector:
    """
    A node to select and prepare a connection to a local LLM model.
    Models should be placed in ComfyUI/models/LLM/your_model_name.
    Requires 'transformers' and 'torch': pip install transformers torch
    """
    @classmethod
    def INPUT_TYPES(cls):
        model_names = get_local_llm_model_names()
        return {
            "required": {
                "local_model_name": (model_names, {"default": model_names[0] if model_names else "No_Local_Models_Found"}),
            },
        }

    # --- CRITICAL: Match the type identifier expected by consuming nodes ---
    # Based on MieNodes structure (TextTranslator, KontextPromptGenerator), this should be "LLMServiceConnector"
    # This allows your node to connect to standard MieNodes prompt generators if desired.
    RETURN_TYPES = ("LLMServiceConnector",)
    FUNCTION = "get_connector"
    CATEGORY = LOCAL_LLM_CATEGORY
    OUTPUT_NODE = False

    def get_connector(self, local_model_name):
        """
        Returns a connector object that can invoke the selected local model.
        The actual model loading happens within the connector's invoke method.
        """
        if not HF_AVAILABLE:
             raise Exception("The 'transformers' and 'torch' libraries are required for the Local LLM node but are not installed.")

        if local_model_name == "No_Local_Models_Found":
             raise Exception("No local LLM models found in models/LLM directory. Please place your models there.")

        model_path = os.path.join(folder_paths.models_dir, "LLM", local_model_name)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Selected local LLM model path not found: {model_path}")

        # Create and return the connector instance. Model loading is deferred.
        connector = LocalLLMServiceConnector(model_path)
        return (connector,)


class LocalLLMServiceConnector:
    """
    Represents the connection to a specific local LLM.
    Handles loading (on first use) and invocation.
    Compatible with TextTranslator and KontextPromptGenerator via the 'invoke' method.
    The type identifier used by nodes expecting this connector is "LLMServiceConnector".
    """
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.is_loaded = False

    def _load_model(self):
        """Loads the model and tokenizer if not already loaded."""
        if self.is_loaded:
            return # Already loaded

        try:
            log(f"Loading local LLM model from: {self.model_path}")
            # --- Model Loading Configuration ---
            tokenizer_kwargs = {
                "trust_remote_code": True # Needed for some non-standard models
            }

            # --- Example Quantization Config (Uncomment and use if needed) ---
            # Requires 'bitsandbytes': pip install bitsandbytes
            # quantization_config = BitsAndBytesConfig(
            #     load_in_4bit=True,
            #     bnb_4bit_compute_dtype=torch.float16,
            #     bnb_4bit_use_double_quant=True,
            #     bnb_4bit_quant_type="nf4"
            # )

            model_kwargs = {
                "trust_remote_code": True,
                # --- Add Quantization Config if using ---
                # "quantization_config": quantization_config, # <-- Add for quantization
                # --- Device Placement ---
                "device_map": "auto", # Automatic device placement (CPU/GPU) - Often crucial with quantization
                 # --- Precision (if not using quantization config) ---
                 # "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
            }

            # --- Load Tokenizer ---
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, **tokenizer_kwargs)
            # Handle models without a pad token (common with Llama-based models)
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                log(f"Set pad_token_id to eos_token_id ({self.tokenizer.eos_token_id})")

            # --- Load Model ---
            # Note: device_map="auto" is often crucial here, especially with quantization
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path, **model_kwargs)

            # Manual device placement is usually handled by device_map="auto"
            # if "device_map" not in model_kwargs and torch.cuda.is_available():
            #     self.model.to("cuda")
            # elif "device_map" not in model_kwargs:
            #     self.model.to("cpu")

            self.is_loaded = True
            log(f"Local LLM model loaded successfully from: {self.model_path}")
        except Exception as e:
            error_msg = f"Failed to load local LLM model from {self.model_path}: {e}"
            log(error_msg)
            # Re-raise to stop execution if loading fails
            raise Exception(error_msg) from e # Chain the exception

    def invoke(self, messages, **generation_kwargs):
        """
        Generates text using the local LLM based on the messages list.
        Mimics the API expected by TextTranslator/KontextPromptGenerator.
        :param messages: List of message dictionaries (like OpenAI format).
                         Example: [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Hello!"}]
        :param generation_kwargs: Additional arguments for text generation (e.g., max_new_tokens, temperature, seed).
        :return: The generated text string.
        """
        try:
            if not self.is_loaded:
                self._load_model() # Load model on first invocation

            if not self.model or not self.tokenizer:
                raise Exception("Local LLM model or tokenizer failed to load.")

            # --- Format messages for the local model ---
            # Try using the tokenizer's chat template if available (more robust)
            prompt = ""
            try:
                # Many models come with a chat template
                prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                if not isinstance(prompt, str):
                     # Fallback if apply_chat_template behaves unexpectedly
                     raise ValueError("apply_chat_template did not return a string")
            except Exception as e:
                # Fallback to simple formatting if chat template fails or isn't available
                log(f"Falling back to simple prompt formatting: {e}")
                prompt_parts = []
                for message in messages:
                    role = message.get('role', 'user')
                    content = message.get('content', '').strip() # Strip content whitespace
                    if content: # Only add non-empty content lines
                        prompt_parts.append(f"{role.capitalize()}: {content}")
                prompt = "\n".join(prompt_parts)
                if prompt: # Add the final prompt indicator only if there's content
                     prompt += "\nAssistant:" # Encourage assistant response
                else:
                     prompt = "Assistant:" # Fallback if messages were empty

            if not prompt.strip():
                 log("Warning: Generated prompt is empty or whitespace.")
                 return "" # Return empty string if prompt is empty

            # --- Tokenize the prompt ---
            try:
                inputs = self.tokenizer(prompt, return_tensors="pt")
                # Consider moving inputs to model's device if not using device_map="auto"
                # if hasattr(self.model, 'device') and self.model.device.type != 'meta':
                #     inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            except Exception as e:
                 error_msg = f"Error tokenizing prompt: {e}"
                 log(error_msg)
                 raise Exception(error_msg) from e

            # --- Set default generation parameters ---
            # These defaults should be reasonable for prompt generation tasks
            default_kwargs = {
                "max_new_tokens": 250, # Slightly higher default for complex prompts
                "temperature": 0.7,    # Default creativity
                "do_sample": True,     # Enable sampling for variety
                "pad_token_id": self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id,
                # "eos_token_id": self.tokenizer.eos_token_id, # Optional: explicitly set EOS
            }
            # Update defaults with any provided kwargs (e.g., from consuming nodes)
            # Filter out kwargs that might cause issues if not explicitly supported
            filtered_kwargs = {k: v for k, v in generation_kwargs.items() if k in ['max_new_tokens', 'temperature', 'top_p', 'top_k', 'do_sample', 'num_beams', 'early_stopping', 'pad_token_id', 'eos_token_id', 'seed']}
            # Handle 'seed' if passed - PyTorch manual seed (affects stochastic operations)
            if 'seed' in generation_kwargs and generation_kwargs['seed'] is not None:
                try:
                    torch.manual_seed(generation_kwargs['seed'])
                except Exception as e:
                    log(f"Warning: Could not set seed: {e}")

            default_kwargs.update(filtered_kwargs)

            # --- Generate text ---
            try:
                self.model.eval()
                with torch.no_grad():
                    outputs = self.model.generate(**inputs, **default_kwargs)
            except Exception as e:
                 error_msg = f"Error during model.generate: {e}"
                 log(error_msg)
                 raise Exception(error_msg) from e

            # --- Decode the generated tokens ---
            try:
                # Extract only the newly generated part (skip the input prompt tokens)
                generated_tokens = outputs[:, inputs['input_ids'].shape[-1]:]
                generated_text = self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
                # Ensure the output is a clean string
                final_text = generated_text.strip()
                return final_text # <-- Return ONLY the generated string

            except Exception as e:
                 error_msg = f"Error decoding generated tokens: {e}"
                 log(error_msg)
                 raise Exception(error_msg) from e

        except Exception as e:
            # Catch any error that occurred within the try block and log it
            error_msg = f"Error in invoke method: {str(e)}"
            log(error_msg)
            # Re-raise the exception so the calling node knows it failed
            raise e # Or raise Exception(error_msg) from e