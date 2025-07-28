# ComfyUI/custom_nodes/ComfyUI_LocalLLMNodes/local_prompt_generator.py
import hashlib
import os
import json

# --- Simple logging utility for this package ---
def log(message):
    print(f"[LocalPromptGen] {message}")

# --- Replicate necessary preset logic locally ---
# We need to replicate the core preset loading/management logic here.

# --- Define paths and functions for user presets ---
script_directory = os.path.dirname(os.path.abspath(__file__))
USER_PRESETS_FILE = os.path.join(script_directory, "user_kontext_presets.json")

# --- Replicate Built-in Presets (from MieNodes' prompt_generator.py) ---
# Ensure these match the original structure exactly.
KONTEXT_PRESETS = {
    "Kontext Standard": {
        "system": "You are an expert AI assistant specializing in generating detailed, creative prompts for image generation models. Your task is to take user-provided image descriptions and edit instructions, then synthesize them into a single, highly descriptive prompt optimized for image generation. Focus on integrating key visual elements like character features, clothing details, scene setting, lighting, and artistic style. Ensure the final prompt is concise, avoids redundancy, and clearly conveys the user's intent for the desired image output."
    },
    "Kontext Detailed": {
        "system": "You are a master prompt engineer for AI image generators. Your role is to meticulously craft prompts by deeply analyzing user inputs. Given descriptions of two images (e.g., a person and clothing) and specific edit instructions, you must seamlessly merge these elements. Prioritize descriptive keywords for physical attributes, textures, colors, background environments, and artistic influences. The resulting prompt should be rich in detail, logically structured, and highly effective at guiding the image generator to produce the envisioned scene with high fidelity."
    },
    "Kontext Minimalist": {
        "system": "You are an AI assistant focused on creating concise, clear prompts for image generation. Your task is to distill user-provided descriptions and edit instructions into a short, essential prompt. Identify the core subject, key action or interaction, and the most important visual style or setting. Eliminate unnecessary details and focus on the primary elements that define the scene. The output prompt should be direct and easy for the image generator to interpret accurately."
    },
    "Kontext Artistic Style Focus": {
        "system": "You are an AI prompt specialist with an emphasis on artistic style and rendering techniques. Users will provide descriptions of elements and specific edits. Your goal is to construct a prompt that heavily emphasizes the desired artistic style (e.g., 'oil painting', 'cyberpunk', 'watercolor', 'photorealistic'). Integrate the provided subject and scene details, but frame them within the context of the specified artistic approach. Highlight relevant techniques, color palettes, brushwork, or visual effects associated with that style to guide the image generator effectively."
    }
}

def load_user_presets():
    """Load user-defined presets from the JSON file."""
    if os.path.exists(USER_PRESETS_FILE):
        try:
            with open(USER_PRESETS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            log(f"Error loading user presets: {e}")
            return {}
    return {}

def get_all_kontext_presets():
    """Get a combined dictionary of built-in and user presets."""
    # Start with built-in presets
    all_presets = KONTEXT_PRESETS.copy()
    # Load and add user presets, potentially overriding built-ins if names clash (usually desired)
    user_presets = load_user_presets()
    all_presets.update(user_presets)
    return all_presets

# Define category directly for this package
LOCAL_PROMPT_CATEGORY = "Local LLM Nodes/Prompt Generators"

class LocalKontextPromptGenerator(object):
    """
    A version of KontextPromptGenerator designed to work specifically with
    the LocalLLMServiceConnector. Replicates the core prompt generation logic exactly.
    """
    @classmethod
    def INPUT_TYPES(cls):
        # --- Replicate INPUT_TYPES exactly ---
        all_presets = get_all_kontext_presets()
        # Ensure there's a default if the list is somehow empty
        default_preset_key = next(iter(all_presets.keys()), "") if all_presets.keys() else ""
        return {
            "required": {
                # --- CRITICAL: Use the standard LLM_SERVICE_CONNECTOR type ---
                # This matches the RETURN_TYPES of SetLocalLLMServiceConnector
                # and is the same type expected by the original KontextPromptGenerator.
                "llm_service_connector": ("LLMServiceConnector",),
                "image1_description": ("STRING", {"default": "", "multiline": True, "tooltip": "Describe the first image"}),
                "image2_description": ("STRING", {"default": "", "multiline": True, "tooltip": "Describe the second image"}),
                "edit_instruction": ("STRING", {"default": "", "multiline": True}),
                "preset": (list(all_presets.keys()), {"default": default_preset_key}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True,
                                 "tooltip": "The random seed used for creating the noise."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("kontext_prompt",)
    FUNCTION = "generate_kontext_prompt"
    CATEGORY = LOCAL_PROMPT_CATEGORY

    # --- Replicate the core generate_kontext_prompt method exactly ---
    def generate_kontext_prompt(self, llm_service_connector, image1_description, image2_description, edit_instruction, preset, seed=None):
        """
        Replicates the exact core logic from KontextPromptGenerator.generate_kontext_prompt.
        """
        # --- Exact replication of the original logic ---
        all_presets = get_all_kontext_presets()
        preset_data = all_presets.get(preset)
        # --- Fixed Syntax Error ---
        if not preset_data: # <-- Corrected condition
            raise ValueError(f"Unknown preset: {preset}")

        # 用户输入拼到user消息中，给LLM最大上下文
        user_content = ""
        if image1_description.strip():
            user_content += f"Image 1 (person) description: {image1_description.strip()}"
        if image2_description.strip():
            user_content += f"Image 2 (clothing) description: {image2_description.strip()}"
        if edit_instruction.strip():
            user_content += f"Edit instruction: {edit_instruction.strip()}"
        if not user_content.strip():
            user_content = "No additional image description or edit instruction provided."

        messages = [
            {"role": "system", "content": preset_data["system"]},
            {"role": "user", "content": user_content},
        ]

        # --- Key Change: Use the local connector's invoke method ---
        # Pass the messages and seed.
        try:
            kontext_prompt = llm_service_connector.invoke(messages, seed=seed)
            # Ensure the output is stripped, like the original
            return (kontext_prompt.strip(),)
        except Exception as e:
            # Handle potential errors during local LLM invocation
            error_msg = f"Error generating kontext prompt with local LLM: {str(e)}"
            log(error_msg) # Log to console
            # Return the error message as the prompt string so the workflow doesn't crash silently
            return (error_msg,)

    # --- Replicate the is_changed method for proper caching ---
    def is_changed(self, llm_service_connector, image1_description, image2_description, edit_instruction, preset, seed):
        """
        Replicates the exact core logic from KontextPromptGenerator.is_changed.
        """
        try:
            hasher = hashlib.md5()
            hasher.update(image1_description.encode('utf-8'))
            hasher.update(image2_description.encode('utf-8'))
            hasher.update(edit_instruction.encode('utf-8'))
            hasher.update(preset.encode('utf-8'))
            hasher.update(str(seed).encode('utf-8'))

            # Incorporate preset system prompt
            all_presets = get_all_kontext_presets()
            preset_data = all_presets.get(preset)
            # --- Fixed Syntax Error ---
            if preset_data and "system" in preset_data: # <-- Corrected condition and variable name
                hasher.update(preset_data["system"].encode('utf-8'))

            # Incorporate connector state (replicate original logic)
            # Original tries get_state(), then falls back to specific attributes.
            # For a local connector, str(connector) is suitable.
            try:
                # Try get_state if it exists on the local connector (unlikely for our simple one)
                connector_state = llm_service_connector.get_state()
            except AttributeError:
                # Fallback: Use a string representation of the connector object
                # This works if the connector's identity is tied to its instance/model path.
                connector_state = str(llm_service_connector)

            hasher.update(connector_state.encode('utf-8'))

            return hasher.hexdigest()
        except Exception as e:
            # If hashing fails, force re-run
            log(f"is_changed error: {e}")
            return float("nan") # Always re-run

# --- Replicate the AddUserKontextPreset and RemoveUserKontextPreset classes ---
# These manage the local user presets file.

class AddUserLocalKontextPreset: # <-- Changed class name prefix
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preset_name": ("STRING", {"default": ""}),
                "system_prompt": ("STRING", {"default": "", "multiline": True}),
            }
        }

    RETURN_TYPES = ("BOOLEAN", "STRING")
    RETURN_NAMES = ("success", "log")
    FUNCTION = "add_preset"
    CATEGORY = LOCAL_PROMPT_CATEGORY

    def add_preset(self, preset_name, system_prompt):
        import datetime
        if not preset_name or not system_prompt:
            log = "Preset name and system prompt must not be empty."
            return (False, log)

        user_presets = load_user_presets()
        if preset_name in user_presets:
            log = f"Preset '{preset_name}' already exists (custom preset)."
            return (False, log)

        user_presets[preset_name] = {"system": system_prompt}
        # Save to the local user presets file
        try:
            with open(USER_PRESETS_FILE, "w", encoding="utf-8") as f:
                json.dump(user_presets, f, ensure_ascii=False, indent=2)
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log = f"Preset '{preset_name}' added successfully at {now}."
            return (True, log)
        except Exception as e:
            log = f"Error saving preset: {e}"
            return (False, log)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Always re-run when called to check for file changes
        return float("nan")

class RemoveUserLocalKontextPreset: # <-- Changed class name prefix
     @classmethod
     def INPUT_TYPES(cls):
         # Load current presets to populate the dropdown
         user_presets = load_user_presets()
         preset_names = list(user_presets.keys())
         # Provide a default if the list is empty
         default_name = preset_names[0] if preset_names else ""
         return {
             "required": {
                 "preset_name": (preset_names, {"default": default_name}),
             }
         }

     RETURN_TYPES = ("BOOLEAN", "STRING")
     RETURN_NAMES = ("success", "log")
     FUNCTION = "remove_preset"
     CATEGORY = LOCAL_PROMPT_CATEGORY

     def remove_preset(self, preset_name):
         import datetime
         user_presets = load_user_presets()
         if preset_name in user_presets:
             del user_presets[preset_name]
             try:
                 # Save to the local user presets file
                 with open(USER_PRESETS_FILE, "w", encoding="utf-8") as f:
                     json.dump(user_presets, f, ensure_ascii=False, indent=2)
                 now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                 log = f"Preset '{preset_name}' removed successfully at {now}."
                 return (True, log)
             except Exception as e:
                 log = f"Error saving after removal: {e}"
                 return (False, log)
         else:
             log = f"Preset '{preset_name}' not found in user presets."
             return (False, log)

     @classmethod
     def IS_CHANGED(cls, **kwargs):
         # Always re-run when called to check for file changes
         return float("nan")
