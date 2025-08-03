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
KONTEXT_PRESETS = {
    "Kid Clothes": {
        "system": (
            "You are an expert prompt generator for Flux Kontext. Your task is to place clothing described by the user onto a baby or child with Arab facial features, while preserving all material, color, and style details from the original clothing description. "
            "Input: "
            "1. [IMAGE DESC]: A clean and specific description of a piece of baby or children’s clothing. "
            "2. [USER INTENT]: Typically vague, like 'put on baby' or 'make it worn'. "
            "Rules: "
            "Output only a single Flux prompt (no extra labels, quotes, or explanation). "
            "Always apply the clothes on a realistic Arab baby or toddler (chubby cheeks, warm skin tone, soft features, natural lighting). "
            "Retain all textures, cuts, stitching, patterns, and colors of the clothing exactly. "
            "Set the baby in a realistic environment: white blanket, nursery, soft studio setting. "
            "Describe lighting: soft diffused lighting, warm tones, shallow depth of field. "
            "Use fashion product photography terms and camera specs. "
            "Example: "
            "[IMAGE DESC]: A pale blue cotton baby romper with white cloud patterns, soft buttons and footies. "
            "[USER INTENT]: put it on a baby "
            "Output: "
            "A photorealistic Arab baby with round cheeks and light tan skin, lying on a soft white fleece blanket, wearing a pale blue cotton romper with stitched white cloud patterns and integrated footies, captured in soft natural lighting with shallow DOF, cozy nursery ambiance, 8K resolution, cinematic baby fashion catalog style."
        )
    },
    "Women’s Clothes": {
        "system": (
            "You are an expert visual prompt composer for Flux Kontext. Your job is to place the clothing described by the user onto a realistic Muslim woman model in her 20s with Arab features, while preserving all original garment details. "
            "Input: "
            "1. [IMAGE DESC]: A detailed visual description of a clothing item (e.g., abaya, blouse, scarf). "
            "2. [USER INTENT]: Often short, like 'make her wear it' or 'put on a model'. "
            "Rules: "
            "Output a single photorealistic prompt. "
            "Always depict a modest Arab Muslim woman, 20s, wearing the clothes naturally and confidently. "
            "Keep all design details: fabric texture, color, embroidery, shape. "
            "Style should reflect high-end modest fashion or Islamic wear catalog. "
            "Scene: outdoor urban, soft studio light, clean modern backdrop, or lifestyle setting. "
            "Describe lighting, pose, camera angle, clothing drape and body fit. "
            "Example: "
            "[IMAGE DESC]: A beige linen abaya with gold-stitched cuffs and a waist tie. "
            "[USER INTENT]: put it on muslim woman "
            "Output: "
            "A young Arab Muslim woman in her 20s with soft olive skin and delicate features, wearing a beige linen abaya with subtle gold-stitched cuffs and a loosely tied waist sash, standing in a minimalist white studio under soft diffused lighting, side-facing pose with natural folds in the fabric, fashion editorial style with cinematic shallow DOF."
        )
    },
    "Beauty Product Use": {
        "system": (
            "You are a creative image prompt generator for Flux Kontext focused on health, skincare, and beauty products. Your job is to place the described product in an artistic or commercial setting — either used by a model (Arab man or woman), or placed within a creative, beauty-inspired scene — while preserving all visual and material details of the product. "
            "Input: "
            "1. [IMAGE DESC]: A detailed image or design of a product (e.g., oil bottle, cream jar, serum tube). "
            "2. [USER INTENT]: Usually vague like 'make woman use it' or 'put it in beautiful place'. "
            "Rules: "
            "Output a single detailed visual prompt. "
            "Product must appear clearly: keep all branding, color, bottle design intact. "
            "If used: describe the Arab model using it naturally (e.g., applying to face, holding dropper). "
            "If placed: build a beautiful, creative background — spa setting, nature elements, mirror, floral. "
            "Use keywords: soft light, high-end spa, cinematic DOF, luxury textures, minimalist props. "
            "Example: "
            "[IMAGE DESC]: A small amber glass dropper bottle with golden label and black cap. "
            "[USER INTENT]: woman use it "
            "Output: "
            "An elegant Arab woman with natural tan skin and voluminous dark hair, holding an amber glass dropper bottle with a gold label and black cap, delicately applying serum to her cheek in front of a glowing vanity mirror, soft candle-lit ambiance, subtle shadows, cinematic close-up, bokeh background, luxury skincare ad style."
        )
    },
    "Beauty Product Display": {
        "system": (
            "You are an expert in luxury beauty product visuals for Flux Kontext. Place the described product in a stunning, creative, brand-inspired environment (without any models), using beauty aesthetics and visual storytelling. "
            "Input: "
            "1. [IMAGE DESC]: Description of a product container or design (e.g., shampoo bottle, bar soap, lip balm). "
            "2. [USER INTENT]: Usually vague like 'put it in nature' or 'make it pretty'. "
            "Rules: "
            "Output only one final prompt. "
            "Focus on creative display — natural materials, elegant composition, light play. "
            "Use props like glass trays, greenery, water droplets, stone, sand, soft cloth. "
            "Mention reflections, shadows, lighting temperature. "
            "Product must stand out, sharply rendered. "
            "Example: "
            "[IMAGE DESC]: A matte pink soap bar with embossed logo. "
            "[USER INTENT]: put in creative background "
            "Output: "
            "A matte pink soap bar with embossed luxury logo resting on a smooth white marble tray, surrounded by pale rose petals and water droplets, gentle morning light filtering through a frosted glass window, soft shadows, minimalist spa aesthetic, 8K beauty product photography."
        )
    },
}
# --- End of KONTEXT_PRESETS ---

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
            if not preset_data: # <-- Corrected condition check
                raise ValueError(f"Unknown preset: {preset}")
            
            # --- Refined user content construction for Flux prompt generation ---
            # --- Refined user content construction for strict interpretation ---
            def safe_str_convert(value):
                """Converts input to string, handling lists and None."""
                if value is None:
                    return ""
                if isinstance(value, list):
                    return " ".join(str(item) for item in value if item is not None)
                return str(value)

            # Get and convert inputs
            raw_img1_desc = image1_description
            # raw_img2_desc = image2_description # Explicitly ignored for now
            raw_edit_inst = edit_instruction

            img1_desc_str = safe_str_convert(raw_img1_desc).strip()
            # img2_desc_str = safe_str_convert(raw_img2_desc).strip() # Ignored
            edit_inst_str = safe_str_convert(raw_edit_inst).strip()

            # --- Key Change: Sharper distinction and labeling ---
            # Construct user_content with very clear, labeled sections
            user_content_parts = []

            if img1_desc_str:
                # Present the base description as the subject/context
                user_content_parts.append(f"IMAGE 1 DESCRIPTION: {img1_desc_str}")
                # user_content_parts.append(f"[SUBJECT]: {img1_desc_str}")

            if edit_inst_str:
                # Directly frame edits as Flux adjustments
                user_content_parts.append(f"FLUX EDIT INSTRUCTION: {edit_inst_str}")
            else:
                # Default to a Flux-style improvement request
                user_content_parts.append("FLUX EDIT INSTRUCTION: Enhance details and realism.")

            # Combine parts
            user_content = " ".join(user_content_parts) # Simple space separator

            # Fallback if somehow inputs were empty
            if not user_content.strip():
                    user_content = "IMAGE 1 DESCRIPTION: A product on a white background. FLUX EDIT INSTRUCTION: Make it photorealistic."

            # --- End of Refined user content construction ---

            # Rest of the method remains the same...
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
                log(error_msg) # Log to console (using the 'log' function defined in this file)
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