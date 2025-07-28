# ComfyUI/custom_nodes/ComfyUI_LocalLLMNodes/__init__.py
# --- Conditional Import Logic ---
LOCAL_LLM_NODES_AVAILABLE = False
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# --- Check for core dependencies ---
# We need transformers and torch for the LLM nodes to work.
try:
    import transformers
    import torch
    # Optionally check for bitsandbytes here if you plan to use quantization immediately
    # import bitsandbytes as bnb
    CORE_DEPS_AVAILABLE = True
except ImportError as e:
    print(f"[ComfyUI_LocalLLMNodes] Core dependencies (transformers, torch) not found or not importable: {e}")
    CORE_DEPS_AVAILABLE = False

if CORE_DEPS_AVAILABLE:
    try:
        # --- Import Node Classes ---
        from .local_llm_connector import SetLocalLLMServiceConnector
        from .local_prompt_generator import (
            LocalKontextPromptGenerator,
            AddUserLocalKontextPreset,
            RemoveUserLocalKontextPreset
        )

        # --- Define Mappings ---
        # Using the class names directly as keys is standard.
        NODE_CLASS_MAPPINGS = {
            "SetLocalLLMServiceConnector": SetLocalLLMServiceConnector,
            "LocalKontextPromptGenerator": LocalKontextPromptGenerator,
            "AddUserLocalKontextPreset": AddUserLocalKontextPreset,
            "RemoveUserLocalKontextPreset": RemoveUserLocalKontextPreset,
        }

        NODE_DISPLAY_NAME_MAPPINGS = {
            "SetLocalLLMServiceConnector": "Set Local LLM Service Connector 🐑",
            "LocalKontextPromptGenerator": "Local Kontext Prompt Generator 🐑",
            "AddUserLocalKontextPreset": "Add User Local Kontext Preset 🐑",
            "RemoveUserLocalKontextPreset": "Remove User Local Kontext Preset 🐑",
        }

        LOCAL_LLM_NODES_AVAILABLE = True
        print("[ComfyUI_LocalLLMNodes] All nodes are available.")

    except Exception as e:
        print(f"[ComfyUI_LocalLLMNodes] Error importing node classes: {e}")
        # If import fails, NODE_CLASS_MAPPINGS and NODE_DISPLAY_NAME_MAPPINGS remain empty,
        # and ComfyUI won't register any nodes from this package.
else:
    print("[ComfyUI_LocalLLMNodes] Nodes will NOT be available due to missing core dependencies.")

# --- Define what ComfyUI sees ---
# ComfyUI looks for these specific dictionaries in __init__.py
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
