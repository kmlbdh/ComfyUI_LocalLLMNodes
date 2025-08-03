# ComfyUI/custom_nodes/ComfyUI_LocalLLMNodes/__init__.py

# --- Conditional Import Logic ---
LOCAL_LLM_NODES_AVAILABLE = False
LOCAL_GGUF_NODES_AVAILABLE = False # Define this early

NODE_CLASS_MAPPINGS = {} # Initialize early
NODE_DISPLAY_NAME_MAPPINGS = {} # Initialize early

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

# --- Check for GGUF dependency ---
try:
    import llama_cpp
    GGUF_DEPS_AVAILABLE = True
except ImportError as e:
    print(f"[ComfyUI_LocalLLMNodes] GGUF dependency (llama-cpp-python) not found: {e}")
    GGUF_DEPS_AVAILABLE = False

# --- Attempt to import Hugging Face based nodes ---
if CORE_DEPS_AVAILABLE:
    try:
        # --- Import Node Classes ---
        from .local_llm_connector import SetLocalLLMServiceConnector
        from .local_prompt_generator import (
            LocalKontextPromptGenerator,
            AddUserLocalKontextPreset,
            RemoveUserLocalKontextPreset
        )
        LOCAL_LLM_NODES_AVAILABLE = True
        print("[ComfyUI_LocalLLMNodes] Hugging Face based LLM nodes are available.")
    except Exception as e:
        print(f"[ComfyUI_LocalLLMNodes] Error importing transformers-based node classes: {e}")
else:
    print("[ComfyUI_LocalLLMNodes] Hugging Face based LLM nodes will NOT be available due to missing core dependencies (transformers, torch).")

# --- Attempt to import GGUF based nodes ---
if GGUF_DEPS_AVAILABLE:
    try:
        from .local_gguf_llm_connector import SetLocalGGUFLLMServiceConnector
        LOCAL_GGUF_NODES_AVAILABLE = True
        print("[ComfyUI_LocalLLMNodes] GGUF LLM Connector node is available.")
    except Exception as e:
        print(f"[ComfyUI_LocalLLMNodes] Error importing GGUF node classes: {e}")
else:
    print("[ComfyUI_LocalLLMNodes] GGUF LLM Connector node will NOT be available due to missing dependency (llama-cpp-python).")

# --- Define Mappings ---
# Populate the dictionaries based on which nodes were successfully imported.

# Add Hugging Face based nodes if available
if LOCAL_LLM_NODES_AVAILABLE:
    NODE_CLASS_MAPPINGS.update({
        "SetLocalLLMServiceConnector": SetLocalLLMServiceConnector,
        "LocalKontextPromptGenerator": LocalKontextPromptGenerator,
        "AddUserLocalKontextPreset": AddUserLocalKontextPreset,
        "RemoveUserLocalKontextPreset": RemoveUserLocalKontextPreset,
    })
    NODE_DISPLAY_NAME_MAPPINGS.update({
        "SetLocalLLMServiceConnector": "Set Local LLM Service Connector (HuggingFace)",
        "LocalKontextPromptGenerator": "Local Kontext Prompt Generator",
        "AddUserLocalKontextPreset": "Add User Local Kontext Preset",
        "RemoveUserLocalKontextPreset": "Remove User Local Kontext Preset",
    })

# Add GGUF based nodes if available
if LOCAL_GGUF_NODES_AVAILABLE:
    NODE_CLASS_MAPPINGS.update({
        "SetLocalGGUFLLMServiceConnector": SetLocalGGUFLLMServiceConnector,
    })
    NODE_DISPLAY_NAME_MAPPINGS.update({
        "SetLocalGGUFLLMServiceConnector": "Set Local GGUF LLM Service Connector",
    })

# --- Define what ComfyUI sees ---
# ComfyUI looks for these specific dictionaries in __init__.py
# They must be defined at the top level of the module.
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]