# arabic_product_node.py

import hashlib
import os
import json

# --- Simple logging utility ---
def log(message):
    print(f"[ArabicProductDesc] {message}")

# --- Paths and Preset Setup (Same as original) ---
script_directory = os.path.dirname(os.path.abspath(__file__))
USER_PRESETS_FILE = os.path.join(script_directory, "user_kontext_presets.json")

# --- Built-in Presets for Arabic Product Descriptions ---
ARABIC_PRODUCT_PRESETS = {
    "عام - وصفي": {
        "system": (
            "أنت خبير تسويق إلكتروني. اكتب وصفًا احترافيًا باللغة العربية للمنتج بناءً على الوصف المرئي أو المعلومات المقدمة. "
            "يجب أن يتضمن الوصف: اسم المنتج (إن عُرف)، المكونات أو المواصفات، الفوائد الرئيسية، طريقة الاستخدام، الفئة المستهدفة، وأي ملاحظات أمان. "
            "استخدم لغة تسويقية جذابة، واضحة، وسلسة. لا تستخدم تنسيق Markdown أو عناوين فرعية. لا تضيف تعليقات خارجية."
        )
    },
    "عناية بالبشرة": {
        "system": (
            "أنت أخصائي عناية بالبشرة. اكتب وصفًا عربيًا احترافيًا لكريم أو سيروم بناءً على الوصف المرئي أو التقني. "
            "ركّز على المكونات الفعالة (مثل SPF، حمض الهيالورونيك)، الفوائد (ترطيب، حماية، تفتيح)، ومدى ملاءمته لأنواع البشرة. "
            "اذكر طريقة الاستخدام (قبل النوم، بعد التنظيف، إلخ)، وكم مرة يُستخدم. استخدم لغة جذابة لكن علمية."
        )
    },
    "طعام ومشروبات": {
        "system": (
            "أنت كاتب محتوى غذائي. اكتب وصفًا شهيًا وملهمًا بالعربية لمنتج غذائي أو شراب. "
            "صف الطعم، القوام، المكونات الطبيعية، القيمة الغذائية، ومتى يُفضل تناوله (صباحًا، بعد التمرين). "
            "استخدم لغة حسية ومشوقة. لا تبالغ. لا تستخدم تنسيق."
        )
    },
    "إلكترونيات": {
        "system": (
            "أنت خبير تقني. اكتب وصفًا عربيًا واضحًا لجهاز إلكتروني (مثل سماعة، شاحن، ساعة ذكية). "
            "اذكر المواصفات الأساسية (البطارية، التوصيل، الجودة)، الفوائد (راحة، دقة، كفاءة)، والجمهور المستهدف. "
            "اجعله سهل الفهم للمستخدم العادي مع لمسة احترافية."
        )
    }
}

def load_user_presets():
    """Load user-defined presets from JSON file."""
    if os.path.exists(USER_PRESETS_FILE):
        try:
            with open(USER_PRESETS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            log(f"Error loading user presets: {e}")
            return {}
    return {}

def get_all_product_presets():
    """Get combined built-in and user-defined presets."""
    all_presets = ARABIC_PRODUCT_PRESETS.copy()
    user_presets = load_user_presets()
    all_presets.update(user_presets)
    return all_presets

# --- Category for UI ---
PRODUCT_DESC_CATEGORY = "Local LLM Nodes/Product Description (Arabic)"


class ArabicProductDescriptionGenerator:
    """
    Generates a detailed Arabic product description using:
    - Optional image description
    - User instruction in Arabic
    - Selected preset (system prompt)
    - Local LLM via LLMServiceConnector
    """

    @classmethod
    def INPUT_TYPES(cls):
        all_presets = get_all_product_presets()
        default_preset = next(iter(all_presets.keys()), "") if all_presets else ""

        return {
            "required": {
                "llm_service_connector": ("LLMServiceConnector",),
                "user_instruction_ar": ("STRING", {
                    "default": "اكتب وصفًا تفصيليًا لهذا المنتج",
                    "multiline": True,
                    "tooltip": "التعليمات من المستخدم بالعربية، مثلاً: اكتب وصفًا تسويقيًا"
                }),
                "preset": (list(all_presets.keys()), {
                    "default": default_preset,
                    "tooltip": "اختر نمط الوصف (عام، عناية بالبشرة، إلخ)"
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "control_after_generate": True
                })
            },
            "optional": {
                "image_description": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "وصف الصورة أو التفاصيل التقنية عن المنتج (اختياري)"
                })
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("arabic_product_description",)
    FUNCTION = "generate_description"
    CATEGORY = PRODUCT_DESC_CATEGORY

    def generate_description(self, llm_service_connector, user_instruction_ar, preset, seed=None, image_description=None):
        # Load preset
        all_presets = get_all_product_presets()
        preset_data = all_presets.get(preset)
        if not preset_data:
            raise ValueError(f"النمط غير معروف: {preset}")

        # Safe string conversion
        def safe_str(s):
            return str(s) if s is not None else ""

        img_desc = safe_str(image_description).strip()
        user_inst = safe_str(user_instruction_ar).strip()

        # Build user content
        user_content_parts = []
        if img_desc:
            user_content_parts.append(f"وصف المنتج أو الصورة: {img_desc}")
        if user_inst:
            user_content_parts.append(f"التعليمات: {user_inst}")
        else:
            user_content_parts.append("التعليمات: اكتب وصفًا تفصيليًا للمنتج.")

        user_content = " ".join(user_content_parts).strip()
        if not user_content:
            user_content = "وصف المنتج: منتج على خلفية بيضاء. التعليمات: اكتب وصفًا تسويقيًا بالعربية."

        # Build messages
        messages = [
            {"role": "system", "content": preset_data["system"]},
            {"role": "user", "content": user_content}
        ]

        # Call LLM
        try:
            response = llm_service_connector.invoke(messages, seed=seed)
            return (response.strip(),)
        except Exception as e:
            error_msg = f"[خطأ] تعذر إنشاء الوصف: {str(e)}"
            log(error_msg)
            return (error_msg,)

    def is_changed(self, llm_service_connector, user_instruction_ar, preset, seed, image_description=None):
        """Ensure node re-runs only when inputs change."""
        try:
            hasher = hashlib.md5()
            hasher.update(str(user_instruction_ar).encode('utf-8'))
            hasher.update(preset.encode('utf-8'))
            hasher.update(str(seed).encode('utf-8'))
            hasher.update(str(image_description).encode('utf-8'))

            # Add preset system prompt
            all_presets = get_all_product_presets()
            preset_data = all_presets.get(preset)
            if preset_data and "system" in preset_data:
                hasher.update(preset_data["system"].encode('utf-8'))

            # Add connector state
            try:
                connector_state = llm_service_connector.get_state()
            except AttributeError:
                connector_state = str(llm_service_connector)
            hasher.update(connector_state.encode('utf-8'))

            return hasher.hexdigest()
        except Exception as e:
            log(f"is_changed error: {e}")
            return float("nan")