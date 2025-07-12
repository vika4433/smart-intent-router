import os
import yaml
from lm_studio_proxy.lm_studio_proxy import send_to_lm_studio


def classify_intent(text: str, config: dict) -> str:
    """
    Enhanced intent classification with better keyword detection for code, math, translation, and creative writing.
    """
    # First check for obvious code patterns
    code_keywords = [
        'write code', 'python function', 'def ', 'import ',
        'reverse a string', 'algorithm', 'implement',
        'function that', 'code to', 'how to code'
    ]
    
    math_keywords = [
        'calculate', 'solve', 'equation', 'math', 'formula',
        'derivative', 'integral', 'sum of', 'product of'
    ]
    
    translate_keywords = [
        'translate', 'translation', 'to english', 'to french', 'to spanish', 'to german', 'to russian', 'to hebrew', 'to chinese',
        'in english', 'in french', 'in spanish', 'in german', 'in russian', 'in hebrew', 'in chinese'
    ]
    
    creative_writing_keywords = [
        'write a story', 'compose an email', 'poem', 'novel', 'short story', 'creative writing',
        'write an email', 'write a poem', 'write a novel', 'write a short story', 'write a letter', 'write a script', 'write a dialogue', 'write a blog post', 'write a review', 'write a speech', 'write a song', 'write lyrics', 'write a fairy tale', 'write a fable', 'write a limerick', 'write a haiku'
    ]
    
    text_lower = text.lower()
    
    # Check for patterns before invoking LLM
    if any(keyword in text_lower for keyword in code_keywords):
        return "code"
    if any(keyword in text_lower for keyword in math_keywords):
        return "math"
    if any(keyword in text_lower for keyword in translate_keywords):
        return "translation"
    if any(keyword in text_lower for keyword in creative_writing_keywords):
        return "creative writing"
    
    # Only proceed with LLM classification if no clear pattern found
    template = config.get("intent_classifier", {}).get("template", "")
    prompt = [
        {"role": "system", "content": template},
        {"role": "user", "content": text}
    ]
    
    model_key = config.get("intent_classifier", {}).get("model", "llama2")
    model_info = next((m for m in config.get("models", []) if m.get("name") == model_key), {})
    model_name = model_info.get("model_name")
    endpoint = model_info.get("endpoint")

    if not model_name or not endpoint:
        return "general"

    try:
        response = send_to_lm_studio(model_name, prompt, endpoint)
        resp = response.strip().lower()
        # Normalize LLM output to valid intent label
        # Remove quotes and prefix if present
        for intent_label in ["code", "math", "translation", "creative writing", "general"]:
            if resp == intent_label:
                return intent_label
            # Accept quoted or prefixed forms
            if resp.replace('"', '').replace("'", "") == intent_label:
                return intent_label
            if resp.startswith("intent:"):
                candidate = resp.split(":", 1)[1].strip().replace('"', '').replace("'", "")
                if candidate == intent_label:
                    return intent_label
        # If not matched, fallback to general
        return "general"
    except Exception as e:
        print("Intent classification failed:", e)
        return "general"
