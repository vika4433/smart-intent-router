import os
import yaml
from lm_studio_proxy.lm_studio_proxy import send_to_lm_studio


def classify_intent(text: str, config: dict) -> str:
    """
    Enhanced intent classification with better code detection.
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
    
    text_lower = text.lower()
    
    # Check for code patterns before invoking LLM
    if any(keyword in text_lower for keyword in code_keywords):
        return "code"
    if any(keyword in text_lower for keyword in math_keywords):
        return "math"
    
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
        return response.strip().lower()
    except Exception as e:
        print("Intent classification failed:", e)
        return "general"
