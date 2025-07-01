from typing import Optional, Dict, Any
from pathlib import Path
#from utils.config_reloader import ConfigReloader, CONFIG_PATH


#config_reloader = ConfigReloader(CONFIG_PATH)

# import threading
# threading.Thread(target=config_reloader.auto_reload, daemon=True).start()

def select_llm_model(intent: str, language: str, config: dict) -> Optional[Dict[str, Any]]:
    """
    Select the best LLM model based on intent and language.

    Returns a dict with at least "model_name" and "endpoint".
    """
    #config = config_reloader.get_config()
    if not config:
        return None

    models = config.get("models", [])
    rules = config.get("rules", [])

    # Rule-based override first
    for rule in rules:
        if rule.get("intent") == intent and rule.get("language") == language:
            forced_model = next(
                (m for m in models if m.get("name") == rule.get("model") and m.get("enabled", True)),
                None,
            )
            if forced_model:
                return {
                    "model_name": forced_model["model_name"],
                    "endpoint": forced_model["endpoint"]
                }

    # Filter by intent/language/enabled
    candidates = [
        m for m in models
        if m.get("enabled", True)
        and intent in m.get("supported_intents", [])
        and language in m.get("supported_languages", [])
    ]
    if not candidates:
        return None

    candidates.sort(key=lambda m: m.get("weight", 1.0), reverse=True)
    best = candidates[0]
    return {
        "model_name": best["model_name"],
        "endpoint": best["endpoint"]
    }

# Example usage:
# result = select_llm_model("code", "ru")
# if result:
#     print("Model:", result["model_name"], "Endpoint:", result["endpoint"])
# else:
#     print("No suitable model found")
