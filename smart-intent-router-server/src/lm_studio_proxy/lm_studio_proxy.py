import re
from typing import List, Dict, Any
import requests
import json

def send_to_lm_studio(
    model_name: str,
    messages: List[Dict[str, Any]],
    endpoint: str = "http://localhost:1234/v1/chat/completions",
    tools: List[Dict[str, Any]] = None,
    tool_choice: str = "auto",
    temperature: float = 0.7
) -> str:
    """
    Sends messages to LM Studio and returns the raw assistant response.
    Supports OpenAI function calling format with tools for function-calling models.
    No formatting or normalization is applied.
    """
    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature
    }
    
    # Add tools for function-calling models
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = tool_choice

    try:
        print(f"DEBUG: LM Studio payload: {json.dumps(payload, indent=2)}")
        response = requests.post(endpoint, json=payload)
        response.raise_for_status()
        response_data = response.json()
        
        print(f"DEBUG: LM Studio raw response: {json.dumps(response_data, indent=2)}")
        
        # Handle function calls in response
        message = response_data['choices'][0]['message']
        if tools and "tool_calls" in message:
            print(f"DEBUG: Found tool_calls in response: {message['tool_calls']}")
            # Return the entire message object for function call handling
            return message
        else:
            print(f"DEBUG: No tool_calls found, returning content: {message.get('content', '')}")
            # Return just the content for regular responses
            return message.get('content', '')
    except Exception as e:
        print(f"DEBUG: LM Studio error: {e}")
        return f"Error communicating with LM Studio: {e}"


def normalize_code_blocks(text: str) -> str:
    """
    Ensures all code blocks are properly wrapped using triple backticks with syntax highlighting.
    - Fixes inline or malformed code blocks
    - Adds ```python for code that looks like Python but isn't wrapped
    """
    text = fix_inline_code_blocks(text)
    text = wrap_unwrapped_python_code(text)
    return text


def fix_inline_code_blocks(text: str) -> str:
    """
    Converts single-line or inline triple backtick code to proper formatted blocks.
    Example:
        '``` def foo(): return bar ```' → '```python\ndef foo(): return bar\n```'
    """
    pattern = r"```(.*?)```"  # Matches content between triple backticks
    def replacer(match):
        code = match.group(1).strip()
        if not code.startswith("python"):
            return f"```python\n{code}\n```"
        return match.group(0)

    return re.sub(pattern, replacer, text, flags=re.DOTALL)


def wrap_unwrapped_python_code(text: str) -> str:
    """
    If the beginning of the message looks like a Python function or code snippet
    and is not already inside a code block, wrap it in a Python code block.
    """
    lines = text.strip().splitlines()
    if not lines:
        return text

    if lines[0].strip().startswith("def ") and "```" not in text:
        code_lines = []
        i = 0
        while i < len(lines) and lines[i].strip():
            code_lines.append(lines[i])
            i += 1
        wrapped = "```python\n" + "\n".join(code_lines) + "\n```"
        remainder = "\n".join(lines[i:])
        return f"{wrapped}\n\n{remainder}".strip()

    return text

def get_available_models_from_lm_studio(
    base_endpoint: str = "http://localhost:1234"
) -> List[Dict[str, Any]]:
    """
    Fetches the list of available models from LM Studio using the OpenAI-compatible API.
    Returns a list of model information dictionaries.
    """
    models_endpoint = f"{base_endpoint}/v1/models"
    
    try:
        response = requests.get(models_endpoint, timeout=5)
        response.raise_for_status()
        
        data = response.json()
        models = data.get("data", [])
        
        # Extract relevant model information
        available_models = []
        for model in models:
            model_info = {
                "id": model.get("id", ""),
                "name": model.get("id", ""),  # LM Studio uses 'id' as the model name
                "object": model.get("object", "model"),
                "created": model.get("created", 0),
                "owned_by": model.get("owned_by", "lm-studio")
            }
            available_models.append(model_info)
        
        return available_models
        
    except requests.exceptions.RequestException as e:
        print(f"Warning: Could not fetch models from LM Studio: {e}")
        return []
    except Exception as e:
        print(f"Error parsing models response from LM Studio: {e}")
        return []

def get_lm_studio_base_endpoint_from_config(config: Dict[str, Any]) -> str:
    """
    Extract the base LM Studio endpoint from the configuration.
    Assumes all models use the same LM Studio instance.
    """
    models = config.get("models", [])
    if models:
        # Get the first enabled model's endpoint and extract base URL
        for model in models:
            if model.get("enabled", True):
                endpoint = model.get("endpoint", "http://localhost:1234/v1/chat/completions")
                # Extract base URL (remove /v1/chat/completions part)
                if "/v1/chat/completions" in endpoint:
                    return endpoint.replace("/v1/chat/completions", "")
                elif "/v1/" in endpoint:
                    return endpoint.split("/v1/")[0]
    
    # Default fallback
    return "http://localhost:1234"
