import re
from typing import List, Dict, Any
import requests

def send_to_lm_studio(
    model_name: str,
    messages: List[Dict[str, Any]],
    endpoint: str = "http://localhost:1234/v1/chat/completions"
) -> str:
    """
    Sends messages to LM Studio and returns the raw assistant response.
    No formatting or normalization is applied.
    """
    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": 0.7
    }

    try:
        response = requests.post(endpoint, json=payload)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
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
