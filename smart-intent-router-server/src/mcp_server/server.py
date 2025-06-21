# src/mcp_server/server.py
from mcp.server.fastmcp import FastMCP
from intent_classifier.intent_classifier import classify_intent
from language_detector.language_detector import detect_language
from llm_selector.llm_selector import select_llm_model
from lm_studio_proxy.lm_studio_proxy import send_to_lm_studio
from response_handler.response_handler import handle_response
from utils.config_reloader import ConfigReloader
from pathlib import Path
import json

mcp = FastMCP("SmartIntentRouter")

# Path to the YAML config
reloader = ConfigReloader(ConfigReloader.CONFIG_PATH)

from typing import List, Dict, Any

@mcp.tool(title="Route Request")
async def route_request(messages: List[Dict[str, Any]]) -> dict:
    """
    Route a user request with conversational context (OpenAI-style messages) to the appropriate LLM.

    Args:
        messages (list): List of message dicts, each with at least:
            - role (str): One of the following roles:
                * "system"     — sets behavior/instructions for the assistant
                * "user"       — messages from the end user (the prompt)
                * "assistant"  — previous responses from the LLM/assistant
                * "tool"       — (optional, for tool-calling scenarios)
                * "function"   — (optional, for function calling scenarios)
            - content (str): Message text content

        Example:
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the capital of France?"},
                {"role": "assistant", "content": "The capital of France is Paris."}
            ]

    Returns:
        dict: The LLM's response, or an error message.
    """
    # Find the latest user message
    def get_latest_user_message(messages):
        for msg in reversed(messages):
            if msg.get("role") == "user" and "content" in msg:
                return msg["content"]
        return ""

    latest_user_message = get_latest_user_message(messages)

    if not latest_user_message:
        return {"error": "No user message found in messages."}

    language = detect_language(latest_user_message)
    intent = classify_intent(latest_user_message)
    model_info = select_llm_model(intent, language)

    if not model_info:
        return {"error": "No suitable LLM model found for this intent and language."}

    model_name = model_info["model_name"]
    endpoint = model_info["endpoint"]
    raw_response = send_to_lm_studio(model_name, messages, endpoint)
    return handle_response(raw_response, model_name, intent, language)



@mcp.list_resources()
async def list_resources():
    return [
        {
            "uri": "router://intents",
            "name": "Intents",
            "description": "List of all supported user intents and their descriptions.",
            "mimeType": "application/json"
        },
        {
            "uri": "router://models",
            "name": "Models",
            "description": "List of all available models, their supported intents, languages, and status.",
            "mimeType": "application/json"
        },
        {
            "uri": "router://rules",
            "name": "Routing Rules",
            "description": "Special-case routing rules for intent/language/model selection.",
            "mimeType": "application/json"
        },
        {
            "uri": f"file://{ConfigReloader.CONFIG_PATH}",
            "name": "Router Config YAML",
            "description": "The raw YAML configuration for the smart intent router.",
            "mimeType": "text/yaml"
        }
    ]


@mcp.read_resource()
async def read_resource(uri: str):
    config = reloader.get_config()
    if uri == "router://intents":
        return json.dumps(config.get("intents", []), ensure_ascii=False, indent=2)
    elif uri == "router://models":
        return json.dumps(config.get("models", []), ensure_ascii=False, indent=2)
    elif uri == "router://rules":
        return json.dumps(config.get("rules", []), ensure_ascii=False, indent=2)
    elif uri == f"file://{ConfigReloader.CONFIG_PATH}":
        with open(ConfigReloader.CONFIG_PATH, "r") as f:
            return f.read()
    else:
        raise ValueError("Resource not found")

def run_server():
    mcp.run(host="0.0.0.0", port=8080, streamable=True)
