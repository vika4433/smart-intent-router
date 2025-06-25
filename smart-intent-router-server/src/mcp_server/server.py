# src/mcp_server/server.py
# Ensure src directory is in sys.path for module imports
import sys
from pathlib import Path
SRC_PATH = str(Path(__file__).parent.parent)
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from mcp.server.fastmcp import FastMCP
from intent_classifier.intent_classifier import classify_intent
from language_detector.language_detector import detect_language
from llm_selector.llm_selector import select_llm_model
from lm_studio_proxy.lm_studio_proxy import send_to_lm_studio
from response_handler.response_handler import handle_response
from utils.config_reloader import ConfigReloader, CONFIG_PATH

import json
import os
from dotenv import load_dotenv

mcp = FastMCP(name="SmartIntentRouter", host="0.0.0.0", port=8050) #, streamable=True)

# Path to the YAML config
reloader = ConfigReloader(CONFIG_PATH)

# Load environment variables from .env file
load_dotenv()
TRANSPORT = os.getenv("TRANSPORT", "sse")

from typing import List, Dict, Any

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
@mcp.tool()#name="Route Request")
async def route_request(messages: List[Dict[str, Any]]) -> dict:
    """
    Route a user request with conversational context (OpenAI-style messages) to the appropriate LLM.
    """
    # Find the latest user message
    def get_latest_user_message(messages):
        for msg in reversed(messages):
            if msg.get("role") == "user" and "content" in msg:
                return msg["content"]
        return ""

    latest_user_message = get_latest_user_message(messages)
    print(f"Received messages: {json.dumps(messages, ensure_ascii=False, indent=2)}")
    print(f"Latest user message: {latest_user_message}")

    if not latest_user_message:
        return {"error": "No user message found in messages."}

    language = detect_language(latest_user_message)
    print(f"Detected language: {language}")
    intent = classify_intent(latest_user_message)
    print(f"Classified intent: {intent}")
    model_info = select_llm_model(intent, language)

    if not model_info:
        return {"error": "No suitable LLM model found for this intent and language."}

    model_name = model_info["model_name"]
    endpoint = model_info["endpoint"]
    raw_response = send_to_lm_studio(model_name, messages, endpoint)
    return handle_response(raw_response, model_name, intent, language)

# Register resource handlers without calling as a function (no parentheses)
# @mcp.list_resources
# def list_resources():
#     return [
#         {
#             "uri": "router://intents",
#             "name": "Intents",
#             "description": "List of all supported user intents and their descriptions.",
#             "mimeType": "application/json"
#         },
#         {
#             "uri": "router://models",
#             "name": "Models",
#             "description": "List of all available models, their supported intents, languages, and status.",
#             "mimeType": "application/json"
#         },
#         {
#             "uri": "router://rules",
#             "name": "Routing Rules",
#             "description": "Special-case routing rules for intent/language/model selection.",
#             "mimeType": "application/json"
#         },
#         {
#             "uri": f"file://{CONFIG_PATH}",
#             "name": "Router Config YAML",
#             "description": "The raw YAML configuration for the smart intent router.",
#             "mimeType": "text/yaml"
#         }
#     ]

@mcp.read_resource
def read_resource(uri: str):
    config = reloader.get_config()
    if uri == "router://intents":
        return json.dumps(config.get("intents", []), ensure_ascii=False, indent=2)
    elif uri == "router://models":
        return json.dumps(config.get("models", []), ensure_ascii=False, indent=2)
    elif uri == "router://rules":
        return json.dumps(config.get("rules", []), ensure_ascii=False, indent=2)
    elif uri == f"file://{CONFIG_PATH}":
        with open(CONFIG_PATH, "r") as f:
            return f.read()
    else:
        raise ValueError("Resource not found")

def run_server(transport: str = "stdio"):    
    if transport == "stdio":
        print("Running server with stdio transport")
        mcp.run(transport="stdio")
    elif transport == "sse":
        print("Running server with SSE transport")
        mcp.run(transport="sse")
    else:
        raise ValueError(f"Unknown transport: {transport}")

# Run the server
if __name__ == "__main__":
    if TRANSPORT == "sse":
        run_server(transport="sse")
    else:
        run_server(transport="stdio")