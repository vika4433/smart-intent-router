# src/mcp_server/server.py
from mcp.server.fastmcp import FastMCP
from intent_classifier.intent_classifier import classify_intent
from language_detector.language_detector import detect_language
from llm_selector.llm_selector import select_llm_model
from lm_studio_proxy.lm_studio_proxy import send_to_lm_studio
from response_handler.response_handler import handle_response


mcp = FastMCP("SmartIntentRouter")


@mcp.tool(title="Route Request")
async def route_request(text: str) -> dict:
    language = detect_language(text)
    intent = classify_intent(text)
    model = select_llm_model(intent, language)
    raw_response = send_to_lm_studio(model, text)
    return handle_response(raw_response, model, intent, language)

def run_server():
    mcp.run(host="0.0.0.0", port=8080, streamable=True)