# src/mcp_server/server.py

import os
import dotenv
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from mcp.server.fastmcp import FastMCP
from utils.config_loader import ConfigLoader, CONFIG_PATH
from intent_classifier.intent_classifier import classify_intent
from language_detector.language_detector import detect_language
from llm_selector.llm_selector import select_llm_model
from lm_studio_proxy.lm_studio_proxy import send_to_lm_studio
from response_handler.response_handler import handle_response
from utils.conversation_manager import ConversationManager
from utils.session_manager import SessionManager
from utils.adapters.mongo_conversation_repository import MongoConversationRepository

dotenv.load_dotenv()
config = ConfigLoader(CONFIG_PATH).get_config()

DEFAULT_INTENT = os.getenv("DEFAULT_INTENT", "general")

mcp = FastMCP(name="smart-intent-router", port=3000, json_response=False, stateless_http=False)

# Create a global repository and conversation manager instance
repository_instance = MongoConversationRepository()
conversation_manager = ConversationManager(repository_instance)

@mcp.tool()
async def classify_intent_only(message: str) -> dict:
    language = detect_language(message)
    intent = classify_intent(message, config) or DEFAULT_INTENT
    model_info = select_llm_model(intent, language, config)
    return {
        "intent": intent,
        "language": language,
        "model_info": model_info,
        "message": "Intent classified successfully. Use route_with_intent() to send to model."
    }

@mcp.tool()
async def route_request(
    message: str,
    user_id: str = "default_user",
    session_id: str = None,
    conversation_id: str = None
) -> dict:
    intent = classify_intent(message, config) or DEFAULT_INTENT

    if not session_id:
        session_id = SessionManager.get_user_session(user_id) or SessionManager.create_session(user_id)

    if not conversation_id:
        session_data = SessionManager.get_session(session_id)
        conversation_id = session_data["active_conversation_id"] if session_data else ConversationManager.create_conversation(session_id)

    return await route_with_intent(
        message=message,
        intent=intent,
        session_id=session_id,
        conversation_id=conversation_id,
        user_id=user_id
    )

@mcp.tool()
async def route_with_intent(
    message: str,
    intent: str,
    session_id: str = None,
    conversation_id: str = None,
    user_id: str = "default_user"
) -> dict:
    # Ensure session_id and user_id are set
    if not session_id:
        session_id = SessionManager.get_user_session(user_id) or SessionManager.create_session(user_id)
    if not user_id:
        # Try to resolve user_id from session
        session_data = SessionManager.get_session(session_id)
        user_id = session_data["user_id"] if session_data and "user_id" in session_data else "default_user"

    # Ensure conversation_id is set and user-centric
    if not conversation_id:
        session_data = SessionManager.get_session(session_id)
        conversation_id = session_data["active_conversation_id"] if session_data and "active_conversation_id" in session_data else conversation_manager.create_conversation(user_id)

    language = detect_language(message)
    model_info = select_llm_model(intent, language, config)
    if not model_info:
        return {"error": "No suitable LLM model found"}

    model_name = model_info["model_name"]
    endpoint = model_info["endpoint"]
    context_limit = model_info.get("context_length", 2000)

    # Use the global conversation_manager instance, but update max_tokens if needed
    conversation_manager.max_tokens = context_limit
    context_messages = conversation_manager.get_smart_context(conversation_id, intent, include_summary=True)

    def to_openai_message(m):
        content = m.get("content", "")
        if isinstance(content, dict):
            content = content.get("response", str(content))
        return {"role": m.get("role"), "content": content}

    openai_messages = []
    if config.get("system_templates", {}).get("markdown_response"):
        openai_messages.append({
            "role": "system",
            "content": config.get("system_templates", {}).get("markdown_response", "")
        })
    openai_messages += [to_openai_message(m) for m in context_messages]
    openai_messages.append({"role": "user", "content": message})

    conversation_manager.add_message(conversation_id, "user", message, intent, model_name)
    raw_response = send_to_lm_studio(model_name, openai_messages, endpoint)
    # Use only the arguments that handle_response expects (4 positional args)
    result = handle_response(
        raw_response,
        model_name,
        intent,
        language
    )
    # Ensure result is always a dict with a 'response' key
    if not isinstance(result, dict):
        result = {"response": result}
    conversation_manager.add_message(conversation_id, "assistant", result.get("response"), intent, model_name)
    return result

@mcp.tool()
async def create_conversation(user_id: str = None, session_id: str = None, title: str = None) -> dict:
    # Prefer user_id, but resolve from session_id if not provided
    if not user_id and session_id:
        session_data = SessionManager.get_session(session_id)
        user_id = session_data["user_id"] if session_data and "user_id" in session_data else None
    if not user_id:
        return {"error": "user_id is required to create a conversation."}
    conversation_id = conversation_manager.create_conversation(user_id, title)
    return {"conversation_id": conversation_id, "user_id": user_id, "session_id": session_id}

@mcp.tool()
async def delete_conversation(conversation_id: str) -> dict:
    success = ConversationManager.delete_conversation(conversation_id)
    return {"success": success}

@mcp.tool()
async def get_conversations(user_id: str) -> dict:
    """Return all conversations for a user (user-centric, not session-centric)."""
    return {"conversations": conversation_manager.get_conversations(user_id)}

@mcp.tool()
async def get_messages(conversation_id: str) -> dict:
    """Return a single conversation by its ID."""
    return {"conversation": conversation_manager.get_messages(conversation_id)}

@mcp.tool()
async def create_session(user_id: str) -> dict:
    session_id = SessionManager.create_session(user_id)
    return {"user_id": user_id, "session_id": session_id}

@mcp.tool()
async def check_session(session_id: str) -> dict:
    session = SessionManager.get_session(session_id)
    if not session:
        return {"active": False}
    return {"active": True, "expires_at": session["expires_at"].isoformat()}


def main():
    logging.basicConfig(level=logging.INFO)
    mcp.run(transport="streamable-http")

if __name__ == "__main__":
    main()
