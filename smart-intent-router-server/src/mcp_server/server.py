# src/mcp_server/server.py

import os
import dotenv
import logging
import sys
import warnings
import time
from pathlib import Path
import json
import asyncio
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List
import asyncio
import uvicorn
import logging
import json
import warnings
import sys
import dotenv
import os
from threading import Thread

warnings.filterwarnings("ignore", category=RuntimeWarning)
sys.path.append(str(Path(__file__).resolve().parent.parent))
from mcp.server.fastmcp import FastMCP
from mcp import types
from utils.config_loader import ConfigLoader, CONFIG_PATH
from intent_classifier.intent_classifier import classify_intent
from language_detector.language_detector import detect_language
from llm_selector.llm_selector import select_llm_model
from lm_studio_proxy.lm_studio_proxy import send_to_lm_studio, get_available_models_from_lm_studio, get_lm_studio_base_endpoint_from_config
from response_handler.response_handler import handle_response
from utils.conversation_manager import ConversationManager
from utils.session_manager import SessionManager
from utils.adapters.mongo_conversation_repository import MongoConversationRepository

# Default intent definitions - used only when configuration is incomplete
DEFAULT_INTENTS = [
    {"name": "code", "description": "Requests related to programming, algorithms, or code examples"},
    {"name": "math", "description": "Requests involving mathematical calculations or problems"},
    {"name": "translation", "description": "Requests to translate text from one language to another"},
    {"name": "creative writing", "description": "Requests to write stories, emails, poems, or other creative content"},
    {"name": "general", "description": "All other requests"}
]

# Default intent classification examples
DEFAULT_INTENT_EXAMPLES = [
    ('Write a Python function to reverse a string', 'code'),
    ('What\'s the square root of 144?', 'math'),
    ('Translate \'Hello\' to French', 'translation'),
    ('Write a story about a brave knight', 'creative writing'),
    ('Tell me about the history of computers', 'general')
]

dotenv.load_dotenv()
config = ConfigLoader(CONFIG_PATH).get_config()

def validate_intent_configuration(strict_mode: bool = False) -> bool:
    """
    Validate that intent configuration is properly set up.
    
    Args:
        strict_mode: If True, raises exception on invalid configuration
    
    Returns:
        True if configuration is valid, False otherwise.
        Logs warnings for missing or incomplete configuration.
    
    Raises:
        ValueError: If strict_mode=True and configuration is invalid
    """
    configured_intents = config.get("intents", [])
    
    if not configured_intents:
        message = "No intents configured in config file. Using default intents."
        logging.warning(message)
        if strict_mode:
            raise ValueError(f"Configuration error: {message}")
        return False
        
    valid_intents = []
    for intent in configured_intents:
        if isinstance(intent, dict):
            name = intent.get("name", "")
            desc = intent.get("description", "")
            if name and desc:
                valid_intents.append(intent)
            else:
                message = f"Intent configuration incomplete: {intent}. Missing name or description."
                logging.warning(message)
                if strict_mode:
                    raise ValueError(f"Configuration error: {message}")
        else:
            message = f"Invalid intent format: {intent}. Expected dict with name and description."
            logging.warning(message)
            if strict_mode:
                raise ValueError(f"Configuration error: {message}")
    
    if not valid_intents:
        message = "No valid intents found in configuration. Please check your config file."
        logging.error(message)
        if strict_mode:
            raise ValueError(f"Configuration error: {message}")
        return False
        
    # Check for intent_classifier template
    intent_classifier_config = config.get("intent_classifier", {})
    template = intent_classifier_config.get("template", "")
    
    if not template.strip():
        logging.warning("No intent_classifier.template configured. Using generated template from intent descriptions.")
    
    logging.info(f"Configuration validation: Found {len(valid_intents)} valid intents")
    return True

# Validate configuration at startup
# Set STRICT_CONFIG_MODE=true to fail startup on configuration errors
STRICT_CONFIG_MODE = os.getenv("STRICT_CONFIG_MODE", "false").lower() == "true"
validate_intent_configuration(strict_mode=STRICT_CONFIG_MODE)

DEFAULT_INTENT = os.getenv("DEFAULT_INTENT", "general")

# Create FastAPI app for client-facing REST API
app = FastAPI(title="Smart Intent Router API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create FastMCP instance for internal MCP tools
mcp = FastMCP(name="smart-intent-router", port=3001, json_response=False, stateless_http=False)

# Create a global repository and conversation manager instance
repository_instance = MongoConversationRepository()
conversation_manager = ConversationManager(repository_instance)

async def get_models_from_config() -> list:
    """Get models from configuration file."""
    config_models = config.get("models", [])
    enabled_models = [m for m in config_models if m.get("enabled", True)]
    
    model_list = []
    for model in enabled_models:
        model_info = {
            "model_name": model.get("model_name"),
            "endpoint": model.get("endpoint"),
            "context_length": model.get("context_length", 2000),
            "supported_intents": model.get("supported_intents", []),
            "supported_languages": model.get("supported_languages", []),
            "weight": model.get("weight", 1.0),
            "is_orchestrator": model.get("is_orchestrator", False),  # PRESERVE orchestrator flag
            "source": "configuration"
        }
        model_list.append(model_info)
    
    return model_list

async def get_models_from_lm_studio() -> list:
    """Get models directly from LM Studio real-time API."""
    # Get LM Studio base endpoint from configuration
    base_endpoint = get_lm_studio_base_endpoint_from_config(config)
    
    # Fetch real-time model list from LM Studio
    lm_studio_models = get_available_models_from_lm_studio(base_endpoint)
    
    if not lm_studio_models:
        return []
    
    model_list = []
    for model in lm_studio_models:
        # Try to match with config for additional metadata
        config_model = None
        config_models = config.get("models", [])
        for cm in config_models:
            if cm.get("model_name") == model.get("id") or cm.get("model_name") == model.get("name"):
                config_model = cm
                break
        
        model_info = {
            "model_name": model.get("id", model.get("name")),
            "id": model.get("id"),
            "endpoint": f"{base_endpoint}/v1/chat/completions",
            "context_length": config_model.get("context_length", 4096) if config_model else 4096,
            "owned_by": model.get("owned_by", "lm-studio"),
            "source": "lm_studio_real_time",
            "status": "loaded and available"
        }
        
        # Add config-specific metadata if available
        if config_model:
            model_info.update({
                "supported_intents": config_model.get("supported_intents", []),
                "supported_languages": config_model.get("supported_languages", []),
                "weight": config_model.get("weight", 1.0)
            })
        
        model_list.append(model_info)
    
    return model_list

async def get_validated_model_info(model_name: str) -> dict:
    """Validate and get model info from LM Studio or config."""
    # Try LM Studio first
    lm_studio_models = await get_models_from_lm_studio()
    for model in lm_studio_models:
        if model.get("model_name") == model_name or model.get("id") == model_name:
            return model
    
    # Fallback to config models
    config_models = await get_models_from_config()
    for model in config_models:
        if model.get("model_name") == model_name:
            return model
    
    return None

async def get_orchestrator_model() -> dict:
    """Get the orchestrator model from configuration."""
    # Try LM Studio models first
    lm_studio_models = await get_models_from_lm_studio()
    for model in lm_studio_models:
        # Check if model is marked as orchestrator in config
        config_models = config.get("models", [])
        for cm in config_models:
            if (cm.get("model_name") == model.get("model_name") or 
                cm.get("model_name") == model.get("id")) and cm.get("is_orchestrator", False):
                return {
                    "model_name": model.get("model_name"),
                    "endpoint": model.get("endpoint"),
                    "context_length": model.get("context_length", 4096)
                }
    
    # Fallback to config models
    config_models = config.get("models", [])
    for model in config_models:
        if model.get("is_orchestrator", False) and model.get("enabled", True):
            return {
                "model_name": model.get("model_name"),
                "endpoint": model.get("endpoint"),
                "context_length": model.get("context_length", 4096)
            }
    
    # Final fallback - use first available model
    if lm_studio_models:
        return {
            "model_name": lm_studio_models[0].get("model_name"),
            "endpoint": lm_studio_models[0].get("endpoint"),
            "context_length": lm_studio_models[0].get("context_length", 4096)
        }
    
    return None

def parse_ai_response(response_text: str) -> dict:
    """Parse AI response for tool/function calls and other commands."""
    response_text = response_text.strip()
    
    # Check for function calls (OpenAI-style)
    if "detect_language(" in response_text or "get_models(" in response_text or "send_to_llm(" in response_text or "classify_intent(" in response_text:
        # Parse function call
        import re
        
        # Look for function calls
        function_pattern = r'(\w+)\((.*?)\)'
        matches = re.findall(function_pattern, response_text, re.DOTALL)
        
        for func_name, args_str in matches:
            if func_name in ["detect_language", "get_models", "send_to_llm", "classify_intent"]:
                # Parse arguments
                parameters = {}
                try:
                    # Simple parameter parsing
                    if args_str.strip():
                        # Handle key=value format
                        arg_pairs = re.findall(r'(\w+)="([^"]*)"', args_str)
                        for key, value in arg_pairs:
                            parameters[key] = value
                        
                        # Handle positional arguments for simple cases
                        if not parameters and func_name == "detect_language":
                            # Extract quoted string
                            quoted_match = re.search(r'"([^"]*)"', args_str)
                            if quoted_match:
                                parameters["prompt"] = quoted_match.group(1)
                
                except Exception:
                    pass
                
                # Check for classified intent in the response
                intent_match = re.search(r'Intent:\s*(\w+)', response_text, re.IGNORECASE)
                result = {
                    "type": "function_call",
                    "function_name": func_name,
                    "parameters": parameters
                }
                
                if intent_match:
                    result["intent_classified"] = intent_match.group(1)
                
                return result
    
    # Check for direct answers that should be function calls - ENHANCED DETECTION
    direct_answer_indicators = [
        # Code-related indicators
        "```", "def ", "import ", "python", "code", "function", "program", 
        "algorithm", "solution", "script", "class ", "return ", "print(",
        
        # Response indicators
        "sure!", "here's", "here is", "certainly", "of course", "below is",
        "simple", "easy", "basic", "following", "example",
        
        # Technical indicators  
        "language", "detect", "models", "available", "prime", "numbers",
        "calculation", "loop", "while", "for", "if", "range("
    ]
    
    # Also check for markdown code blocks or explanations
    if ("```" in response_text or 
        response_text.strip().startswith("Sure!") or
        response_text.strip().startswith("Here's") or
        response_text.strip().startswith("Certainly") or
        "def " in response_text or
        "print(" in response_text):
        return {"type": "direct_answer_needs_correction", "content": response_text}
    
    if any(keyword in response_text.lower() for keyword in direct_answer_indicators):
        return {"type": "direct_answer_needs_correction", "content": response_text}
    
    # Check for tool calls (MCP-style)
    if "tool:" in response_text.lower() or "call tool:" in response_text.lower():
        return {"type": "tool_call", "tool_name": "unknown", "parameters": {}}
    
    # Check for list tools request
    if "list tools" in response_text.lower() or "available tools" in response_text.lower():
        return {"type": "list_tools"}
    
    # Default to final response
    return {"type": "final_response", "content": response_text}

@mcp.resource("resource://smart-intent-router/models")
async def get_models() -> str:
    """Provide list of available models, preferring LM Studio real-time data with config fallback."""
    
    # Try to get models from LM Studio first
    lm_studio_models = await get_models_from_lm_studio()
    
    if lm_studio_models:
        models_str = str(lm_studio_models)
        return f"""Available models in LM Studio (real-time data):
{models_str}

MODEL SELECTION GUIDANCE:
Analyze each model's capabilities and select based on:
1. supported_intents - match with your classified intent
2. supported_languages - match with detected language  
3. Model name hints:
   - "*coder*" or "*code*" models → code intent
   - "*math*" or "*wizard*" models → math intent
   - "llama" models → general/creative writing
4. weight (higher = better performance)
5. context_length (for long inputs)

All listed models are currently loaded and ready to use.
Use the exact model_name from the list when calling send_to_llm."""
    else:
        # Fallback to configuration if LM Studio is not available
        config_models = await get_models_from_config()
        models_str = str(config_models)
        return f"""Available models (from configuration - LM Studio not accessible):
{models_str}

MODEL SELECTION GUIDANCE:
Analyze each model's capabilities and select based on:
1. supported_intents - match with your classified intent
2. supported_languages - match with detected language  
3. Model name hints for specialization
4. weight (higher = better performance)

Use the exact model_name from the list when calling send_to_llm."""

@mcp.tool()
async def send_to_llm(
    model_name: str,
    user_message: str,
    system_prompt: str = None,
    conversation_id: str = None,
    user_id: str = "default_user"
) -> dict:
    """Send a message to a specific LLM model with optional system prompt and context."""
    
    print(f"🚀 DEBUG [send_to_llm]: Starting request to model '{model_name}' with message: '{user_message[:100]}{'...' if len(user_message) > 100 else ''}'")
    
    # Validate model exists using real-time data or configuration
    model_info = await get_validated_model_info(model_name)
    
    if not model_info:
        return {"error": f"Model '{model_name}' not found or not available in LM Studio"}
    
    endpoint = model_info["endpoint"]
    context_limit = model_info.get("context_length", 4096)
    
    # Build messages array
    openai_messages = []
    
    # Add system prompt if provided
    if system_prompt:
        openai_messages.append({
            "role": "system", 
            "content": system_prompt
        })
    
    # Add conversation context if conversation_id provided
    if conversation_id:
        conversation_manager.max_tokens = context_limit
        context_messages = conversation_manager.get_smart_context(conversation_id, "general", include_summary=True)
        
        def to_openai_message(m):
            content = m.get("content", "")
            if isinstance(content, dict):
                content = content.get("response", str(content))
            return {"role": m.get("role"), "content": content}
        
        openai_messages += [to_openai_message(m) for m in context_messages]
    
    # Add current user message
    openai_messages.append({"role": "user", "content": user_message})
    
    # Send to LM Studio
    try:
        print(f"📤 DEBUG [send_to_llm]: Sending {len(openai_messages)} messages to LM Studio endpoint: {endpoint}")
        for i, msg in enumerate(openai_messages):
            print(f"  Message {i+1}: {msg['role']} - {msg['content'][:50]}{'...' if len(msg['content']) > 50 else ''}")
        
        raw_response = send_to_lm_studio(model_name, openai_messages, endpoint)
        print(f"📥 DEBUG [send_to_llm]: Received raw response from LM Studio (length: {len(str(raw_response))})")
        print(f"📥 DEBUG [send_to_llm]: Response preview: {str(raw_response)[:200]}{'...' if len(str(raw_response)) > 200 else ''}")
        
        # Check if the response is an error message
        if isinstance(raw_response, str) and "Error communicating with LM Studio" in raw_response:
            raise Exception(raw_response)
        
        result = handle_response(raw_response, model_name, "general", "en")
        print(f"✅ DEBUG [send_to_llm]: Processed response successfully. Final result keys: {list(result.keys()) if isinstance(result, dict) else 'not dict'}")
        
        if not isinstance(result, dict):
            result = {"response": result}
        
        # Save to conversation if conversation_id provided
        if conversation_id:
            conversation_manager.add_message(conversation_id, "user", user_message, "general", model_name)
            conversation_manager.add_message(conversation_id, "assistant", result.get("response"), "general", model_name)
        
        return {
            "response": result.get("response"),
            "model_used": model_name,
            "success": True
        }
        
    except Exception as e:
        print(f"❌ DEBUG [send_to_llm]: Error occurred - {str(e)}")
        # If the requested model fails (e.g., not loaded due to resources), 
        # try fallback to the orchestrator model with appropriate system prompt
        if "model_not_found" in str(e) or "Failed to load model" in str(e) or "Error communicating with LM Studio" in str(e):
            fallback_model = "llama-2-7b-chat-hf-function-calling-v2"
            try:
                print(f"Model {model_name} failed to load, trying fallback model {fallback_model}")
                
                # Create appropriate system prompt for the fallback model based on the original intended model
                fallback_system_prompt = None
                if "coder" in model_name.lower() or "code" in model_name.lower():
                    fallback_system_prompt = "You are a helpful coding assistant. Provide clear, working code examples with explanations."
                elif "math" in model_name.lower() or "wizard" in model_name.lower():
                    fallback_system_prompt = "You are a helpful math assistant. Solve problems step by step and show your work clearly."
                else:
                    fallback_system_prompt = "You are a helpful assistant. Provide clear and informative responses."
                
                # Modify the messages for fallback - replace system prompt if one exists
                fallback_messages = openai_messages.copy()
                if fallback_messages and fallback_messages[0].get("role") == "system":
                    fallback_messages[0] = {"role": "system", "content": fallback_system_prompt}
                else:
                    fallback_messages.insert(0, {"role": "system", "content": fallback_system_prompt})
                
                raw_response = send_to_lm_studio(fallback_model, fallback_messages, endpoint)
                
                # Check if fallback also returns an error
                if isinstance(raw_response, str) and "Error communicating with LM Studio" in raw_response:
                    return {"error": f"Both {model_name} and fallback {fallback_model} failed: {raw_response}"}
                
                result = handle_response(raw_response, fallback_model, "general", "en")
                
                if not isinstance(result, dict):
                    result = {"response": result}
                
                # Save to conversation if conversation_id provided
                if conversation_id:
                    conversation_manager.add_message(conversation_id, "user", user_message, "general", fallback_model)
                    conversation_manager.add_message(conversation_id, "assistant", result.get("response"), "general", fallback_model)
                
                return {
                    "response": result.get("response"),
                    "model_used": f"{fallback_model} (fallback from {model_name})",
                    "success": True
                }
            except Exception as fallback_error:
                return {"error": f"Both {model_name} and fallback {fallback_model} failed: {str(fallback_error)}"}
        else:
            return {"error": f"Failed to get response from {model_name}: {str(e)}"}

@mcp.tool(name="classify_intent")
async def classify_intent_tool(message: str, conversation_context: str = None) -> dict:
    """Classify intent for a given message with optional conversation context."""
    
    # If conversation context is provided, combine it with the current message
    # to help with context-aware intent classification
    if conversation_context:
        # Create a combined message that includes context for better classification
        context_aware_message = f"Previous context: {conversation_context}\nCurrent message: {message}"
        intent = classify_intent(context_aware_message, config) or DEFAULT_INTENT
    else:
        intent = classify_intent(message, config) or DEFAULT_INTENT
        
    return {
        "intent": intent,
        "message": "Intent classified successfully.",
        "context_used": bool(conversation_context)
    }

@mcp.tool(name="detect_language")
async def detect_language_tool(prompt: str) -> dict:
    """Detect language for a given prompt."""
    language = detect_language(prompt)
    return {"language": language}

@mcp.tool(name="select_llm_model")
async def select_llm_model_tool(intent: str, language: str) -> dict:
    """Select LLM model for given intent and language."""
    model_info = select_llm_model(intent, language, config)
    if not model_info:
        return {"error": "No suitable LLM model found"}
    return {"model_info": model_info}

async def route_request_impl(
    message: str,
    user_id: str = "default_user",
    session_id: str = None,
    conversation_id: str = None,
    intent: str = None,
    language: str = None,
    llm: str = None
) -> dict:
    # Step 1: intent
    if not intent:
        intent = classify_intent(message, config) or DEFAULT_INTENT
    else:
        # Validate intent (simple check: must be in config intents)
        valid_intents = config.get("intents", [])
        if valid_intents and intent not in valid_intents:
            intent = classify_intent(message, config) or DEFAULT_INTENT
    # Step 2: language
    if not language:
        language = detect_language(message)
    else:
        # Validate language (simple check: must be in config languages)
        valid_languages = config.get("languages", [])
        if valid_languages and language not in valid_languages:
            language = detect_language(message)
    # Step 3: llm/model_name
    model_info = None
    if not llm:
        model_info = select_llm_model(intent, language, config)
    else:
        # First check if model is available in real-time from LM Studio
        validated_model = await get_validated_model_info(llm)
        if validated_model:
            # Convert to expected format for compatibility
            model_info = {
                "model_name": validated_model["model_name"],
                "endpoint": validated_model["endpoint"],
                "context_length": validated_model["context_length"]
            }
        else:
            # Fallback to traditional validation and model selection
            model_info = select_llm_model(intent, language, config)
    if not model_info:
        return {"error": "No suitable LLM model found"}
    model_name = model_info["model_name"]
    endpoint = model_info["endpoint"]
    context_limit = model_info.get("context_length", 2000)
    
    # Session and conversation logic
    if not session_id:
        session_id = SessionManager.get_user_session(user_id) or SessionManager.create_session(user_id)
    if not conversation_id:
        session_data = SessionManager.get_session(session_id)
        conversation_id = session_data["active_conversation_id"] if session_data and "active_conversation_id" in session_data else conversation_manager.create_conversation(user_id)
    conversation_manager.max_tokens = context_limit
    context_messages = conversation_manager.get_smart_context(conversation_id, intent, include_summary=True)
    def to_openai_message(m):
        content = m.get("content", "")
        if isinstance(content, dict):
            content = content.get("response", str(content))
        return {"role": m.get("role"), "content": content}
    openai_messages = []
    # Add the correct system message for the classified intent
    system_templates = config.get("system_templates", {})
    system_message = system_templates.get(intent, system_templates.get("general", ""))
    if system_message:
        openai_messages.append({
            "role": "system",
            "content": system_message
        })
    openai_messages += [to_openai_message(m) for m in context_messages]
    openai_messages.append({"role": "user", "content": message})
    conversation_manager.add_message(conversation_id, "user", message, intent, model_name)
    raw_response = send_to_lm_studio(model_name, openai_messages, endpoint)
    result = handle_response(
        raw_response,
        model_name,
        intent,
        language
    )
    if not isinstance(result, dict):
        result = {"response": result}
    conversation_manager.add_message(conversation_id, "assistant", result.get("response"), intent, model_name)
    return {
        "response": result.get("response"),
        "intent": intent,
        "language": language,
        "model": model_name,
        "model_info": model_info,
        "conversation_id": conversation_id,
        "session_id": session_id
    }


async def create_conversation_impl(user_id: str = None, session_id: str = None, title: str = None) -> dict:
    # Prefer user_id, but resolve from session_id if not provided
    if not user_id and session_id:
        session_data = SessionManager.get_session(session_id)
        user_id = session_data["user_id"] if session_data and "user_id" in session_data else None
    if not user_id:
        return {"error": "user_id is required to create a conversation."}
    conversation_id = conversation_manager.create_conversation(user_id, title)
    return {"conversation_id": conversation_id, "user_id": user_id, "session_id": session_id}

async def delete_conversation_impl(conversation_id: str) -> dict:
    success = ConversationManager.delete_conversation(conversation_id)
    return {"success": success}

async def get_conversations_impl(user_id: str) -> dict:
    """Return all conversations for a user (user-centric, not session-centric)."""
    return {"conversations": conversation_manager.get_conversations(user_id)}

async def get_messages_impl(conversation_id: str) -> dict:
    """Return a single conversation by its ID."""
    return {"conversation": conversation_manager.get_messages(conversation_id)}

async def create_session_impl(user_id: str) -> dict:
    session_id = SessionManager.create_session(user_id)
    return {"user_id": user_id, "session_id": session_id}

async def check_session_impl(session_id: str) -> dict:
    session = SessionManager.get_session(session_id)
    if not session:
        return {"active": False}
    return {"active": True, "expires_at": session["expires_at"].isoformat()}

async def delete_messages_impl(conversation_id: str) -> dict:
    """Clear all messages for a given conversation."""
    conversation_manager.repo.clear_messages(conversation_id)
    return {"success": True}

async def execute_tool_call(tool_name: str, parameters: dict, conversation_id: str = None, user_id: str = "default_user", user_message: str = None) -> dict:
    """Execute a tool call from the orchestrator model with strict validation."""
    
    print(f"🔧 DEBUG [execute_tool_call]: Executing tool: {tool_name} with parameters: {parameters}")
    
    # 🚨 CRITICAL: REJECT INVALID FUNCTION CALLS
    valid_functions = ["send_to_llm", "get_models", "detect_language", "classify_intent"]
    
    if tool_name not in valid_functions:
        print(f"🚨 CRITICAL ERROR: Orchestrator called INVALID function: {tool_name}")
        print(f"   Valid functions are: {valid_functions}")
        print(f"   REJECTING this function call!")
        return {"error": f"Invalid function '{tool_name}'. Only {valid_functions} are allowed."}
    
    if tool_name == "send_to_llm":
        # 🚨 STRICT VALIDATION: Ensure orchestrator uses the correct original user message
        final_user_message = parameters.get("user_message")
        
        # Validate that the user_message matches the original request
        if final_user_message and user_message and final_user_message != user_message:
            print(f"🚨 CRITICAL BUG: Orchestrator used WRONG user_message!")
            print(f"   Orchestrator sent: '{final_user_message}'")
            print(f"   Current request is: '{user_message}'")
            print(f"   🔧 FORCE-CORRECTING to use current request")
            final_user_message = user_message
        
        # If user_message is missing or empty, use the original user_message from context
        if not final_user_message:
            final_user_message = user_message
            print(f"🔧 DEBUG [execute_tool_call]: Using fallback user_message for send_to_llm: {user_message}")
        
        # 🔧 FINAL SAFETY CHECK: Always ensure we use the current request
        if final_user_message != user_message:
            print(f"🚨 FINAL SAFETY: Force-correcting user_message to current request")
            final_user_message = user_message
        
        if not final_user_message:
            return {"error": "user_message parameter is required for send_to_llm"}
        
        # 🚨 STRICT MODEL VALIDATION: Ensure model selection makes sense for the intent
        model_name = parameters.get("model_name")
        if user_message and model_name:
            # Basic intent detection for validation - use word boundaries to avoid partial matches
            import re
            user_lower = user_message.lower()
            
            # Check for code intent first (higher priority)
            code_patterns = [
                r'\b(write|create|build|develop|generate)\b.*\b(function|code|script|program|algorithm)\b',
                r'\b(python|javascript|java|c\+\+|html|css|sql)\b',
                r'\b(def |class |import |from |return)\b',
                r'\b(debug|fix|refactor|optimize)\b.*\b(code|function|script)\b'
            ]
            is_code = any(re.search(pattern, user_lower) for pattern in code_patterns)
            
            # Check for creative writing intent (emails, letters, stories, etc.)
            creative_patterns = [
                r'\b(write|compose|draft)\b.*\b(email|letter|message|story|poem|essay)\b',
                r'\b(help.*wording|help.*writing|help.*compose)\b',
                r'\b(email.*manager|message.*manager|letter.*boss)\b',
                r'\b(salary.*request|request.*salary|raise.*request)\b'
            ]
            is_creative = any(re.search(pattern, user_lower) for pattern in creative_patterns)
            
            # Check for math intent (only if not code or creative)
            if not is_code and not is_creative:
                math_patterns = [
                    r'\b(calculate|compute|solve|find)\b.*\b(equation|formula|number|sum|product)\b',
                    r'\b(square root|factorial|logarithm|derivative|integral)\b',
                    r'\b\d+\s*[\+\-\*\/]\s*\d+\b',  # Actual arithmetic expressions
                    r'\bwhat\s+is\s+\d+.*[\+\-\*\/].*\d+\b'
                ]
                is_math = any(re.search(pattern, user_lower) for pattern in math_patterns)
            else:
                is_math = False
            
            # Validate model selection
            if is_math and 'wizardmath' not in model_name.lower():
                print(f"🚨 MODEL VALIDATION ERROR: Math question should use wizardmath model!")
                print(f"   Question: '{user_message}'")
                print(f"   Orchestrator chose: '{model_name}'")
                print(f"   CORRECTING to: 'wizardmath-7b-v1.1'")
                model_name = "wizardmath-7b-v1.1"
            elif is_code and 'coder' not in model_name.lower() and 'qwen' not in model_name.lower():
                print(f"🚨 MODEL VALIDATION ERROR: Code question should use coding model!")
                print(f"   Question: '{user_message}'")
                print(f"   Orchestrator chose: '{model_name}'")
                print(f"   CORRECTING to: 'qwen2.5-coder-7b-instruct'")
                model_name = "qwen2.5-coder-7b-instruct"
            elif is_creative and 'llama' not in model_name.lower():
                print(f"🚨 MODEL VALIDATION ERROR: Creative writing should use general model!")
                print(f"   Question: '{user_message}'")
                print(f"   Orchestrator chose: '{model_name}'")
                print(f"   CORRECTING to: 'llama-2-7b-chat-hf-function-calling-v2'")
                model_name = "llama-2-7b-chat-hf-function-calling-v2"
        
        print(f"🔧 DEBUG [execute_tool_call]: send_to_llm will use user_message: {final_user_message}")
        print(f"🔧 DEBUG [execute_tool_call]: send_to_llm will use model: {model_name}")
        
        # Inject conversation_id and user_id from the request context (ignore any values from orchestrator)
        print(f"🔧 DEBUG [execute_tool_call]: Injecting conversation_id: {conversation_id}, user_id: {user_id}")
        
        print(f"🔧 DEBUG [execute_tool_call]: About to call send_to_llm function")
        result = await send_to_llm(
            model_name=model_name,  # Use the validated model name
            user_message=final_user_message,  # Use the verified user message
            system_prompt=parameters.get("system_prompt"),
            conversation_id=conversation_id,  # Use the conversation_id from request context
            user_id=user_id  # Use the user_id from request context
        )
        print(f"🔧 DEBUG [execute_tool_call]: send_to_llm result: {result}")
        return result
    elif tool_name == "get_models":
        models_data = await get_models()
        print(f"get_models result: {models_data[:200]}..." if len(str(models_data)) > 200 else f"get_models result: {models_data}")
        return {"models": models_data, "success": True}
    elif tool_name == "detect_language":
        # 🚨 STRICT VALIDATION: Ensure orchestrator uses correct parameters
        prompt_text = parameters.get("prompt")
        
        # Check if orchestrator used wrong parameter names
        if not prompt_text and parameters.get("type"):
            wrong_param = parameters.get("type")
            print(f"🚨 PARAMETER ERROR: Orchestrator used 'type' instead of 'prompt'!")
            print(f"   Orchestrator used: 'type': '{wrong_param}'")
            print(f"   Should be: 'prompt': '{user_message}'")
            print(f"   CORRECTING to use original user message")
            prompt_text = user_message
        
        # If prompt is missing or empty, use the original user_message
        if not prompt_text:
            prompt_text = user_message
            print(f"🔧 DEBUG [execute_tool_call]: Using fallback user_message for detect_language: {user_message}")
        
        # Validate that the prompt matches the original request
        if prompt_text and user_message and prompt_text != user_message:
            print(f"🚨 VALIDATION ERROR: Orchestrator used wrong prompt for detect_language!")
            print(f"   Orchestrator used: '{prompt_text}'")
            print(f"   Original request was: '{user_message}'")
            print(f"   CORRECTING to use original request")
            prompt_text = user_message
        
        if not prompt_text:
            return {"error": "prompt parameter is required for detect_language"}
        
        print(f"🔧 DEBUG [execute_tool_call]: Calling detect_language with prompt: {prompt_text}")
        return await detect_language_tool(prompt_text)
    elif tool_name == "classify_intent":
        # Check if classify_intent tool is enabled
        orchestrator_config = config.get("orchestrator", {})
        if orchestrator_config.get("classify_intent_tool_enabled", False):
            return await classify_intent_tool(
                message=parameters.get("message"),
                conversation_context=parameters.get("conversation_context")
            )
        else:
            return {"error": "classify_intent tool is disabled. Please classify intent manually using the provided guidance in the system prompt."}
    else:
        return {"error": f"Unknown tool: {tool_name}"}

async def route_request_ai_impl(
    user_message: str,
    user_id: str = "default_user", 
    session_id: str = None,
    conversation_id: str = None,
    max_iterations: int = 5,
    notification_callback=None
) -> dict:
    """
    AI-driven request router with dynamic orchestrator selection.
    
    The orchestrator is selected based on the is_orchestrator flag in the configuration.
    Falls back to the first enabled model if no orchestrator is explicitly marked.
    
    Workflow:
    1. Web client sends user prompt to server
    2. Server gets system prompt, adds user prompt, tools/resources info and sends to orchestrator
    3. Orchestrator classifies intent and requests available models from server
    4. Server runs get_models() and provides list back to orchestrator
    5. Orchestrator decides which model is best for the intent and requests send_to_llm
    6. Server sends request to selected model and provides response back to orchestrator
    7. Orchestrator provides final response and server sends it to web client
    """
    
    # Send initial notification
    if notification_callback:
        start_time = time.time()
        print(f"⏰ DEBUG [timing]: Process started at {start_time}")
        await notification_callback({
            "type": "process_started",
            "user_message": user_message,
            "message": f"🔍 **Starting AI routing process for:** {user_message[:50]}{'...' if len(user_message) > 50 else ''}",
            "timestamp": start_time
        })
    
    # Set up session and conversation
    if not session_id:
        session_id = SessionManager.get_user_session(user_id) or SessionManager.create_session(user_id)
    if not conversation_id:
        session_data = SessionManager.get_session(session_id)
        conversation_id = session_data["active_conversation_id"] if session_data and "active_conversation_id" in session_data else conversation_manager.create_conversation(user_id)
    
    # Get orchestrator model from configuration using is_orchestrator flag
    orchestrator_model = await get_orchestrator_model()
    
    if not orchestrator_model:
        if notification_callback:
            await notification_callback({
                "type": "error",
                "message": "❌ **Error:** No orchestrator model available"
            })
        return {"error": "No orchestrator model available"}
    
    # Send notification for orchestrator selection
    if notification_callback:
        await notification_callback({
            "type": "orchestrator_selected",
            "model": orchestrator_model["model_name"],
            "message": f"🧠 **Orchestrator selected:** {orchestrator_model['model_name']}"
        })
    
    print(f"DEBUG: Selected orchestrator model: {orchestrator_model}")
    
    # Get system prompt from configuration
    orchestrator_config = config.get("orchestrator", {})
    system_prompt = orchestrator_config.get("system_prompt")
    
    if not system_prompt:
        return {"error": "Orchestrator system prompt not configured. Please add 'orchestrator.system_prompt' to your configuration file."}
    
    print(f"DEBUG: System prompt length: {len(system_prompt)} characters")
    print(f"DEBUG: First 200 chars of system prompt: {system_prompt[:200]}...")
    
    # Check if classify_intent tool is enabled
    classify_intent_tool_enabled = orchestrator_config.get("classify_intent_tool_enabled", False)
    
    # For function-calling models, use proper OpenAI function calling format
    if "function-calling" in orchestrator_model.get("model_name", "").lower():
        # Get tools in OpenAI function calling format
        openai_tools = get_openai_function_tools()
        
        # Modify system prompt based on classify_intent_tool_enabled flag
        if classify_intent_tool_enabled:
            # Use the standard system prompt for function-calling models with classify_intent tool
            enhanced_system_prompt = system_prompt
        else:
            # Use the alternative system prompt without classify_intent tool
            enhanced_system_prompt = orchestrator_config.get("system_prompt_no_classify_tool")
            if not enhanced_system_prompt:
                # Fallback: modify the existing prompt
                enhanced_system_prompt = system_prompt.replace(
                    "STEP 1: Call classify_intent function first\n    Response format: classify_intent(message=\"write python code to reverse a string\")",
                    "STEP 1: Classify the intent yourself first, then provide result and continue\n    Response format: Intent: [intent] | detect_language(prompt=\"write python code to reverse a string\")"
                ).replace(
                    "STEP 2: After getting intent result, call detect_language function",
                    "STEP 2: After classifying intent, call detect_language function"
                ).replace(
                    "- classify_intent(message=\"text\")",
                    "- classify_intent: DISABLED - classify intent directly in your response"
                )
    else:
        # For non-function-calling models, use discovery approach
        available_tools = get_internal_tools()
        enhanced_system_prompt = system_prompt
        enhanced_system_prompt += "\n\nUse the available tools to route requests. Ask for tool list if needed."
        openai_tools = None
    
    # Build conversation history with context from previous messages
    conversation_history = [
        {"role": "system", "content": enhanced_system_prompt}
    ]
    
    # CRITICAL FIX: Do NOT add conversation context to orchestrator for function-calling models
    # Previous conversation examples contaminate the orchestrator and make it give direct answers
    # The orchestrator should only see: system prompt + current user message
    if conversation_id and not ("function-calling" in orchestrator_model.get("model_name", "").lower()):
        # Only add context for non-function-calling models
        conversation_manager.max_tokens = orchestrator_model.get("context_length", 4096)
        context_messages = conversation_manager.get_smart_context(conversation_id, "general", include_summary=True)
        
        # Convert context messages to OpenAI format and add to history
        def to_openai_message(m):
            content = m.get("content", "")
            if isinstance(content, dict):
                content = content.get("response", str(content))
            return {"role": m.get("role"), "content": content}
        
        # Add context messages to help orchestrator understand conversation flow
        context_openai_messages = [to_openai_message(m) for m in context_messages]
        conversation_history.extend(context_openai_messages)
    
    # Add current user message
    conversation_history.append({"role": "user", "content": user_message})
    
    # Track workflow state and enforce step sequence for function-calling models
    workflow_state = {
        "step_1_classify_intent": False,
        "step_2_detect_language": False, 
        "step_3_get_models": False,
        "step_4_send_to_llm": False
    }
    
    # Debug: Log the enhanced system prompt to see what the orchestrator is receiving
    print("\n" + "="*50)
    print("DEBUG: Enhanced system prompt being sent to orchestrator:")
    print("="*50)
    print(enhanced_system_prompt)
    print("="*50)
    print(f"User message: {user_message}")
    print("="*50)
    
    iterations = 0
    final_response = None
    
    while iterations < max_iterations:
        iterations += 1
        
        # Update workflow state based on conversation history for function-calling models
        if "function-calling" in orchestrator_model.get("model_name", "").lower() and openai_tools:
            # Check conversation history to determine completed steps
            step_1_done = False
            step_2_done = False  
            step_3_done = False
            step_4_done = False
            
            for message in conversation_history:
                if message.get("role") == "user" and "classify_intent result:" in message.get("content", ""):
                    step_1_done = True
                elif message.get("role") == "user" and "detect_language result:" in message.get("content", ""):
                    step_2_done = True
                elif message.get("role") == "user" and "get_models result:" in message.get("content", ""):
                    step_3_done = True
                elif message.get("role") == "user" and "send_to_llm result:" in message.get("content", ""):
                    step_4_done = True
            
            # Create explicit next step instruction based on current state
            next_step_prompt = ""
            
            if classify_intent_tool_enabled:
                # 4-step sequence: classify_intent → detect_language → get_models → send_to_llm
                if not step_1_done:
                    next_step_prompt = f"\n\n🚨 EXECUTE STEP 1 NOW: classify_intent(message=\"{user_message}\")"
                elif not step_2_done:
                    next_step_prompt = f"\n\n🚨 EXECUTE STEP 2 NOW: detect_language(prompt=\"{user_message}\")"
                elif not step_3_done:
                    next_step_prompt = "\n\n🚨 EXECUTE STEP 3 NOW: get_models()"
                elif not step_4_done:
                    # Don't hardcode model - let orchestrator choose based on classified intent
                    next_step_prompt = f"\n\n🚨 EXECUTE STEP 4 NOW: send_to_llm(model_name=\"[SELECT_APPROPRIATE_MODEL]\", user_message=\"{user_message}\")"
            else:
                # 3-step sequence: detect_language → get_models → send_to_llm
                # (intent classification happens in system prompt)
                if not step_2_done:  # Step 1 in no-tool mode
                    next_step_prompt = f"\n\n🚨 EXECUTE STEP 1 NOW: detect_language(prompt=\"{user_message}\")"
                elif not step_3_done:  # Step 2 in no-tool mode
                    next_step_prompt = "\n\n🚨 EXECUTE STEP 2 NOW: get_models()"
                elif not step_4_done:  # Step 3 in no-tool mode
                    # Don't hardcode model - let orchestrator choose based on classified intent
                    next_step_prompt = f"\n\n🚨 EXECUTE STEP 3 NOW: send_to_llm(model_name=\"[SELECT_APPROPRIATE_MODEL]\", user_message=\"{user_message}\")"
            
            # Inject explicit instruction into system prompt
            max_steps = 4 if classify_intent_tool_enabled else 3
            if next_step_prompt and iterations <= max_steps:
                # Create a fresh system prompt with the step instruction (don't modify original)
                current_system_prompt = enhanced_system_prompt + next_step_prompt
                conversation_history[0]["content"] = current_system_prompt
                print(f"DEBUG: Adding explicit step instruction: {next_step_prompt.strip()}")
            else:
                # Reset to original system prompt
                conversation_history[0]["content"] = enhanced_system_prompt
        
        # Get response from orchestrator
        try:
            print(f"DEBUG: Sending {len(conversation_history)} messages to orchestrator model {orchestrator_model['model_name']}")
            print(f"DEBUG: System message being sent: {conversation_history[0]['content'][:300]}...")
            
            # For function-calling models, use tools parameter
            if "function-calling" in orchestrator_model.get("model_name", "").lower() and openai_tools:
                print(f"DEBUG: Full tool set has {len(openai_tools)} tools")
                
                # FILTER TOOLS TO FORCE SEQUENTIAL EXECUTION
                # Adjust filtering based on classify_intent_tool_enabled setting
                filtered_tools = []
                
                if classify_intent_tool_enabled:
                    # 4-step sequence: classify_intent → detect_language → get_models → send_to_llm
                    if iterations == 1:
                        filtered_tools = [tool for tool in openai_tools if tool["function"]["name"] == "classify_intent"]
                        print(f"DEBUG: Step 1 (with classify tool) - Filtering to classify_intent only")
                    elif iterations == 2:
                        filtered_tools = [tool for tool in openai_tools if tool["function"]["name"] == "detect_language"]
                        print(f"DEBUG: Step 2 (with classify tool) - Filtering to detect_language only")
                    elif iterations == 3:
                        filtered_tools = [tool for tool in openai_tools if tool["function"]["name"] == "get_models"]
                        print(f"DEBUG: Step 3 (with classify tool) - Filtering to get_models only")
                    elif iterations == 4:
                        filtered_tools = [tool for tool in openai_tools if tool["function"]["name"] == "send_to_llm"]
                        print(f"DEBUG: Step 4 (with classify tool) - Filtering to send_to_llm only")
                    else:
                        filtered_tools = openai_tools
                        print(f"DEBUG: Using all tools as fallback (with classify tool)")
                else:
                    # 3-step sequence: detect_language → get_models → send_to_llm
                    # (intent classification happens in system prompt)
                    if iterations == 1:
                        filtered_tools = [tool for tool in openai_tools if tool["function"]["name"] == "detect_language"]
                        print(f"DEBUG: Step 1 (no classify tool) - Filtering to detect_language only")
                    elif iterations == 2:
                        filtered_tools = [tool for tool in openai_tools if tool["function"]["name"] == "get_models"]
                        print(f"DEBUG: Step 2 (no classify tool) - Filtering to get_models only")
                    elif iterations == 3:
                        filtered_tools = [tool for tool in openai_tools if tool["function"]["name"] == "send_to_llm"]
                        print(f"DEBUG: Step 3 (no classify tool) - Filtering to send_to_llm only")
                    else:
                        filtered_tools = openai_tools
                        print(f"DEBUG: Using all tools as fallback (no classify tool)")
                
                print(f"DEBUG: Sending {len(filtered_tools)} filtered tools to orchestrator")
                
                # Use "required" to force function calling, but let model choose which function
                if len(filtered_tools) == 1:
                    tool_choice = "required"  # Force a function call
                    print(f"DEBUG: Forcing function call with 'required' - available: {filtered_tools[0]['function']['name']}")
                else:
                    tool_choice = "auto"
                    print("DEBUG: Using auto tool choice with multiple tools")
                
                response = send_to_lm_studio(
                    orchestrator_model["model_name"],
                    conversation_history,
                    orchestrator_model["endpoint"],
                    tools=filtered_tools,
                    tool_choice=tool_choice,
                    temperature=0.2
                )
            else:
                print("DEBUG: Sending messages without tools to non-function-calling orchestrator")
                response = send_to_lm_studio(
                    orchestrator_model["model_name"],
                    conversation_history,
                    orchestrator_model["endpoint"],
                    temperature=0.2
                )
            
            # Handle different response types
            if isinstance(response, dict) and "tool_calls" in response:
                # This is a function call response
                tool_calls = response["tool_calls"]
                print(f"DEBUG: Received {len(tool_calls)} tool calls from orchestrator")
                
                for tool_call in tool_calls:
                    function_name = tool_call["function"]["name"]
                    function_args = json.loads(tool_call["function"]["arguments"])
                    
                    print(f"DEBUG: Executing function call: {function_name} with args: {function_args}")
                    
                    # Send notification for function call execution
                    if notification_callback:
                        current_time = time.time()
                        print(f"⏰ DEBUG [timing]: Function {function_name} starting at {current_time}")
                        await notification_callback({
                            "type": "function_call",
                            "function_name": function_name,
                            "parameters": function_args,
                            "message": f"🔧 Calling function: **{function_name}**",
                            "timestamp": current_time
                        })
                    
                    # Execute the function call
                    function_result = await execute_tool_call(function_name, function_args, conversation_id, user_id, user_message)
                    
                    # Send notification for function result
                    if notification_callback:
                        if function_name == "classify_intent":
                            intent = function_result.get("intent", "unknown")
                            print(f"🔔 DEBUG [orchestrator]: Sending notification for classify_intent with intent: {intent}")
                            await notification_callback({
                                "type": "intent_classified",
                                "intent": intent,
                                "message": f"🎯 Intent classified: **{intent}**"
                            })
                            print(f"✅ DEBUG [orchestrator]: Notification sent for classify_intent")
                        elif function_name == "detect_language":
                            language = function_result.get("language", "unknown")
                            await notification_callback({
                                "type": "language_detected",
                                "language": language,
                                "message": f"🌍 Language detected: **{language}**"
                            })
                        elif function_name == "get_models":
                            # Count actual models from LM Studio
                            lm_studio_models = await get_models_from_lm_studio()
                            models_count = len(lm_studio_models) if lm_studio_models else 0
                            await notification_callback({
                                "type": "models_retrieved",
                                "count": models_count,
                                "message": f"🤖 Retrieved {models_count} available models"
                            })
                        elif function_name == "send_to_llm":
                            model_used = function_result.get("model_used", "unknown")
                            print(f"🔔 DEBUG [orchestrator]: Sending notification for send_to_llm with model: {model_used}")
                            await notification_callback({
                                "type": "llm_called",
                                "model": model_used,
                                "message": f"🚀 Calling LLM: **{model_used}**"
                            })
                            print(f"✅ DEBUG [orchestrator]: Notification sent for send_to_llm")
                    
                    # For orchestrator, if this is a send_to_llm call, use the response as the final answer
                    if function_name == "send_to_llm" and function_result.get("success"):
                        final_response = function_result.get("response", "")
                        print(f"🏁 DEBUG [orchestrator]: Got final response from send_to_llm (length: {len(final_response)})")
                        print(f"🏁 DEBUG [orchestrator]: Final response preview: {final_response[:100]}{'...' if len(final_response) > 100 else ''}")
                        
                        # Send notification for final response
                        if notification_callback:
                            print(f"🔔 DEBUG [orchestrator]: Sending final_response notification")
                            await notification_callback({
                                "type": "final_response",
                                "response": final_response,
                                "message": "✅ **Final response generated**"
                            })
                            print(f"✅ DEBUG [orchestrator]: Final response notification sent")
                        break
                    
                    # Add function result to conversation
                    conversation_history.append({
                        "role": "assistant",
                        "content": response.get("content", ""),
                        "tool_calls": tool_calls
                    })
                    conversation_history.append({
                        "role": "tool", 
                        "tool_call_id": tool_call["id"],
                        "content": json.dumps(function_result)
                    })
                
                # If we found a final response, break out of the loop
                if final_response:
                    break
                    
                # Continue to next iteration to get final response
                continue
            else:
                # This is a text response - parse it for fallback handling
                response_text = response if isinstance(response, str) else response.get("content", "")
                
                print(f"\n--- ITERATION {iterations} ---")
                print(f"Orchestrator response: {response_text}")
                print("-" * 50)
                
                # Send notification for orchestrator response
                if notification_callback:
                    await notification_callback({
                        "type": "orchestrator_response",
                        "iteration": iterations,
                        "response": response_text
                    })
                
                conversation_history.append({"role": "assistant", "content": response_text})
                
                # Parse for tool/resource calls using MCP protocol format (fallback for non-function-calling models)
                parsed_response = parse_ai_response(response_text)
                
                print(f"Parsed response type: {parsed_response['type']}")
                if parsed_response["type"] == "tool_call":
                    print(f"Tool: {parsed_response['tool_name']}")
                    print(f"Parameters: {parsed_response['parameters']}")
                elif parsed_response["type"] == "function_call":
                    print(f"Function: {parsed_response['function_name']}")
                    print(f"Parameters: {parsed_response['parameters']}")
                print("-" * 50)                # Handle function calls (OpenAI-style for function-calling models)
                if parsed_response["type"] == "function_call":
                    # Check if intent was already classified in the response
                    if "intent_classified" in parsed_response:
                        # Intent was classified directly, simulate classify_intent result
                        intent = parsed_response["intent_classified"]
                        print(f"Intent already classified: {intent}")
                        
                        # Send notification for intent classification
                        if notification_callback:
                            await notification_callback({
                                "type": "intent_classified",
                                "intent": intent,
                                "message": f"🎯 Intent classified: **{intent}**"
                            })
                        
                        # Add a simulated classify_intent result to conversation history
                        conversation_history.append({
                            "role": "assistant",
                            "content": f"Intent classified directly: {intent}"
                        })
                        conversation_history.append({
                            "role": "user", 
                            "content": f"Function classify_intent result: {{\"intent\": \"{intent}\", \"message\": \"Intent classified successfully.\", \"context_used\": false}}"
                        })
                    
                    # Send notification for function call execution
                    if notification_callback:
                        await notification_callback({
                            "type": "function_call",
                            "function_name": parsed_response["function_name"],
                            "parameters": parsed_response["parameters"],
                            "message": f"🔧 Calling function: **{parsed_response['function_name']}**"
                        })
                    
                    # Execute the actual function call (detect_language, get_models, or send_to_llm)
                    # For send_to_llm, ensure user_message is provided and not empty
                    if parsed_response["function_name"] == "send_to_llm" and (
                        "user_message" not in parsed_response["parameters"] or 
                        not parsed_response["parameters"].get("user_message", "").strip()
                    ):
                        parsed_response["parameters"]["user_message"] = user_message
                        print(f"Added missing/empty user_message to send_to_llm: {user_message}")
                    
                    function_result = await execute_tool_call(parsed_response["function_name"], parsed_response["parameters"], conversation_id, user_id, user_message)
                    
                    # Send notification for function result
                    if notification_callback:
                        if parsed_response["function_name"] == "classify_intent":
                            intent = function_result.get("intent", "unknown")
                            print(f"🔔 DEBUG [orchestrator-fallback]: Sending notification for classify_intent with intent: {intent}")
                            await notification_callback({
                                "type": "intent_classified",
                                "intent": intent,
                                "message": f"🎯 Intent classified: **{intent}**"
                            })
                            print(f"✅ DEBUG [orchestrator-fallback]: Notification sent for classify_intent")
                        elif parsed_response["function_name"] == "detect_language":
                            language = function_result.get("language", "unknown")
                            await notification_callback({
                                "type": "language_detected",
                                "language": language,
                                "message": f"🌍 Language detected: **{language}**"
                            })
                        elif parsed_response["function_name"] == "get_models":
                            # Count actual models from LM Studio
                            lm_studio_models = await get_models_from_lm_studio()
                            models_count = len(lm_studio_models) if lm_studio_models else 0
                            await notification_callback({
                                "type": "models_retrieved",
                                "count": models_count,
                                "message": f"🤖 Retrieved {models_count} available models"
                            })
                        elif parsed_response["function_name"] == "send_to_llm":
                            model_used = function_result.get("model_used", "unknown")
                            await notification_callback({
                                "type": "llm_called",
                                "model": model_used,
                                "message": f"🚀 Calling LLM: **{model_used}**"
                            })
                    
                    # For orchestrator, if this is a send_to_llm call, use the response as the final answer
                    if parsed_response["function_name"] == "send_to_llm" and function_result.get("success"):
                        final_response = function_result.get("response", "")
                        
                        # Send notification for final response
                        if notification_callback:
                            await notification_callback({
                                "type": "final_response",
                                "response": final_response,
                                "message": "✅ **Final response generated**"
                            })
                        break
                    
                    # Add function result to conversation
                    conversation_history.append({
                        "role": "user", 
                        "content": f"Function {parsed_response['function_name']} result: {json.dumps(function_result)}"
                    })
                    continue
                
                elif parsed_response["type"] == "tool_call":
                    # Execute tool call
                    tool_result = await execute_tool_call(parsed_response["tool_name"], parsed_response["parameters"], conversation_id, user_id, user_message)
                    
                    # For orchestrator, if this is a send_to_llm call, use the response as the final answer
                    if parsed_response["tool_name"] == "send_to_llm" and tool_result.get("success"):
                        final_response = tool_result.get("response", "")
                        break
                    
                    # Add tool result to conversation
                    conversation_history.append({
                        "role": "user", 
                        "content": f"Tool {parsed_response['tool_name']} result: {json.dumps(tool_result)}"
                    })
                    continue
                
                elif parsed_response["type"] == "list_tools":
                    # Use internal helper function for consistency with MCP endpoints
                    available_tools = get_internal_tools()
                    conversation_history.append({
                        "role": "user",
                        "content": f"Available tools: {json.dumps(available_tools, indent=2)}"
                    })
                    continue
                
                elif parsed_response["type"] == "direct_answer_needs_correction":
                    # Handle direct answers that need to be corrected to function calls
                    if "function-calling" in orchestrator_model.get("model_name", "").lower():
                        print("CRITICAL ERROR: Function-calling orchestrator provided direct answer instead of function call!")
                        
                        # Determine the correct function call based on the user message
                        if "code" in user_message.lower() or "python" in user_message.lower() or "programming" in user_message.lower():
                            correct_call = f'send_to_llm(model_name="qwen2.5-coder-7b-instruct", user_message="{user_message}")'
                        elif "math" in user_message.lower() or "calculate" in user_message.lower() or "square root" in user_message.lower():
                            correct_call = f'send_to_llm(model_name="wizardmath-7b-v1.1", user_message="{user_message}")'
                        else:
                            correct_call = f'send_to_llm(model_name="llama-2-7b-chat-hf-function-calling-v2", user_message="{user_message}")'
                        
                        correction_message = f"""CRITICAL ERROR: You provided a direct answer but you MUST call a function.

You said: "{response_text[:100]}..."

But you should have said EXACTLY: {correct_call}

STRICT REQUIREMENT: Your response must be ONLY a function call, nothing else. Try again with the correct format."""
                        
                        conversation_history.append({
                            "role": "user",
                            "content": correction_message
                        })
                        continue
                    else:
                        # For non-function-calling models, treat as final response
                        final_response = parsed_response["content"]
                        break
                        
                elif parsed_response["type"] == "final_response":
                    # This is the final response
                    final_response = parsed_response["content"]
                    break
                
                else:
                    # Check if this is a direct answer from the orchestrator (which is wrong)
                    if iterations == 1 and not parsed_response["type"] in ["function_call", "tool_call"]:
                        # For function-calling models, this is a critical error
                        if "function-calling" in orchestrator_model.get("model_name", "").lower():
                            print("CRITICAL ERROR: Function-calling orchestrator provided direct answer instead of function call!")
                            
                            # Check if response contains code, explanations, or other direct content
                            response_lower = response_text.lower()
                            is_direct_answer = any([
                                "```" in response_text,  # Code blocks
                                "def " in response_lower,  # Function definitions
                                "print(" in response_lower,  # Print statements
                                "certainly" in response_lower,  # Polite responses
                                "here is" in response_lower,  # Direct explanations
                                "below is" in response_lower,  # Direct explanations
                                "this script" in response_lower,  # Code explanations
                                len(response_text) > 50  # Long responses are likely direct answers
                            ])
                            
                            if is_direct_answer:
                                # Determine the correct function call based on the user message
                                if "code" in user_message.lower() or "python" in user_message.lower() or "programming" in user_message.lower():
                                    correct_call = f'send_to_llm(model_name="qwen2.5-coder-7b-instruct", user_message="{user_message}")'
                                elif "math" in user_message.lower() or "calculate" in user_message.lower() or "square root" in user_message.lower():
                                    correct_call = f'send_to_llm(model_name="wizardmath-7b-v1.1", user_message="{user_message}")'
                                else:
                                    correct_call = f'send_to_llm(model_name="llama-2-7b-chat-hf-function-calling-v2", user_message="{user_message}")'
                                
                                correction_message = f"""CRITICAL ERROR: You provided a direct answer but you MUST call a function.

You said: "{response_text[:100]}..."

But you should have said EXACTLY: {correct_call}

STRICT REQUIREMENT: Your response must be ONLY a function call, nothing else. Try again with the correct format."""
                                
                                conversation_history.append({
                                    "role": "user",
                                    "content": correction_message
                                })
                                continue
                            else:
                                # If it's a very short response that might be a malformed function call, treat as final
                                final_response = response_text
                                break
                        else:
                            # For non-function-calling models, use resource-based approach
                            correction_message = """You provided a direct answer, but you must use tools instead. Follow this exact format:

Step 1: Call get_models tool to see available models:
TOOL_CALL: get_models

Step 2: Once you get the models, select the appropriate model for code tasks and use send_to_llm tool."""
                            
                            conversation_history.append({
                                "role": "user",
                                "content": correction_message
                            })
                            continue
                    else:
                        # Fallback - treat as final response
                        final_response = response_text
                        break
                
        except Exception as e:
            return {"error": f"Orchestration error: {str(e)}"}
    
    if not final_response:
        final_response = "Orchestration completed but no final response generated."
    
    # Send final completion notification
    if notification_callback:
        await notification_callback({
            "type": "stream_complete",
            "message": "✅ **AI routing process completed successfully!**"
        })
    
    # Save to conversation
    conversation_manager.add_message(conversation_id, "user", user_message, "orchestrated", orchestrator_model["model_name"])
    conversation_manager.add_message(conversation_id, "assistant", final_response, "orchestrated", orchestrator_model["model_name"])
    
    return {
        "response": final_response,
        "orchestrator_model": orchestrator_model["model_name"],
        "conversation_id": conversation_id,
        "session_id": session_id,
        "iterations": iterations,
        "workflow": "fully_orchestrated"
    }

def get_internal_tools():
    """
    Get internal tool metadata for the AI router.
    Returns the same structure as list_tools() but in dict format for easy use.
    Only includes essential tools that the LLM orchestrator needs.
    """
    tools = [
        {
            "name": "send_to_llm",
            "description": "Send a message to a specific LLM model with optional system prompt and context",
            "parameters": {
                "model_name": {"type": "string", "description": "Name of the model to use"},
                "user_message": {"type": "string", "description": "The user's message to send"},
                "system_prompt": {"type": "string", "description": "Optional system prompt for the model"}
                # Note: conversation_id and user_id are handled by the server automatically
            }
        },
        {
            "name": "get_models",
            "description": "Get list of available language models with their capabilities and status",
            "parameters": {}
        },
        {
            "name": "detect_language",
            "description": "Detect the language of a prompt", 
            "parameters": {
                "prompt": {"type": "string", "description": "Text to analyze for language"}
            }
        }
    ]
    
    # Add classify_intent tool only if enabled in configuration
    orchestrator_config = config.get("orchestrator", {})
    if orchestrator_config.get("classify_intent_tool_enabled", False):
        tools.append({
            "name": "classify_intent", 
            "description": "Classify the intent of a user message with optional conversation context for better accuracy",
            "parameters": {
                "message": {"type": "string", "description": "Message to classify"},
                "conversation_context": {"type": "string", "description": "Optional previous conversation context to improve classification accuracy"}
            }
        })
    
    return tools

def get_openai_function_tools():
    """
    Convert internal tools to OpenAI function calling format.
    Returns tools in the format expected by LM Studio's OpenAI-compatible API.
    """
    internal_tools = get_internal_tools()
    openai_tools = []
    
    for tool in internal_tools:
        # Convert required parameters
        required_params = []
        properties = {}
        
        for param_name, param_info in tool["parameters"].items():
            properties[param_name] = {
                "type": param_info["type"],
                "description": param_info["description"]
            }
            # Mark required parameters (assuming all non-optional ones are required)
            if "optional" not in param_info.get("description", "").lower():
                if param_name in ["model_name", "user_message", "prompt", "message", "intent"]:  # Core required params
                    required_params.append(param_name)
        
        openai_tool = {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required_params
                }
            }
        }
        
        openai_tools.append(openai_tool)
    
    return openai_tools

# MCP Protocol endpoints
async def list_tools():
    """List all available tools for dynamic discovery."""
    return [
        types.Tool(
            name="route_request",
            description="Route a user request using rule-based routing to determine the appropriate LLM service",
            inputSchema={
                "type": "object",
                "properties": {
                    "request": {
                        "type": "string",
                        "description": "The user request to route"
                    }
                },
                "required": ["request"]
            }
        ),
        types.Tool(
            name="route_request_ai",
            description="Route a user request using AI-driven intelligent routing with an LLM orchestrator",
            inputSchema={
                "type": "object",
                "properties": {
                    "request": {
                        "type": "string",
                        "description": "The user request to route"
                    }
                },
                "required": ["request"]
            }
        ),
        types.Tool(
            name="get_models_from_config",
            description="Get available models from configuration file",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        types.Tool(
            name="get_models_from_lm_studio",
            description="Get available models directly from LM Studio real-time API",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        types.Tool(
            name="get_conversation_history",
            description="Retrieve conversation history for a session",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "The session identifier"
                    }
                },
                "required": ["session_id"]
            }
        ),
        types.Tool(
            name="health_check",
            description="Check the health status of the Smart Intent Router system",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        )
    ]

async def list_resources():
    """List all available resources for dynamic discovery."""
    return [
        types.Resource(
            uri="smart-intent-router://config",
            name="System Configuration",
            description="Current configuration settings for the Smart Intent Router",
            mimeType="application/json"
        ),
        types.Resource(
            uri="smart-intent-router://llm-services",
            name="Available LLM Services",
            description="List of configured LLM services and their capabilities",
            mimeType="application/json"
        ),
        types.Resource(
            uri="smart-intent-router://routing-rules",
            name="Routing Rules",
            description="Current rule-based routing configuration",
            mimeType="application/json"
        ),
        types.Resource(
            uri="smart-intent-router://system-status",
            name="System Status",
            description="Real-time system health and service availability",
            mimeType="application/json"
        )
    ]

# Pydantic models for FastAPI endpoints
class RouteRequest(BaseModel):
    message: str
    user_id: str = "default_user"
    session_id: Optional[str] = None
    conversation_id: Optional[str] = None
    intent: Optional[str] = None
    language: Optional[str] = None
    llm: Optional[str] = None

class RouteRequestAI(BaseModel):
    user_message: str
    user_id: str = "default_user"
    session_id: Optional[str] = None
    conversation_id: Optional[str] = None
    max_iterations: int = 5

class CreateConversation(BaseModel):
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    title: Optional[str] = None

class CreateSession(BaseModel):
    user_id: str

class DeleteConversation(BaseModel):
    conversation_id: str

class GetMessages(BaseModel):
    conversation_id: str

class GetConversationContext(BaseModel):
    conversation_id: str
    max_messages: int = 5

class CheckSession(BaseModel):
    session_id: str

# FastAPI Endpoints for Client-Facing APIs
@app.post("/route_request")
async def route_request_endpoint(request: RouteRequest):
    """Route a user request using rule-based routing."""
    try:
        result = await route_request_impl(
            message=request.message,
            user_id=request.user_id,
            session_id=request.session_id,
            conversation_id=request.conversation_id,
            intent=request.intent,
            language=request.language,
            llm=request.llm
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/route_request_ai")
async def route_request_ai_endpoint(request: RouteRequestAI):
    """Route a user request using AI-driven intelligent routing."""
    try:
        # Remove the @mcp.tool() decorator from route_request_ai and call it directly
        result = await route_request_ai_impl(
            user_message=request.user_message,
            user_id=request.user_id,
            session_id=request.session_id,
            conversation_id=request.conversation_id,
            max_iterations=request.max_iterations
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/route_request_ai_stream")
async def route_request_ai_stream_endpoint(
    user_message: str,
    user_id: str = "user",
    session_id: str = None,
    conversation_id: str = None,
    max_iterations: int = 5
):
    """Route a user request using AI-driven intelligent routing with real-time notifications via SSE."""
    from fastapi.responses import StreamingResponse
    import asyncio
    
    async def stream_ai_routing():
        """Generate Server-Sent Events for AI routing process"""
        try:
            # Create a queue to collect notifications
            notification_queue = asyncio.Queue()
            
            # Send initial event
            yield f"data: {json.dumps({'type': 'stream_started', 'message': 'Starting AI routing process...'})}\n\n"
            
            # Set up notification callback for real-time updates
            async def send_notification(notification_data):
                try:
                    print(f"🔔 DEBUG [SSE stream]: Queuing notification for client: {notification_data}")
                    await notification_queue.put(notification_data)
                except Exception as e:
                    print(f"❌ DEBUG [SSE stream]: Error queuing SSE notification: {e}")
            
            # Create task to process the AI request
            async def ai_task():
                try:
                    print(f"🚀 DEBUG [SSE stream]: Starting AI task for message: {user_message[:50]}{'...' if len(user_message) > 50 else ''}")
                    result = await route_request_ai_impl(
                        user_message=user_message,
                        user_id=user_id,
                        session_id=session_id,
                        conversation_id=conversation_id,
                        max_iterations=max_iterations,
                        notification_callback=send_notification
                    )
                    print(f"✅ DEBUG [SSE stream]: AI task completed, sending final result")
                    await notification_queue.put({'type': 'final_result', 'data': result})
                    await notification_queue.put(None)  # Signal completion
                except Exception as e:
                    print(f"❌ DEBUG [SSE stream]: Error in AI task: {e}")
                    await notification_queue.put({'type': 'error', 'message': str(e)})
                    await notification_queue.put(None)  # Signal completion
            
            # Start the AI task
            task = asyncio.create_task(ai_task())
            
            # Stream notifications as they come
            while True:
                try:
                    # Wait for a notification with timeout
                    notification = await asyncio.wait_for(notification_queue.get(), timeout=2.0)
                    if notification is None:  # Completion signal
                        print(f"🏁 DEBUG [SSE stream]: Received completion signal, ending stream")
                        break
                    print(f"📤 DEBUG [SSE stream]: Sending notification to client: {notification}")
                    sse_data = f"data: {json.dumps(notification)}\n\n"
                    print(f"📤 DEBUG [SSE stream]: SSE formatted data: {sse_data[:100]}{'...' if len(sse_data) > 100 else ''}")
                    yield sse_data
                except asyncio.TimeoutError:
                    # Send keep-alive
                    print(f"💓 DEBUG [SSE stream]: Sending keep-alive")
                    yield f"data: {json.dumps({'type': 'keep_alive'})}\n\n"
                    continue
            
            # Ensure task is done
            await task
            
        except Exception as e:
            print(f"Error in stream_ai_routing: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return StreamingResponse(
        stream_ai_routing(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "*"
        }
    )

@app.post("/conversations")
async def create_conversation_endpoint(request: CreateConversation):
    """Create a new conversation."""
    try:
        result = await create_conversation_impl(
            user_id=request.user_id,
            session_id=request.session_id,
            title=request.title
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/conversations/{conversation_id}")
async def delete_conversation_endpoint(conversation_id: str):
    """Delete a conversation."""
    try:
        result = await delete_conversation_impl(conversation_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/conversations/{user_id}")
async def get_conversations_endpoint(user_id: str):
    """Get all conversations for a user."""
    try:
        result = await get_conversations_impl(user_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/conversations/{conversation_id}/messages")
async def get_messages_endpoint(conversation_id: str):
    """Get messages for a specific conversation."""
    try:
        result = await get_messages_impl(conversation_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/conversations/{conversation_id}/context")
async def get_conversation_context_endpoint(conversation_id: str, request: GetConversationContext):
    """Get conversation context for intent classification."""
    try:
        result = await get_conversation_context(
            conversation_id=conversation_id,
            max_messages=request.max_messages
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/conversations/{conversation_id}/messages")
async def delete_messages_endpoint(conversation_id: str):
    """Clear all messages for a conversation."""
    try:
        result = await delete_messages_impl(conversation_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/sessions")
async def create_session_endpoint(request: CreateSession):
    """Create a new session."""
    try:
        result = await create_session_impl(request.user_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions/{session_id}")
async def check_session_endpoint(session_id: str):
    """Check if a session is active."""
    try:
        result = await check_session_impl(session_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check_endpoint():
    """Check the health status of the Smart Intent Router system."""
    try:
        result = await health_check_impl()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stream_response/{conversation_id}")
async def stream_response_endpoint(conversation_id: str, message: str):
    """Stream real-time responses from the LLM (example of server-to-client events)."""
    
    async def generate_streaming_response():
        # This is an example of how you could implement streaming responses
        # In a real implementation, you'd modify send_to_llm to yield chunks
        
        yield f"data: {{\"type\": \"status\", \"message\": \"Processing request...\"}}\n\n"
        
        # Simulate processing steps
        steps = [
            "Classifying intent...",
            "Detecting language...", 
            "Selecting optimal model...",
            "Sending to LLM...",
            "Generating response..."
        ]
        
        for step in steps:
            yield f"data: {{\"type\": \"progress\", \"message\": \"{step}\"}}\n\n"
            await asyncio.sleep(0.5)  # Simulate processing time
        
        # In a real implementation, you'd stream the actual LLM response chunks here
        response_chunks = [
            "This is a streaming response ",
            "that demonstrates real-time ",
            "server-to-client communication ",
            "using Server-Sent Events (SSE)."
        ]
        
        for chunk in response_chunks:
            yield f"data: {{\"type\": \"response_chunk\", \"content\": \"{chunk}\"}}\n\n"
            await asyncio.sleep(0.3)
        
        yield f"data: {{\"type\": \"complete\", \"message\": \"Response completed\"}}\n\n"
    
    return StreamingResponse(
        generate_streaming_response(), 
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream"
        }
    )

# WebSocket endpoint for AI routing with real-time notifications
@app.websocket("/ws/ai_route/{conversation_id}")
async def ai_route_websocket_endpoint(websocket: WebSocket, conversation_id: str):
    """WebSocket endpoint for AI routing with real-time step-by-step notifications."""
    await websocket.accept()
    
    async def send_notification(notification_data):
        """Send notification to WebSocket client"""
        try:
            await websocket.send_text(json.dumps({
                "type": "notification",
                "data": notification_data,
                "timestamp": json.dumps(asyncio.get_event_loop().time())
            }))
        except Exception as e:
            print(f"Failed to send WebSocket notification: {e}")
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            if message_data.get("type") == "user_message":
                user_message = message_data.get("message")
                user_id = message_data.get("user_id", "default_user")
                session_id = message_data.get("session_id")
                max_iterations = message_data.get("max_iterations", 5)
                
                # Send initial acknowledgment
                await websocket.send_text(json.dumps({
                    "type": "status", 
                    "message": "Starting AI routing process..."
                }))
                
                # Process the request with notifications
                try:
                    result = await route_request_ai_impl(
                        user_message=user_message,
                        user_id=user_id,
                        session_id=session_id,
                        conversation_id=conversation_id,
                        max_iterations=max_iterations,
                        notification_callback=send_notification
                    )
                    
                    # Send the final response
                    await websocket.send_text(json.dumps({
                        "type": "final_response",
                        "data": result
                    }))
                    
                except Exception as e:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": str(e)
                    }))
                    
    except WebSocketDisconnect:
        print(f"AI routing WebSocket disconnected for conversation {conversation_id}")

# WebSocket endpoint for real-time bidirectional communication
@app.websocket("/ws/{conversation_id}")
async def websocket_endpoint(websocket: WebSocket, conversation_id: str):
    """WebSocket endpoint for real-time bidirectional communication."""
    await websocket.accept()
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            if message_data.get("type") == "user_message":
                user_message = message_data.get("message")
                user_id = message_data.get("user_id", "default_user")
                
                # Send acknowledgment
                await websocket.send_text(json.dumps({
                    "type": "status", 
                    "message": "Processing your request..."
                }))
                
                # Process the request (this would be your actual routing logic)
                try:
                    result = await route_request_impl(
                        message=user_message,
                        user_id=user_id,
                        conversation_id=conversation_id
                    )
                    
                    # Send the response back
                    await websocket.send_text(json.dumps({
                        "type": "response",
                        "data": result
                    }))
                    
                except Exception as e:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": str(e)
                    }))
                    
    except WebSocketDisconnect:
        print(f"WebSocket disconnected for conversation {conversation_id}")

def main():
    import uvicorn
    import logging
    
    logging.basicConfig(level=logging.INFO)
    
    # Since the web client uses REST API endpoints directly, 
    # we only need to run the FastAPI server, not the MCP server
    print("Starting FastAPI server on port 8000...")
    print("MCP server disabled - using FastAPI endpoints only")
    
    # Run FastAPI server directly (no threading needed)
    uvicorn.run(app, host="0.0.0.0", port=8000)

def parse_ai_response(response_text: str) -> dict:
    """
    Parse AI response to extract function calls or direct responses.
    Supports both OpenAI function calling format and custom TOOL_CALL format.
    Returns a dict with type and parsed data.
    """
    response_text = response_text.strip()
    
    # Check for intent classification with function call format: "Intent: [intent] | function_call(...)"
    if "Intent:" in response_text and "|" in response_text:
        import re
        parts = response_text.split("|", 1)
        if len(parts) == 2:
            intent_part = parts[0].strip()
            function_part = parts[1].strip()
            
            # Extract the intent
            intent_match = re.search(r'Intent:\s*(\w+)', intent_part)
            if intent_match:
                intent = intent_match.group(1)
                
                # Parse the function call part
                function_result = parse_single_function_call(function_part)
                if function_result.get("type") == "function_call":
                    # Add the intent information to the result
                    function_result["intent_classified"] = intent
                    return function_result
    
    # Check for intent in comments: "function_call(...)  # Intent: code"
    import re
    if "# Intent:" in response_text:
        intent_match = re.search(r'#\s*Intent:\s*(\w+)', response_text, re.IGNORECASE)
        if intent_match:
            intent = intent_match.group(1)
            print(f"🎯 DEBUG: Extracted intent from comment: {intent}")
            
            # Remove the comment and parse the function call
            clean_text = re.sub(r'\s*#.*$', '', response_text, flags=re.MULTILINE)
            function_result = parse_single_function_call(clean_text)
            if function_result.get("type") == "function_call":
                function_result["intent_classified"] = intent
                return function_result
    
    # Check for multiple function calls in sequence (orchestrator issue)
    lines = response_text.split('\n')
    function_calls = []
    
    for line in lines:
        line = line.strip()
        if ('(' in line and ')' in line and 
            any(func in line for func in ['classify_intent', 'detect_language', 'get_models', 'send_to_llm'])):
            function_calls.append(line)
    
    # If we found multiple function calls, return only the FIRST one (enforce step-by-step)
    if len(function_calls) > 1:
        print(f"WARNING: Orchestrator returned {len(function_calls)} function calls, but should return only ONE at a time.")
        print(f"Function calls found: {function_calls}")
        print("Executing ONLY the first one to enforce step-by-step workflow.")
        first_call = function_calls[0]
        return parse_single_function_call(first_call)
    elif len(function_calls) == 1:
        return parse_single_function_call(function_calls[0])
    
    # PRIORITY 1: Check for resource calls first (they should be handled before tool calls)
    if "RESOURCE_CALL:" in response_text:
        # Extract resource name
        for line in response_text.split('\n'):
            line = line.strip()
            if line.startswith("RESOURCE_CALL:"):
                resource_name = line.replace("RESOURCE_CALL:", "").strip()
                return {
                    "type": "resource_call",
                    "resource_name": resource_name
                }
    
    # PRIORITY 2: Check for tool calls (legacy format)
    elif "TOOL_CALL:" in response_text:
        lines = response_text.split('\n')
        tool_name = None
        parameters = {}
        
        for line in lines:
            line = line.strip()
            if line.startswith("TOOL_CALL:"):
                tool_name = line.replace("TOOL_CALL:", "").strip()
            elif line.startswith("PARAMETERS:"):
                try:
                    param_text = line.replace("PARAMETERS:", "").strip()
                    parameters = json.loads(param_text)
                except json.JSONDecodeError:
                    # If JSON parsing fails, try to extract from the text
                    parameters = {}
        
        if tool_name:
            return {
                "type": "tool_call",
                "tool_name": tool_name,
                "parameters": parameters
            }
    
    # PRIORITY 3: Check for capability discovery requests
    elif "LIST_TOOLS" in response_text:
        return {"type": "list_tools"}
    elif "LIST_RESOURCES" in response_text:
        return {"type": "list_resources"}
    
    # PRIORITY 4: Check if this is a direct answer that should be corrected (for function-calling models)
    response_lower = response_text.lower()
    is_direct_answer = any([
        "```" in response_text,  # Code blocks
        "def " in response_lower,  # Function definitions  
        "print(" in response_lower,  # Print statements
        "certainly" in response_lower,  # Polite responses
        "here is" in response_lower,  # Direct explanations
        "below is" in response_lower,  # Direct explanations
        "this script" in response_lower,  # Code explanations
        "function that" in response_lower,  # Function descriptions
        len(response_text) > 100  # Long responses are likely direct answers
    ])
    
    if is_direct_answer:
        return {
            "type": "direct_answer_needs_correction",
            "content": response_text
        }
    
    # PRIORITY 5: Otherwise, treat as final response
    return {
        "type": "final_response",
        "content": response_text
    }

def parse_single_function_call(function_call_text: str) -> dict:
    """
    Parse a single function call text into structured format.
    
    Examples:
    - classify_intent(message="write python code")
    - detect_language(prompt="hello world")
    - get_models()
    - send_to_llm(model_name="gpt-3", user_message="hello")
    - JSON format: {"type": "function", "function": {"name": "...", "arguments": {...}}}
    """
    import re
    import json
    
    function_call_text = function_call_text.strip()
    
    # Check if this is a JSON response (some models return JSON instead of direct function calls)
    if function_call_text.startswith('{') and function_call_text.endswith('}'):
        try:
            json_response = json.loads(function_call_text)
            if (json_response.get("type") == "function" and 
                "function" in json_response and 
                "name" in json_response["function"]):
                
                function_name = json_response["function"]["name"]
                arguments = json_response["function"].get("arguments", {})
                
                # Map unrecognized function names to our actual functions
                if function_name == "reverse_string":
                    # This was likely meant to be a direct answer, redirect to classify_intent
                    return {
                        "type": "function_call",
                        "function_name": "classify_intent",
                        "parameters": {"message": "write python code to reverse a string"}
                    }
                
                # If it's one of our valid functions, use it
                if function_name in ["classify_intent", "detect_language", "get_models", "send_to_llm"]:
                    return {
                        "type": "function_call",
                        "function_name": function_name,
                        "parameters": arguments
                    }
        except json.JSONDecodeError:
            pass
    
    # Extract function name and parameters using regex
    pattern = r'(\w+)\(([^)]*)\)'
    match = re.match(pattern, function_call_text.strip())
    
    if not match:
        return {
            "type": "final_response",
            "content": function_call_text
        }
    
    function_name = match.group(1)
    params_str = match.group(2)
    
    # Parse parameters
    parameters = {}
    
    if params_str.strip():
        # Split by comma, handling quoted strings
        import re
        
        # Find all parameter assignments: key="value" or key=value
        param_pattern = r'(\w+)=(["\'])((?:(?!\2)[^\\]|\\.)*)(\2)'
        matches = re.findall(param_pattern, params_str)
        
        for match in matches:
            key = match[0]
            value = match[2]  # The content between quotes
            parameters[key] = value
        
        # Also handle case where there are no key= assignments
        if not parameters and '"' in params_str:
            # Try to extract positional arguments for specific functions
            quoted_values = re.findall(r'"([^"]*)"', params_str)
            if function_name == "classify_intent" and len(quoted_values) >= 1:
                parameters["message"] = quoted_values[0]
            elif function_name == "detect_language" and len(quoted_values) >= 1:
                parameters["prompt"] = quoted_values[0]
            elif function_name == "send_to_llm" and len(quoted_values) >= 2:
                parameters["model_name"] = quoted_values[0]
                parameters["user_message"] = quoted_values[1]
    
    return {
        "type": "function_call",
        "function_name": function_name,
        "parameters": parameters
    }

async def get_validated_model_info(model_name: str) -> dict:
    """
    Validate that a model exists and get its information.
    Uses the generic get_model method for consistency.
    """
    model_info = await get_model(model_name)
    
    if not model_info:
        return None
    
    return {
        "model_name": model_info["model_name"],
        "endpoint": model_info["endpoint"],
        "context_length": 4096,  # Default context length
        "source": model_info["source"],
        "available": True
    }

async def get_model(name: str, fallback_to_first: bool = False) -> dict:
    """
    Find and return a specific model by name or name pattern.
    
    Args:
        name: Model name or pattern to search for (case-insensitive)
        fallback_to_first: If True, return first available model if specified model not found
    
    Search priority:
    1. LM Studio real-time models (exact match)
    2. LM Studio real-time models (partial match)
    3. Configuration models (exact match)
    4. Configuration models (partial match)
    5. Fallback to first available model (if fallback_to_first=True)
    
    Returns:
        dict: Model info with model_name, endpoint, and source, or None if not found
    """
    if not name:
        logging.error("Model name cannot be empty")
        return None
    
    name_lower = name.lower()
    
    # First, try to find model in LM Studio models
    lm_studio_models = await get_models_from_lm_studio()
    if lm_studio_models:
        # Exact match first
        for model in lm_studio_models:
            model_name = model.get("model_name", "")
            if model_name.lower() == name_lower:
                result = {
                    "model_name": model.get("model_name"),
                    "endpoint": model.get("endpoint"),
                    "source": "lm_studio_real_time"
                }
                logging.info(f"Found exact match for '{name}' in LM Studio: {model.get('model_name')}")
                return result
        
        # Partial match if no exact match
        for model in lm_studio_models:
            model_name = model.get("model_name", "").lower()
            if name_lower in model_name:
                result = {
                    "model_name": model.get("model_name"),
                    "endpoint": model.get("endpoint"),
                    "source": "lm_studio_real_time"
                }
                logging.info(f"Found partial match for '{name}' in LM Studio: {model.get('model_name')}")
                return result
    
    # If not found in LM Studio, check configuration
    config_models = await get_models_from_config()
    if config_models:
        # Exact match first
        for model in config_models:
            model_name = model.get("model_name", "")
            if model_name.lower() == name_lower:
                result = {
                    "model_name": model.get("model_name"),
                    "endpoint": model.get("endpoint"),
                    "source": "configuration"
                }
                logging.info(f"Found exact match for '{name}' in config: {model.get('model_name')}")
                return result
        
        # Partial match if no exact match
        for model in config_models:
            model_name = model.get("model_name", "").lower()
            if name_lower in model_name:
                result = {
                    "model_name": model.get("model_name"),
                    "endpoint": model.get("endpoint"),
                    "source": "configuration"
                }
                logging.info(f"Found partial match for '{name}' in config: {model.get('model_name')}")
                return result
    
    # Fallback: use first available model if requested
    if fallback_to_first:
        if lm_studio_models:
            first_model = lm_studio_models[0]
            result = {
                "model_name": first_model.get("model_name"),
                "endpoint": first_model.get("endpoint"),
                "source": "lm_studio_fallback"
            }
            logging.warning(f"Model '{name}' not found, using fallback from LM Studio: {first_model.get('model_name')}")
            return result
        elif config_models:
            result = {
                "model_name": config_models[0].get("model_name"),
                "endpoint": config_models[0].get("endpoint"),
                "source": "config_fallback"
            }
            logging.warning(f"Model '{name}' not found, using fallback from config: {config_models[0].get('model_name')}")
            return result
    
    # Model not found
    logging.error(f"Model '{name}' not found in LM Studio or configuration")
    return None

async def get_orchestrator_model() -> dict:
    """
    Find and return the orchestrator model based on the is_orchestrator flag in configuration.
    
    Priority:
    1. First model marked with is_orchestrator: true in config
    2. Fallback to first enabled model if none marked as orchestrator
    
    Returns:
        dict: Model info with model_name, endpoint, and source, or None if not found
    """
    config_models = await get_models_from_config()
    
    if not config_models:
        logging.error("No models found in configuration")
        return None
    
    # Look for a model marked as orchestrator
    for model in config_models:
        if model.get("is_orchestrator") is True and model.get("enabled", True):
            result = {
                "model_name": model.get("model_name"),
                "endpoint": model.get("endpoint"),
                "source": "configuration_orchestrator",
                "name": model.get("name")  # Include the friendly name for logging
            }
            logging.info(f"Found orchestrator model: {model.get('name')} ({model.get('model_name')})")
            return result
    
    # Fallback: use first enabled model
    for model in config_models:
        if model.get("enabled", True):
            result = {
                "model_name": model.get("model_name"),
                "endpoint": model.get("endpoint"),
                "source": "configuration_fallback",
                "name": model.get("name")
            }
            logging.warning(f"No orchestrator model found, using fallback: {model.get('name')} ({model.get('model_name')})")
            return result
    
    logging.error("No enabled models found in configuration")
    return None

async def health_check_impl() -> dict:
    """Check the health status of the Smart Intent Router system and LM Studio connectivity."""
    
    lm_studio_models = await get_models_from_lm_studio()
    config_models = await get_models_from_config()
    
    base_endpoint = get_lm_studio_base_endpoint_from_config(config)
    
    health_status = {
        "system_status": "healthy",
        "lm_studio_connectivity": len(lm_studio_models) > 0,
        "lm_studio_endpoint": base_endpoint,
        "models_available_in_lm_studio": len(lm_studio_models),
        "models_configured": len(config_models),
        "real_time_model_discovery": len(lm_studio_models) > 0,
        "fallback_to_config": len(lm_studio_models) == 0 and len(config_models) > 0
    }
    
    if lm_studio_models:
        health_status["available_models"] = [
            {
                "name": model.get("model_name"),
                "source": "lm_studio_real_time"
            }
            for model in lm_studio_models
        ]
    elif config_models:
        health_status["available_models"] = [
            {
                "name": model.get("model_name"),
                "source": "configuration_fallback"
            }
            for model in config_models
        ]
        health_status["warning"] = "LM Studio not accessible, using configuration fallback"
    else:
        health_status["system_status"] = "degraded"
        health_status["error"] = "No models available from LM Studio or configuration"
    
    return health_status

@mcp.tool()
async def get_models_from_config_tool() -> dict:
    """Get available models from configuration file."""
    models = await get_models_from_config()
    return {
        "models": models,
        "count": len(models),
        "source": "configuration"
    }

@mcp.tool()
async def get_models_from_lm_studio_tool() -> dict:
    """Get available models directly from LM Studio real-time API."""
    models = await get_models_from_lm_studio()
    return {
        "models": models,
        "count": len(models),
        "source": "lm_studio_real_time",
        "available": len(models) > 0
    }

@mcp.tool()
async def get_model_tool(model_name: str, fallback_to_first: bool = False) -> dict:
    """Get a specific model by name or pattern."""
    model = await get_model(model_name, fallback_to_first)
    if not model:
        return {"error": f"Model '{model_name}' not found"}
    return {
        "model": model,
        "success": True
    }

@mcp.tool()
async def select_best_model(intent: str, language: str = "en") -> dict:
    """
    Intelligently select the best model based on intent and language.
    Returns the most suitable model name for the given requirements.
    """
    # Get all available models (prefer LM Studio real-time data)
    lm_studio_models = await get_models_from_lm_studio()
    if lm_studio_models:
        available_models = lm_studio_models
    else:
        available_models = await get_models_from_config()
    
    if not available_models:
        return {"error": "No models available"}
    
    # Model selection logic
    best_model = None
    best_score = 0
    
    for model in available_models:
        score = 0
        model_name = model.get("model_name", "").lower()
        
        # 1. Check supported_intents (highest priority)
        supported_intents = model.get("supported_intents", [])
        if intent in supported_intents:
            score += 100
        
        # 2. Check supported_languages
        supported_languages = model.get("supported_languages", [])
        if language in supported_languages:
            score += 50
        
        # 3. Model name specialization heuristics
        if intent == "code" and ("coder" in model_name or "code" in model_name):
            score += 80
        elif intent == "math" and ("math" in model_name or "wizard" in model_name):
            score += 80
        elif intent == "translation" and ("translate" in model_name or "multilingual" in model_name):
            score += 80
        
        # 4. Model weight as tiebreaker
        weight = model.get("weight", 1.0)
        score += weight * 10
        
        # 5. Fallback scoring for general models
        if intent == "general" and "llama" in model_name:
            score += 30
        
        if score > best_score:
            best_score = score
            best_model = model
    
    if not best_model:
        # Fallback to first available model
        best_model = available_models[0]
    
    return {
        "selected_model": best_model.get("model_name"),
        "selection_score": best_score,
        "model_info": best_model,
        "selection_reason": f"Selected based on intent '{intent}', language '{language}', and model capabilities",
        "success": True
    }

@mcp.tool()
async def get_orchestrator_model_tool() -> dict:
    """Get the orchestrator model based on the is_orchestrator flag in configuration."""
    orchestrator_model = await get_orchestrator_model()
    if not orchestrator_model:
        return {"error": "No orchestrator model available"}
    return {
        "orchestrator_model": orchestrator_model,
        "success": True
    }

@mcp.tool()
async def get_conversation_context(conversation_id: str, max_messages: int = 5) -> dict:
    """Get recent conversation context to help with intent classification and response generation."""
    if not conversation_id:
        return {"context": "", "message_count": 0}
    
    try:
        # Get recent messages for context
        messages = conversation_manager.get_messages(conversation_id)
        recent_messages = messages.get("conversation", [])[-max_messages:] if messages else []
        
        # Format context as a readable string
        context_parts = []
        for msg in recent_messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if isinstance(content, dict):
                content = content.get("response", str(content))
            context_parts.append(f"{role}: {content}")
        
        context_string = " | ".join(context_parts)
        
        return {
            "context": context_string,
            "message_count": len(recent_messages),
            "success": True
        }
    except Exception as e:
        return {"error": f"Failed to get conversation context: {str(e)}"}

if __name__ == "__main__":
    main()
