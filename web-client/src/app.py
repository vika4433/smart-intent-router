import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import streamlit as st
import re
import os
import requests
import json
import time
import threading
from typing import Optional
from dotenv import load_dotenv

# Apply nest_asyncio to patch the event loop (not needed for HTTP client)

# Load environment variables
load_dotenv()
SERVER_URL = os.getenv("SERVER_URL", "http://localhost:8000")  # Updated to FastAPI port

# --- Helper function to check server status ---
def check_server_status():
    """Check if the Smart Intent Router server is reachable"""
    try:
        # Try the FastAPI health endpoint
        response = requests.get(f"{SERVER_URL}/health", timeout=5)
        return response.status_code == 200
    except Exception:
        return False

# --- REST API Client for FastAPI Server ---
class RestAPIClient:
    def __init__(self):
        self.base_url = SERVER_URL
        self.session = requests.Session()
        self._connection_failed = False
        self._error_message = None
    
    def check_connection(self):
        """Check if the server is available with retry logic"""
        if self._connection_failed:
            return False
        
        # Try multiple times with longer delays for startup scenarios
        max_attempts = 10
        for attempt in range(max_attempts):
            try:
                response = requests.get(f"{self.base_url}/health", timeout=10)
                if response.status_code == 200:
                    return True
            except Exception:
                if attempt < max_attempts - 1:  # Don't sleep on the last attempt
                    time.sleep(2)  # Wait 2 seconds before retry
                continue
        
        # If all attempts failed, mark connection as failed
        self._connection_failed = True
        self._error_message = f"Failed to connect to Smart Intent Router server at {self.base_url}"
        return False
    
    def show_connection_error_if_needed(self):
        """Show connection error with troubleshooting info - only call after st.set_page_config()"""
        if self._connection_failed and self._error_message:
            st.error(f"❌ {self._error_message}")
            
            with st.expander("🔧 Troubleshooting"):
                st.markdown("""
                **To fix this issue:**
                
                1. **Start the server** in a terminal:
                   ```bash
                   cd smart-intent-router-server
                   python src/mcp_server/server.py
                   ```
                
                2. **Check the server is running** by visiting: http://localhost:8000/health
                
                3. **Refresh this page** once the server is running
                """)
            
            if st.button("🔄 Retry Connection"):
                self._connection_failed = False
                self._error_message = None
                st.rerun()
            
            st.stop()
    
    def create_session(self, user_id: str) -> dict:
        """Create a new session"""
        if not self.check_connection():
            return {"error": "Cannot connect to server"}
        
        try:
            response = self.session.post(
                f"{self.base_url}/sessions",
                json={"user_id": user_id},
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def create_conversation(self, user_id: str, session_id: str = None, title: str = None) -> dict:
        """Create a new conversation"""
        if not self.check_connection():
            return {"error": "Cannot connect to server"}
        
        try:
            payload = {"user_id": user_id}
            if session_id:
                payload["session_id"] = session_id
            if title:
                payload["title"] = title
                
            response = self.session.post(
                f"{self.base_url}/conversations",
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def get_conversations(self, user_id: str) -> dict:
        """Get all conversations for a user"""
        if not self.check_connection():
            return {"error": "Cannot connect to server"}
        
        try:
            response = self.session.get(
                f"{self.base_url}/conversations/{user_id}",
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def get_messages(self, conversation_id: str) -> dict:
        """Get messages for a conversation"""
        if not self.check_connection():
            return {"error": "Cannot connect to server"}
        
        try:
            response = self.session.get(
                f"{self.base_url}/conversations/{conversation_id}/messages",
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def delete_messages(self, conversation_id: str) -> dict:
        """Delete all messages in a conversation"""
        if not self.check_connection():
            return {"error": "Cannot connect to server"}
        
        try:
            response = self.session.delete(
                f"{self.base_url}/conversations/{conversation_id}/messages",
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def route_request(self, message: str, user_id: str, session_id: str = None, conversation_id: str = None, intent: str = None, language: str = None, llm: str = None) -> dict:
        """Route a request using rule-based routing"""
        if not self.check_connection():
            return {"error": "Cannot connect to server"}
        
        try:
            payload = {
                "message": message,
                "user_id": user_id
            }
            if session_id:
                payload["session_id"] = session_id
            if conversation_id:
                payload["conversation_id"] = conversation_id
            if intent:
                payload["intent"] = intent
            if language:
                payload["language"] = language
            if llm:
                payload["llm"] = llm
                
            response = self.session.post(
                f"{self.base_url}/route_request",
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def route_request_ai(self, user_message: str, user_id: str, session_id: str = None, conversation_id: str = None, max_iterations: int = 5) -> dict:
        """Route a request using AI-driven routing"""
        if not self.check_connection():
            return {"error": "Cannot connect to server"}
        
        try:
            payload = {
                "user_message": user_message,
                "user_id": user_id,
                "max_iterations": max_iterations
            }
            if session_id:
                payload["session_id"] = session_id
            if conversation_id:
                payload["conversation_id"] = conversation_id
                
            response = self.session.post(
                f"{self.base_url}/route_request_ai",
                json=payload,
                timeout=240  # Increased timeout for AI processing with LM Studio
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def route_request_ai_stream(self, user_message: str, user_id: str, session_id: str = None, conversation_id: str = None, max_iterations: int = 5):
        """Route a request using AI-driven routing with real-time notifications via SSE"""
        if not self.check_connection():
            yield {"error": "Cannot connect to server"}
            return
        
        try:
            # Build query parameters
            params = {
                "user_message": user_message,
                "user_id": user_id,
                "max_iterations": max_iterations
            }
            if session_id:
                params["session_id"] = session_id
            if conversation_id:
                params["conversation_id"] = conversation_id
                
            response = self.session.get(
                f"{self.base_url}/route_request_ai_stream",
                params=params,
                timeout=240,  # Longer timeout for streaming
                stream=True
            )
            response.raise_for_status()
            
            # Parse Server-Sent Events
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    print(f"🌐 DEBUG [route_request_ai_stream]: Raw SSE line received: {line[:100]}{'...' if len(line) > 100 else ''}")
                    if line.startswith('data: '):
                        try:
                            data_content = line[6:]  # Remove 'data: ' prefix
                            print(f"🌐 DEBUG [route_request_ai_stream]: Parsing JSON data: {data_content[:100]}{'...' if len(data_content) > 100 else ''}")
                            data = json.loads(data_content)
                            print(f"🌐 DEBUG [route_request_ai_stream]: Successfully parsed notification: {data.get('type', 'unknown_type')}")
                            yield data
                        except json.JSONDecodeError as e:
                            print(f"🌐 DEBUG [route_request_ai_stream]: JSON decode error: {e}")
                            continue
        except Exception as e:
            yield {"error": str(e)}

# Initialize global REST API client (lazy initialization)
_api_client = None

def get_api_client():
    global _api_client
    if _api_client is None:
        _api_client = RestAPIClient()
    return _api_client

# --- Get API Client ---
def get_client():
    """Get the REST API client instance"""
    return get_api_client()

def clean_response(response: str) -> str:
    response = re.sub(r'```(\w*)', r'\n```\1\n', response)
    response = response.replace('```', '\n```\n')
    response = re.sub(r'pip install py pdf2', 'pip install pypdf2', response)
    return response.strip()

def display_markdown_with_code(response: str):
    st.markdown("""
    <style>
        .stMarkdown { line-height: 1.7; font-size: 16px; color: #333; }
        div[data-testid="stCodeBlock"] {
            margin: 1.5em 0;
            border-radius: 8px;
            border-left: 4px solid #4e8cff;
            background: #f8f9fa;
        }
        pre {
            white-space: pre-wrap;
            font-family: 'SFMono-Regular', Menlo, monospace;
            padding: 1em !important;
            margin: 0;
        }
        code:not(pre code) {
            background: #f3f4f6;
            padding: 0.2em 0.4em;
            border-radius: 3px;
            font-size: 0.9em;
            font-family: 'SFMono-Regular', Menlo, monospace;
        }
        .stMarkdown p {
            margin-bottom: 1em;
        }
    </style>
    """, unsafe_allow_html=True)
    segments = re.split(r'(```[\s\S]*?```)', response)
    for segment in segments:
        if segment.startswith('```'):
            language_match = re.match(r'```(\w+)', segment)
            language = language_match.group(1) if language_match else ''
            code_content = re.sub(r'```(\w+)?\n?', '', segment).replace('```', '')
            if language:
                st.markdown(
                    f'<div style="font-family: monospace; color: #666; font-size: 0.9em;">{language}</div>',
                    unsafe_allow_html=True)
            st.code(code_content, language=language.lower() if language else None)
        else:
            if segment.strip():
                st.markdown(segment, unsafe_allow_html=True)

def extract_session_id(response):
    # If response is a list and first element has .text, use it
    if isinstance(response, list) and hasattr(response[0], 'text'):
        try:
            data = json.loads(response[0].text)
            return data.get("session_id")
        except Exception:
            return None
    # If response is a dict
    if isinstance(response, dict):
        return response.get("session_id")
    # If response is a string
    if isinstance(response, str):
        try:
            data = json.loads(response)
            return data.get("session_id")
        except Exception:
            return None
    return None

def display_chat_message(role: str, content):
    avatar = "🧑" if role == "user" else "🤖"
    bubble_color = "#e6f0ff" if role == "user" else "#f3f3f3"
    align = "flex-end" if role == "user" else "flex-start"
    # Extract LLM response if content is a dict with 'response'
    if not isinstance(content, str):
        if isinstance(content, dict) and "response" in content:
            content = content["response"]
        else:
            try:
                content = json.dumps(content, indent=2, ensure_ascii=False)
            except Exception:
                content = str(content)
    # Only show content in the bubble for user messages
    if role == "user":
        bubble_html = f"""
        <div style='display: flex; justify-content: {align}; margin-bottom: 0.5em;'>
            <div style='background: {bubble_color}; border-radius: 12px; padding: 0.8em 1.2em; max-width: 80%; box-shadow: 0 1px 4px #0001;'>
                <span style='font-size: 1.2em; margin-right: 0.5em;'>{avatar}</span>
                <span style='vertical-align: middle;'>{content}</span>
            </div>
        </div>
        """
    else:
        bubble_html = f"""
        <div style='display: flex; justify-content: {align}; margin-bottom: 0.5em;'>
            <div style='background: {bubble_color}; border-radius: 12px; padding: 0.8em 1.2em; max-width: 80%; box-shadow: 0 1px 4px #0001;'>
                <span style='font-size: 1.2em; margin-right: 0.5em;'>{avatar}</span>
            </div>
        </div>
        """
    st.markdown(bubble_html, unsafe_allow_html=True)
    if role == "assistant":
        display_markdown_with_code(content)

def initialize_session_and_conversation():
    client = get_client()
    
    # Step 1: Create session (only if not already in session_state)
    response = client.create_session("user")
    session_id = extract_session_id(response)
    st.session_state.session_id = session_id

    # Step 2: Get conversations for this user
    conversation_id = None
    response = client.get_conversations("user")
    try:
        conversations = response["conversations"] if isinstance(response, dict) else eval(response)["conversations"]
        if conversations and isinstance(conversations, list) and len(conversations) > 0:
            conversation_id = conversations[0]["conversation_id"] if isinstance(conversations[0], dict) and "conversation_id" in conversations[0] else conversations[0]
    except Exception:
        conversation_id = None

    # Step 3: If no conversation, create one
    if not conversation_id and session_id:
        response = client.create_conversation("user", session_id, "New Conversation")
        try:
            conversation_id = response["conversation_id"] if isinstance(response, dict) else eval(response)["conversation_id"]
        except Exception:
            conversation_id = None
    st.session_state.conversation_id = conversation_id

    # Step 4: Fetch and display messages for the latest conversation
    st.session_state.messages = []
    if conversation_id:
        response = client.get_messages(conversation_id)
        try:
            # response is expected to be a dict with key 'conversation' containing a list of messages
            if isinstance(response, dict):
                messages = response.get("conversation", [])
            elif isinstance(response, str):
                try:
                    response_dict = json.loads(response)
                    messages = response_dict.get("conversation", [])
                except Exception:
                    messages = []
            else:
                messages = []
            if isinstance(messages, list):
                for msg in messages:
                    if "role" in msg and "content" in msg:
                        # Store the full message dict for assistant messages, not just role/content
                        st.session_state.messages.append(msg)
        except Exception:
            pass

def display_real_time_reasoning(user_message: str):
    """Display the AI reasoning process in real-time with step-by-step notifications"""
    
    # Create containers for real-time updates
    reasoning_container = st.container()
    with reasoning_container:
        st.markdown("### 🧠 AI Reasoning Process")
        
        # Create containers that can be updated in real-time
        progress_container = st.container()
        workflow_container = st.container()
        live_container = st.container()
        
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()
        
        with workflow_container:
            workflow_display = st.empty()
        
        with live_container:
            live_updates = st.empty()
    
    # Initialize workflow tracking
    workflow_steps = {
        "intent": None,
        "language": None,
        "models_count": None,
        "model_used": None
    }
    
    step_count = 0
    max_steps = 5
    final_ai_response = None
    live_updates_list = []
    
    def update_displays():
        """Update all displays with current state"""
        # Update progress
        progress = min(step_count / max_steps, 1.0)
        progress_bar.progress(progress)
        
        # Update workflow display
        workflow_text = "🤖 **Real-time AI Workflow:**\n\n"
        
        if workflow_steps["intent"]:
            workflow_text += f"✅ **Step 1:** classify_intent → **{workflow_steps['intent']}**\n\n"
        elif step_count >= 1:
            workflow_text += f"🔄 **Step 1:** classify_intent → Analyzing...\n\n"
        else:
            workflow_text += f"⏳ **Step 1:** classify_intent\n\n"
        
        if workflow_steps["language"]:
            workflow_text += f"✅ **Step 2:** detect_language → **{workflow_steps['language']}**\n\n"
        elif step_count >= 2:
            workflow_text += f"🔄 **Step 2:** detect_language → Detecting...\n\n"
        elif workflow_steps["intent"]:
            workflow_text += f"⏳ **Step 2:** detect_language\n\n"
        
        if workflow_steps["models_count"]:
            workflow_text += f"✅ **Step 3:** get_models → **{workflow_steps['models_count']} available**\n\n"
        elif step_count >= 3:
            workflow_text += f"🔄 **Step 3:** get_models → Loading...\n\n"
        elif workflow_steps["language"]:
            workflow_text += f"⏳ **Step 3:** get_models\n\n"
        
        if workflow_steps["model_used"]:
            workflow_text += f"✅ **Step 4:** send_to_llm → **{workflow_steps['model_used']}**\n\n"
        elif step_count >= 4:
            workflow_text += f"🔄 **Step 4:** send_to_llm → Processing...\n\n"
        elif workflow_steps["models_count"]:
            workflow_text += f"⏳ **Step 4:** send_to_llm\n\n"
        
        if final_ai_response:
            workflow_text += f"🎉 **Complete:** Response ready!"
        
        workflow_display.markdown(workflow_text)
        
        # Update live feed
        if live_updates_list:
            updates_text = "📡 **Live Updates:**\n\n" + "\n".join([f"• {update}" for update in live_updates_list[-5:]])
            live_updates.markdown(updates_text)
    
    # Initialize displays
    status_text.info("🚀 Starting AI reasoning process...")
    update_displays()
    
    # Stream the AI routing process
    client = get_client()
    
    try:
        for notification in client.route_request_ai_stream(
            user_message, 
            "user", 
            st.session_state.session_id, 
            st.session_state.conversation_id
        ):
            if "error" in notification:
                status_text.error(f"❌ Error: {notification['error']}")
                return None
            
            notification_type = notification.get("type", "")
            print(f"📱 DEBUG [app.py]: Processing notification: {notification_type}")
            
            # Add to live updates
            timestamp = time.strftime("%H:%M:%S")
            
            # Process different notification types and update immediately
            if notification_type == "intent_classified":
                intent = notification.get("intent", "unknown")
                workflow_steps["intent"] = intent
                step_count = max(step_count, 1)
                status_text.info(f"🎯 Intent classified: {intent}")
                live_updates_list.append(f"{timestamp} - 🎯 Intent: {intent}")
                update_displays()
                time.sleep(0.1)  # Small delay for UI update
                
            elif notification_type == "language_detected":
                language = notification.get("language", "unknown")
                workflow_steps["language"] = language
                step_count = max(step_count, 2)
                status_text.info(f"🌍 Language detected: {language}")
                live_updates_list.append(f"{timestamp} - 🌍 Language: {language}")
                update_displays()
                time.sleep(0.1)  # Small delay for UI update
                
            elif notification_type == "models_retrieved":
                count = notification.get("count", 0)
                workflow_steps["models_count"] = count
                step_count = max(step_count, 3)
                status_text.info(f"🤖 Found {count} available models")
                live_updates_list.append(f"{timestamp} - 🤖 Models: {count} available")
                update_displays()
                time.sleep(0.1)  # Small delay for UI update
                
            elif notification_type == "llm_called":
                model = notification.get("model", "unknown")
                workflow_steps["model_used"] = model
                
                # Infer intent from model selection (since orchestrator classifies internally)
                inferred_intent = "unknown"
                if "qwen2.5-coder" in model.lower() or "coder" in model.lower():
                    inferred_intent = "code"
                elif "wizardmath" in model.lower() or "math" in model.lower():
                    inferred_intent = "math"
                elif "llama" in model.lower():
                    inferred_intent = "general"
                
                workflow_steps["intent"] = inferred_intent
                
                step_count = max(step_count, 4)
                status_text.info(f"🚀 Using model: {model}")
                live_updates_list.append(f"{timestamp} - 🚀 Using: {model}")
                update_displays()
                time.sleep(0.1)  # Small delay for UI update
                
            elif notification_type == "final_response":
                final_ai_response = notification.get("response", "")
                step_count = 5
                progress_bar.progress(1.0)
                status_text.success("✅ Response generated!")
                live_updates_list.append(f"{timestamp} - ✨ Response ready")
                update_displays()
                
            elif notification_type == "final_result":
                result = notification.get("data", {})
                if isinstance(result, dict) and "response" in result:
                    final_ai_response = result["response"]
                step_count = 5
                progress_bar.progress(1.0)
                status_text.success("✅ Processing complete!")
                live_updates_list.append(f"{timestamp} - ✨ Final result ready")
                update_displays()
                
            elif notification_type == "stream_complete":
                progress_bar.progress(1.0)
                status_text.success("✅ Process completed!")
                live_updates_list.append(f"{timestamp} - ✅ Stream complete")
                update_displays()
                break
                
    except Exception as e:
        status_text.error(f"❌ Error during streaming: {str(e)}")
        print(f"📱 DEBUG [app.py]: Streaming error: {str(e)}")
        return None
    
    # Create final workflow summary for chat
    final_workflow = f"""🤖 **AI Workflow Summary:**

🔧 **Step 1:** classify_intent
   → Intent: {workflow_steps['intent'] or 'inferred from model selection'}

🔧 **Step 2:** detect_language
   → Language: {workflow_steps['language'] or 'unknown'}

🔧 **Step 3:** get_models
   → Models available: {workflow_steps['models_count'] or 'unknown'}

🔧 **Step 4:** send_to_llm → {workflow_steps['model_used'] or 'unknown'}
   → Processing with: {workflow_steps['model_used'] or 'unknown'}

✅ **Result:** Processing complete"""
    
    # Add both workflow summary and final response to chat
    print(f"📱 DEBUG [app.py]: Adding workflow summary and response to session messages")
    st.session_state.messages.append({"role": "assistant", "content": final_workflow})
    
    if final_ai_response:
        print(f"📱 DEBUG [app.py]: Adding final AI response: {final_ai_response[:100]}...")
        st.session_state.messages.append({"role": "assistant", "content": final_ai_response})
    
    return {
        "response": final_ai_response,
        "workflow": final_workflow,
        "status": "completed"
    }

def display_current_streaming_state(user_message):
    """Display the current streaming state with real-time updates"""
    
    # Create UI containers
    reasoning_container = st.container()
    with reasoning_container:
        st.markdown("### 🧠 AI Reasoning Process")
        
        # Progress and status
        progress_container = st.container()
        workflow_container = st.container()
        live_feed_container = st.container()
        
        # Extract current workflow state
        workflow_steps = extract_workflow_from_notifications()
        step_count = count_completed_steps()
        
        with progress_container:
            progress = min(step_count / 4.0, 1.0)
            st.progress(progress)
            
            if st.session_state.streaming_state == "starting":
                st.info("🚀 Initializing AI reasoning process...")
            elif st.session_state.streaming_state == "active":
                st.info("🔄 Processing in real-time...")
            elif st.session_state.streaming_state == "completed":
                st.success("✅ Process completed!")
            elif st.session_state.streaming_state == "error":
                st.error("❌ Error occurred")
        
        with workflow_container:
            display_workflow_progress(workflow_steps, step_count)
        
        with live_feed_container:
            display_live_notification_feed()

def extract_workflow_from_notifications():
    """Extract workflow state from current notifications"""
    steps = {
        "intent": None,
        "language": None,
        "models_count": None,
        "model_used": None
    }
    
    for notification in st.session_state.streaming_notifications:
        notification_type = notification.get("type", "")
        
        if notification_type == "intent_classified":
            steps["intent"] = notification.get("intent", "unknown")
        elif notification_type == "language_detected":
            steps["language"] = notification.get("language", "unknown")
        elif notification_type == "models_retrieved":
            steps["models_count"] = notification.get("count", 0)
        elif notification_type == "llm_called":
            steps["model_used"] = notification.get("model", "unknown")
    
    return steps

def count_completed_steps():
    """Count how many main steps have been completed"""
    completed = 0
    for notification in st.session_state.streaming_notifications:
        notification_type = notification.get("type", "")
        if notification_type in ["intent_classified", "language_detected", "models_retrieved", "llm_called"]:
            completed += 1
    return completed

def display_workflow_progress(workflow_steps, current_step):
    """Display workflow progress with real-time status"""
    
    workflow_text = "🤖 **Real-time AI Workflow:**\\n\\n"
    
    # Step 1: Intent Classification
    if workflow_steps["intent"]:
        workflow_text += f"✅ **Step 1:** classify_intent → **{workflow_steps['intent']}**\\n\\n"
    elif current_step >= 1:
        workflow_text += f"🔄 **Step 1:** classify_intent → Analyzing...\\n\\n"
    else:
        workflow_text += f"⏳ **Step 1:** classify_intent\\n\\n"
    
    # Step 2: Language Detection
    if workflow_steps["language"]:
        workflow_text += f"✅ **Step 2:** detect_language → **{workflow_steps['language']}**\\n\\n"
    elif current_step >= 2:
        workflow_text += f"🔄 **Step 2:** detect_language → Detecting...\\n\\n"
    elif workflow_steps["intent"]:
        workflow_text += f"⏳ **Step 2:** detect_language\\n\\n"
    
    # Step 3: Model Retrieval
    if workflow_steps["models_count"]:
        workflow_text += f"✅ **Step 3:** get_models → **{workflow_steps['models_count']} available**\\n\\n"
    elif current_step >= 3:
        workflow_text += f"🔄 **Step 3:** get_models → Loading...\\n\\n"
    elif workflow_steps["language"]:
        workflow_text += f"⏳ **Step 3:** get_models\\n\\n"
    
    # Step 4: LLM Processing
    if workflow_steps["model_used"]:
        workflow_text += f"✅ **Step 4:** send_to_llm → **{workflow_steps['model_used']}**\\n\\n"
    elif current_step >= 4:
        workflow_text += f"🔄 **Step 4:** send_to_llm → Processing...\\n\\n"
    elif workflow_steps["models_count"]:
        workflow_text += f"⏳ **Step 4:** send_to_llm\\n\\n"
    
    if st.session_state.streaming_final_response:
        workflow_text += f"🎉 **Complete:** Response ready!"
    
    st.markdown(workflow_text)

def display_live_notification_feed():
    """Display live feed of recent notifications"""
    if not st.session_state.streaming_notifications:
        return
    
    # Show last 5 notifications with timestamps
    recent_notifications = st.session_state.streaming_notifications[-5:]
    
    feed_text = "📡 **Live Updates:**\\n\\n"
    for i, notification in enumerate(recent_notifications):
        notification_type = notification.get("type", "unknown")
        timestamp = time.strftime("%H:%M:%S")
        
        # Add emoji and description for each notification type
        if notification_type == "intent_classified":
            intent = notification.get("intent", "unknown")
            feed_text += f"• {timestamp} - 🎯 Intent: **{intent}**\\n"
        elif notification_type == "language_detected":
            language = notification.get("language", "unknown")
            feed_text += f"• {timestamp} - 🌍 Language: **{language}**\\n"
        elif notification_type == "models_retrieved":
            count = notification.get("count", 0)
            feed_text += f"• {timestamp} - 🤖 Models: **{count} available**\\n"
        elif notification_type == "llm_called":
            model = notification.get("model", "unknown")
            feed_text += f"• {timestamp} - 🚀 Using: **{model}**\\n"
        elif notification_type == "final_response":
            feed_text += f"• {timestamp} - ✨ **Response generated!**\\n"
        else:
            feed_text += f"• {timestamp} - 📋 {notification_type}\\n"
    
    st.markdown(feed_text)

def create_workflow_summary_from_notifications():
    """Create final workflow summary for chat"""
    workflow_steps = extract_workflow_from_notifications()
    
    return f"""🤖 **AI Workflow Summary:**

🔧 **Step 1:** classify_intent
   → Intent: {workflow_steps['intent'] or 'unknown'}

🔧 **Step 2:** detect_language
   → Language: {workflow_steps['language'] or 'unknown'}

🔧 **Step 3:** get_models
   → Models available: {workflow_steps['models_count'] or 'unknown'}

🔧 **Step 4:** send_to_llm → {workflow_steps['model_used'] or 'unknown'}
   → Processing with: {workflow_steps['model_used'] or 'unknown'}

✅ **Result:** Processing complete"""

def process_stream_in_background(user_message: str):
    """Process the streaming in a background thread"""
    try:
        client = get_client()
        
        for notification in client.route_request_ai_stream(
            user_message, 
            "user", 
            st.session_state.session_id, 
            st.session_state.conversation_id
        ):
            if "error" in notification:
                st.session_state.streaming_final_response = f"Error: {notification['error']}"
                break
            
            # Add notification to session state
            st.session_state.streaming_notifications.append(notification)
            
            notification_type = notification.get("type", "")
            
            if notification_type in ["intent_classified", "language_detected", "models_retrieved", "llm_called"]:
                st.session_state.streaming_step_count += 1
            
            elif notification_type == "final_response":
                final_response = notification.get("response", "")
                st.session_state.streaming_final_response = final_response
                break
                
            elif notification_type == "final_result":
                result = notification.get("data", {})
                if isinstance(result, dict) and "response" in result:
                    st.session_state.streaming_final_response = result["response"]
                break
                
            elif notification_type == "stream_complete":
                # Mark as complete if we haven't captured the response yet
                if st.session_state.streaming_final_response is None:
                    st.session_state.streaming_final_response = "Process completed"
                break
                
    except Exception as e:
        st.session_state.streaming_final_response = f"Error: {str(e)}"

def display_streaming_progress():
    """Display the current streaming progress"""
    # Create containers for real-time updates
    reasoning_container = st.container()
    with reasoning_container:
        st.markdown("### 🧠 AI Reasoning Process")
        
        progress_container = st.container()
        workflow_container = st.container()
        
        with progress_container:
            progress = min(st.session_state.streaming_step_count / 4.0, 1.0)
            st.progress(progress)
            
            if st.session_state.streaming_active:
                st.info("🔄 Processing your request in real-time...")
            else:
                st.success("✅ Processing complete!")
        
        with workflow_container:
            # Display current workflow state based on notifications received
            workflow_steps = extract_workflow_steps_from_notifications()
            display_current_workflow(workflow_steps)

def extract_workflow_steps_from_notifications():
    """Extract workflow information from received notifications"""
    steps = {
        "intent": None,
        "language": None,
        "models_count": None,
        "model_used": None
    }
    
    for notification in st.session_state.streaming_notifications:
        notification_type = notification.get("type", "")
        
        if notification_type == "intent_classified":
            steps["intent"] = notification.get("intent", "unknown")
        elif notification_type == "language_detected":
            steps["language"] = notification.get("language", "unknown")
        elif notification_type == "models_retrieved":
            steps["models_count"] = notification.get("count", 0)
        elif notification_type == "llm_called":
            steps["model_used"] = notification.get("model", "unknown")
    
    return steps

def display_current_workflow(workflow_steps):
    """Display the current workflow state"""
    current_step = st.session_state.streaming_step_count
    
    workflow_text = "🤖 **Real-time AI Workflow:**\n\n"
    
    # Step 1: Intent Classification
    if workflow_steps["intent"]:
        workflow_text += f"✅ **Step 1:** classify_intent → {workflow_steps['intent']}\n\n"
    elif current_step >= 1:
        workflow_text += f"� **Step 1:** classify_intent → Analyzing...\n\n"
    else:
        workflow_text += f"⏳ **Step 1:** classify_intent\n\n"
    
    # Step 2: Language Detection
    if workflow_steps["language"]:
        workflow_text += f"✅ **Step 2:** detect_language → {workflow_steps['language']}\n\n"
    elif current_step >= 2:
        workflow_text += f"🔄 **Step 2:** detect_language → Detecting...\n\n"
    elif workflow_steps["intent"]:
        workflow_text += f"⏳ **Step 2:** detect_language\n\n"
    
    # Step 3: Model Retrieval
    if workflow_steps["models_count"]:
        workflow_text += f"✅ **Step 3:** get_models → {workflow_steps['models_count']} available\n\n"
    elif current_step >= 3:
        workflow_text += f"🔄 **Step 3:** get_models → Loading...\n\n"
    elif workflow_steps["language"]:
        workflow_text += f"⏳ **Step 3:** get_models\n\n"
    
    # Step 4: LLM Processing
    if workflow_steps["model_used"]:
        workflow_text += f"✅ **Step 4:** send_to_llm → {workflow_steps['model_used']}\n\n"
    elif current_step >= 4:
        workflow_text += f"🔄 **Step 4:** send_to_llm → Processing...\n\n"
    elif workflow_steps["models_count"]:
        workflow_text += f"⏳ **Step 4:** send_to_llm\n\n"
    
    if st.session_state.streaming_final_response:
        workflow_text += f"🎉 **Complete:** Response ready!"
    
    st.markdown(workflow_text)

def create_final_workflow_summary():
    """Create the final workflow summary from completed notifications"""
    workflow_steps = extract_workflow_steps_from_notifications()
    
    return f"""🤖 **AI Workflow Summary:**

🔧 **Step 1:** classify_intent
   → Intent: {workflow_steps['intent'] or 'unknown'}

🔧 **Step 2:** detect_language
   → Language: {workflow_steps['language'] or 'unknown'}

🔧 **Step 3:** get_models
   → Models available: {workflow_steps['models_count'] or 'unknown'}

🔧 **Step 4:** send_to_llm → {workflow_steps['model_used'] or 'unknown'}
   → Processing with: {workflow_steps['model_used'] or 'unknown'}

✅ **Result:** Processing complete"""

def main():
    st.set_page_config(
        page_title="Smart Code Assistant",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Check connection after setting page config
    get_api_client().show_connection_error_if_needed()
    
    st.title("💬 Smart Code Assistant")
    st.markdown("""
    <style>
    .block-container { padding-top: 1.5rem; }
    .stChatInput { position: fixed; bottom: 0; left: 0; width: 100%; background: #fff; z-index: 100; }
    </style>
    """, unsafe_allow_html=True)
    
    # Add mode selection in sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        show_reasoning = st.checkbox(
            "🧠 Show AI Reasoning Process", 
            value=True,
            help="Display step-by-step reasoning process in real-time"
        )
        st.markdown("---")
        st.markdown("**🔍 How it works:**")
        if show_reasoning:
            st.markdown("""
            1. 🎯 **Intent Classification** - Analyzes what you're asking for
            2. 🌍 **Language Detection** - Identifies the language
            3. 🤖 **Model Selection** - Chooses the best AI model
            4. 🚀 **Response Generation** - Gets your answer
            """)
        else:
            st.markdown("Enable reasoning mode to see the AI's thought process!")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Only initialize session/conversation once per Streamlit session
    if "initialized" not in st.session_state or not st.session_state.initialized:
        initialize_session_and_conversation()
        st.session_state.initialized = True
    
    # Clear Chat button
    if st.button("🧹 Clear Chat", help="Delete all messages for this conversation"):
        client = get_client()
        response = client.delete_messages(st.session_state.conversation_id)
        # Clear local chat history
        st.session_state.messages = []
        st.rerun()
    
    # Display chat history
    for msg in st.session_state.messages:
        display_chat_message(msg["role"], msg["content"])
    
    # Handle new user message
    if prompt := st.chat_input("Type your question or request (code, writing, math, translation, etc.)..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        display_chat_message("user", prompt)
        
        if show_reasoning:
            # Use real-time reasoning mode
            with st.spinner("Processing with real-time reasoning..."):
                result = display_real_time_reasoning(prompt)
                
                # After processing completes, force a rerun to show new messages
                if result and result.get("status") == "completed":
                    print(f"📱 DEBUG [app.py]: Real-time reasoning completed, forcing rerun to display new messages")
                    st.rerun()  # Force page refresh to show new messages
        else:
            # Use regular mode
            with st.spinner("Thinking..."):
                client = get_client()
                response = client.route_request_ai(prompt, "user", st.session_state.session_id, st.session_state.conversation_id)
                
                # Try to parse as JSON, extract 'response' if present, store full dict if possible
                response_obj = None
                if isinstance(response, str):
                    try:
                        response_obj = json.loads(response)
                    except Exception:
                        response_obj = None
                elif isinstance(response, dict):
                    response_obj = response
                
                # If response_obj is a dict with 'response', store full dict, else store as string
                if isinstance(response_obj, dict) and "response" in response_obj:
                    st.session_state.messages.append({"role": "assistant", "content": response_obj})
                    display_chat_message("assistant", response_obj)
                else:
                    cleaned = clean_response(response if isinstance(response, str) else str(response))
                    st.session_state.messages.append({"role": "assistant", "content": cleaned})
                    display_chat_message("assistant", cleaned)

if __name__ == "__main__":
    main()
