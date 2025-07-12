import streamlit as st
import re
import os
import asyncio
from typing import Optional
from dotenv import load_dotenv
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
import json

# Load environment variables
load_dotenv()
SERVER_URL = os.getenv("SERVER_URL", "http://localhost:3000")
MCP_ROUTE_URL = f"{SERVER_URL.rstrip('/')}/mcp/"

# Global async MCP session for Streamlit
class MCPClient:
    def __init__(self):
        pass  # No persistent session/streams

    async def call_tool(self, tool: str, tool_args: dict) -> str:
        streams_context = streamablehttp_client(url=MCP_ROUTE_URL)
        read_stream, write_stream, _ = await streams_context.__aenter__()
        session_context = ClientSession(read_stream, write_stream)
        session = await session_context.__aenter__()
        await session.initialize()
        try:
            result = await session.call_tool(tool, tool_args)
            # Robustly extract text from CallToolResult
            if hasattr(result, 'content'):
                content = result.content
                if isinstance(content, list) and len(content) > 0 and hasattr(content[0], 'text'):
                    return content[0].text
                elif isinstance(content, str):
                    return content
                else:
                    return str(content)
            else:
                return str(result)
        finally:
            await session.__aexit__(None, None, None)
            await streams_context.__aexit__(None, None, None)

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
    # Step 1: Create session (only if not already in session_state)
    tool_args = {"user_id": "user"}
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    response = loop.run_until_complete(st.session_state.mcp_client.call_tool("create_session", tool_args))
    session_id = extract_session_id(response)
    st.session_state.session_id = session_id

    # Step 2: Get conversations for this user
    conversation_id = None
    tool_args = {"user_id": "user"}
    response = loop.run_until_complete(st.session_state.mcp_client.call_tool("get_conversations", tool_args))
    try:
        conversations = response["conversations"] if isinstance(response, dict) else eval(response)["conversations"]
        if conversations and isinstance(conversations, list) and len(conversations) > 0:
            conversation_id = conversations[0]["conversation_id"] if isinstance(conversations[0], dict) and "conversation_id" in conversations[0] else conversations[0]
    except Exception:
        conversation_id = None

    # Step 3: If no conversation, create one
    if not conversation_id and session_id:
        tool_args = {"user_id": "user", "session_id": session_id, "title": "New Conversation"}
        response = loop.run_until_complete(st.session_state.mcp_client.call_tool("create_conversation", tool_args))
        try:
            conversation_id = response["conversation_id"] if isinstance(response, dict) else eval(response)["conversation_id"]
        except Exception:
            conversation_id = None
    st.session_state.conversation_id = conversation_id

    # Step 4: Fetch and display messages for the latest conversation
    st.session_state.messages = []
    if conversation_id:
        tool_args = {"conversation_id": conversation_id}
        response = loop.run_until_complete(st.session_state.mcp_client.call_tool("get_messages", tool_args))
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

def main():
    st.set_page_config(
        page_title="Smart Code Assistant",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    st.title("💬 Smart Code Assistant")
    st.markdown("""
    <style>
    .block-container { padding-top: 1.5rem; }
    .stChatInput { position: fixed; bottom: 0; left: 0; width: 100%; background: #fff; z-index: 100; }
    </style>
    """, unsafe_allow_html=True)
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "mcp_client" not in st.session_state:
        st.session_state.mcp_client = MCPClient()
    # Only initialize session/conversation once per Streamlit session
    if "initialized" not in st.session_state or not st.session_state.initialized:
        initialize_session_and_conversation()
        st.session_state.initialized = True
    # Display chat history
    for msg in st.session_state.messages:
        display_chat_message(msg["role"], msg["content"])
    # Handle new user message
    if prompt := st.chat_input("Type your question or request (code, writing, math, translation, etc.)..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        display_chat_message("user", prompt)
        with st.spinner("Thinking..."):
            tool_args = {
                "message": prompt,
                "user_id": "user",
                "session_id": st.session_state.session_id,
                "conversation_id": st.session_state.conversation_id
            }
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            response = loop.run_until_complete(st.session_state.mcp_client.call_tool("route_request", tool_args))
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
