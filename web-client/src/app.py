import streamlit as st
import httpx
import os
import re
import asyncio
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()
SERVER_URL = os.getenv("SERVER_URL", "http://localhost:8080")

def clean_response(response: str) -> str:
    """Fix common formatting issues in LLM responses"""
    # Normalize code block markers
    response = re.sub(r'```(\w*)', r'\n```\1\n', response)
    response = response.replace('```', '\n```\n')
    # Fix pip install commands
    response = re.sub(r'pip install py pdf2', 'pip install pypdf2', response)
    return response.strip()

def display_markdown_with_code(response: str):
    """Render markdown with perfect code block handling"""
    # Inject custom CSS
    st.markdown("""
    <style>
        /* Main text container */
        .stMarkdown {
            line-height: 1.7;
            font-size: 16px;
            color: #333;
        }
        
        /* Code block styling */
        div[data-testid="stCodeBlock"] {
            margin: 1.5em 0;
            border-radius: 8px;
            border-left: 4px solid #4e8cff;
            background: #f8f9fa;
        }
        
        /* Code text */
        pre {
            white-space: pre-wrap;
            font-family: 'SFMono-Regular', Menlo, monospace;
            padding: 1em !important;
            margin: 0;
        }
        
        /* Inline code */
        code:not(pre code) {
            background: #f3f4f6;
            padding: 0.2em 0.4em;
            border-radius: 3px;
            font-size: 0.9em;
            font-family: 'SFMono-Regular', Menlo, monospace;
        }
        
        /* Fix spacing between paragraphs */
        .stMarkdown p {
            margin-bottom: 1em;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Split response into segments
    segments = re.split(r'(```[\s\S]*?```)', response)
    
    for segment in segments:
        if segment.startswith('```'):
            # Process code block
            language_match = re.match(r'```(\w+)', segment)
            language = language_match.group(1) if language_match else ''
            code_content = re.sub(r'```(\w+)?\n?', '', segment).replace('```', '')
            
            if language:
                st.markdown(f'<div style="font-family: monospace; color: #666; font-size: 0.9em;">{language}</div>', 
                           unsafe_allow_html=True)
            st.code(code_content, language=language.lower() if language else None)
        else:
            # Process regular text
            if segment.strip():
                st.markdown(segment, unsafe_allow_html=True)

async def send_message(messages: List[Dict[str, Any]]) -> str:
    """Send messages to API endpoint"""
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
            response = await client.post(
                f"{SERVER_URL}/route_request_stream",
                json={"messages": messages}
            )
            response.raise_for_status()
            return clean_response(response.text)
    except Exception as e:
        return f"Error: {str(e)}"

def display_chat_message(role: str, content: str):
    """Display chat message with proper formatting"""
    with st.chat_message(role):
        if role == "assistant":
            if content.startswith("Error:"):
                st.error(content)
            else:
                display_markdown_with_code(content)
        else:
            st.markdown(content)

def main():
    st.set_page_config(
        page_title="Smart Code Assistant",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    st.title("💬 Smart Code Assistant")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for msg in st.session_state.messages:
        display_chat_message(msg["role"], msg["content"])
    
    # Handle user input
    if prompt := st.chat_input("Ask about Python code..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        display_chat_message("user", prompt)
        
        with st.spinner("Generating response..."):
            try:
                response = asyncio.run(send_message(st.session_state.messages))
                st.session_state.messages.append({"role": "assistant", "content": response})
                display_chat_message("assistant", response)
            except Exception as e:
                st.error(f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    main()