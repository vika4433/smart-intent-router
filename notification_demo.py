#!/usr/bin/env python3
"""
Real-time AI Reasoning Chat Demo

Simple chat interface that shows AI reasoning process in real-time.
"""

import streamlit as st
import requests
import json
import time
from datetime import datetime

# Configure page
st.set_page_config(
    page_title="AI Chat Demo",
    page_icon="�",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Hide sidebar completely
st.markdown("""
<style>
    .stSidebar {display: none;}
    .main > div {padding-top: 2rem;}
    
    /* Chat container */
    .chat-container {
        height: 500px;
        overflow-y: auto;
        padding: 1rem;
        border: 1px solid #ddd;
        border-radius: 10px;
        background-color: #fafafa;
        margin-bottom: 1rem;
    }
    
    /* User message */
    .user-msg {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 12px 16px;
        border-radius: 18px;
        margin: 8px 0;
        margin-left: 60px;
        text-align: left;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    
    /* AI step */
    .ai-step {
        background-color: #f8f9fa;
        color: #495057;
        padding: 8px 12px;
        border-radius: 12px;
        margin: 4px 0;
        margin-right: 60px;
        font-size: 0.9rem;
        border-left: 3px solid #007bff;
    }
    
    /* AI response */
    .ai-response {
        background-color: #e9ecef;
        color: #212529;
        padding: 12px 16px;
        border-radius: 18px;
        margin: 8px 0;
        margin-right: 60px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    
    /* Processing indicator */
    .processing {
        background-color: #fff3cd;
        color: #856404;
        padding: 8px 12px;
        border-radius: 12px;
        margin: 4px 0;
        margin-right: 60px;
        font-size: 0.9rem;
        border-left: 3px solid #ffc107;
        animation: pulse 1.5s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    
    /* Input area */
    .input-container {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: white;
        padding: 1rem;
        border-top: 1px solid #ddd;
        box-shadow: 0 -2px 10px rgba(0,0,0,0.1);
    }
    
    /* Hide streamlit elements */
    .stTextInput > div > div {
        border-radius: 25px;
    }
    
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'processing' not in st.session_state:
    st.session_state.processing = False

# Title
st.title("� AI Chat")
st.markdown("**Real-time AI reasoning process**")

# Chat display area
if st.session_state.messages:
    # Create scrollable chat container
    chat_html = '<div class="chat-container">'
    
    for msg in st.session_state.messages:
        timestamp = msg.get('timestamp', '')
        
        if msg['type'] == 'user':
            chat_html += f'''
            <div class="user-msg">
                <strong>You:</strong> {msg['content']}
                <div style="font-size: 0.8rem; margin-top: 5px; opacity: 0.8;">{timestamp}</div>
            </div>
            '''
        
        elif msg['type'] == 'processing':
            chat_html += f'''
            <div class="processing">
                ⏳ {msg['content']}
            </div>
            '''
        
        elif msg['type'] == 'ai_step':
            chat_html += f'''
            <div class="ai-step">
                {msg['content']}
            </div>
            '''
        
        elif msg['type'] == 'ai_response':
            chat_html += f'''
            <div class="ai-response">
                <strong>AI:</strong> {msg['content']}
                <div style="font-size: 0.8rem; margin-top: 5px; opacity: 0.8;">
                    Model: {msg.get('model', 'Unknown')} • {timestamp}
                </div>
            </div>
            '''
    
    chat_html += '</div>'
    st.markdown(chat_html, unsafe_allow_html=True)
else:
    st.info("👋 Start chatting! Type your message below.")

# Add some space for the fixed input
st.markdown("<br><br><br>", unsafe_allow_html=True)

# Input area (fixed at bottom)
with st.container():
    col1, col2 = st.columns([5, 1])
    
    with col1:
        user_input = st.text_input(
            "",
            placeholder="Type your message here...",
            disabled=st.session_state.processing,
            key="user_input_field"
        )
    
    with col2:
        send_clicked = st.button(
            "Send" if not st.session_state.processing else "...",
            disabled=st.session_state.processing or not user_input.strip(),
            key="send_button"
        )

# Handle send button or Enter key
if send_clicked and user_input.strip():
    st.session_state.processing = True
    
    # Add user message
    st.session_state.messages.append({
        'type': 'user',
        'content': user_input.strip(),
        'timestamp': datetime.now().strftime("%H:%M:%S")
    })
    
    # Add processing indicator
    st.session_state.messages.append({
        'type': 'processing',
        'content': 'Processing your request...'
    })
    
    # Force refresh to show processing
    st.rerun()

# Process the request in real-time
if st.session_state.processing and len(st.session_state.messages) > 0:
    last_msg = st.session_state.messages[-1]
    
    if last_msg['type'] == 'processing':
        # Remove processing indicator
        st.session_state.messages.pop()
        
        # Get the user message
        user_msg = None
        for msg in reversed(st.session_state.messages):
            if msg['type'] == 'user':
                user_msg = msg['content']
                break
        
        if user_msg:
            try:
                # Make streaming request
                response = requests.post(
                    "http://localhost:8000/route_request_ai_stream",
                    json={
                        "user_message": user_msg,
                        "user_id": "demo_user",
                        "max_iterations": 3
                    },
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "text/event-stream"
                    },
                    stream=True,
                    timeout=120  # Increased timeout to 2 minutes for LLM processing
                )
                
                if response.status_code == 200:
                    for line in response.iter_lines():
                        if line:
                            line = line.decode('utf-8')
                            if line.startswith('data: '):
                                try:
                                    data = json.loads(line[6:])
                                    
                                    # Process different event types
                                    if data['type'] == 'process_started':
                                        st.session_state.messages.append({
                                            'type': 'ai_step',
                                            'content': '🔄 Starting AI routing process...'
                                        })
                                        st.rerun()
                                    
                                    elif data['type'] == 'orchestrator_selected':
                                        model = data.get('model', 'Unknown')
                                        st.session_state.messages.append({
                                            'type': 'ai_step',
                                            'content': f'🧠 Orchestrator selected: {model}'
                                        })
                                        st.rerun()
                                    
                                    elif data['type'] == 'intent_classified':
                                        intent = data.get('intent', 'unknown')
                                        st.session_state.messages.append({
                                            'type': 'ai_step',
                                            'content': f'classified intent: {intent}'
                                        })
                                        st.rerun()
                                    
                                    elif data['type'] == 'language_detected':
                                        language = data.get('language', 'unknown')
                                        st.session_state.messages.append({
                                            'type': 'ai_step',
                                            'content': f'call tool: detect_language'
                                        })
                                        st.session_state.messages.append({
                                            'type': 'ai_step',
                                            'content': f'language: {language}'
                                        })
                                        st.rerun()
                                    
                                    elif data['type'] == 'models_retrieved':
                                        count = data.get('count', 0)
                                        st.session_state.messages.append({
                                            'type': 'ai_step',
                                            'content': f'call tool: get_models'
                                        })
                                        st.session_state.messages.append({
                                            'type': 'ai_step',
                                            'content': f'models: {count} available'
                                        })
                                        st.rerun()
                                    
                                    elif data['type'] == 'llm_called':
                                        model = data.get('model', 'unknown')
                                        st.session_state.messages.append({
                                            'type': 'ai_step',
                                            'content': f'call tool: send_to_llm'
                                        })
                                        st.rerun()
                                    
                                    elif data['type'] == 'final_result':
                                        result_data = data.get('data', {})
                                        response_text = result_data.get('response', 'No response')
                                        model_used = result_data.get('orchestrator_model', 'Unknown')
                                        
                                        st.session_state.messages.append({
                                            'type': 'ai_step',
                                            'content': 'final response:'
                                        })
                                        st.session_state.messages.append({
                                            'type': 'ai_response',
                                            'content': response_text,
                                            'model': model_used,
                                            'timestamp': datetime.now().strftime("%H:%M:%S")
                                        })
                                        break
                                    
                                    elif data['type'] == 'error':
                                        st.session_state.messages.append({
                                            'type': 'ai_step',
                                            'content': f'❌ Error: {data.get("message", "Unknown error")}'
                                        })
                                        break
                                        
                                except json.JSONDecodeError:
                                    continue
                else:
                    st.session_state.messages.append({
                        'type': 'ai_step',
                        'content': f'❌ Server error: HTTP {response.status_code}'
                    })
                    
            except Exception as e:
                st.session_state.messages.append({
                    'type': 'ai_step',
                    'content': f'❌ Connection error: {str(e)}'
                })
        
        # Reset processing state
        st.session_state.processing = False
        st.rerun()

# Auto-scroll JavaScript
st.markdown("""
<script>
    // Auto-scroll to bottom
    const chatContainer = document.querySelector('.chat-container');
    if (chatContainer) {
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }
</script>
""", unsafe_allow_html=True)
