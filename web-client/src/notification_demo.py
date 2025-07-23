#!/usr/bin/env python3
"""
Real-time AI Reasoning Process Demo
Shows step-by-step notifications from the Smart Intent Router
"""

import streamlit as st
import requests
import json
import time
from typing import Dict, List

# Configuration
SERVER_URL = "http://localhost:8000"

def main():
    st.set_page_config(
        page_title="AI Reasoning Process Demo",
        page_icon="🧠",
        layout="wide"
    )
    
    st.title("🧠 Smart Intent Router - AI Reasoning Process")
    st.markdown("See the AI's step-by-step thought process in real-time!")
    
    # Input section
    st.header("📝 Ask a Question")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        user_message = st.text_input(
            "Enter your message:",
            placeholder="e.g., 'write python code to reverse a string' or 'what is the square root of 144?'",
            key="user_input"
        )
    
    with col2:
        st.write("")  # Spacing
        submit_button = st.button("🚀 Process Request", type="primary")
    
    # Example questions
    st.markdown("### 💡 Try these examples:")
    example_col1, example_col2, example_col3 = st.columns(3)
    
    with example_col1:
        if st.button("🐍 Python Code Example"):
            st.session_state.user_input = "write python code to reverse a string"
            st.rerun()
    
    with example_col2:
        if st.button("🔢 Math Example"):
            st.session_state.user_input = "what is the square root of 144?"
            st.rerun()
    
    with example_col3:
        if st.button("💬 General Example"):
            st.session_state.user_input = "tell me about artificial intelligence"
            st.rerun()
    
    # Process request
    if submit_button and user_message:
        st.header("🔍 AI Reasoning Process")
        
        # Create containers for real-time updates
        process_container = st.container()
        
        with process_container:
            # Show processing indicator
            progress_placeholder = st.empty()
            steps_placeholder = st.empty()
            
            with progress_placeholder:
                st.info("🔄 Starting AI reasoning process...")
            
            # Make request to streaming endpoint
            try:
                response = requests.post(
                    f"{SERVER_URL}/route_request_ai_stream",
                    json={"user_message": user_message},
                    stream=True,
                    timeout=60
                )
                
                if response.status_code == 200:
                    steps = []
                    final_result = None
                    
                    progress_placeholder.success("✅ Connected to AI reasoning stream")
                    
                    # Process Server-Sent Events
                    for line in response.iter_lines():
                        if line:
                            line = line.decode('utf-8')
                            if line.startswith('data: '):
                                try:
                                    data = json.loads(line[6:])  # Remove 'data: ' prefix
                                    
                                    if data.get('type') == 'process_started':
                                        steps.append({"step": "🔍", "message": data.get('message', ''), "time": time.time()})
                                    
                                    elif data.get('type') == 'orchestrator_selected':
                                        steps.append({"step": "🧠", "message": data.get('message', ''), "time": time.time()})
                                    
                                    elif data.get('type') == 'intent_classified':
                                        steps.append({"step": "🎯", "message": data.get('message', ''), "time": time.time()})
                                    
                                    elif data.get('type') == 'language_detected':
                                        steps.append({"step": "🌍", "message": data.get('message', ''), "time": time.time()})
                                    
                                    elif data.get('type') == 'models_retrieved':
                                        steps.append({"step": "🤖", "message": data.get('message', ''), "time": time.time()})
                                    
                                    elif data.get('type') == 'llm_called':
                                        steps.append({"step": "🚀", "message": data.get('message', ''), "time": time.time()})
                                    
                                    elif data.get('type') == 'final_response':
                                        steps.append({"step": "✅", "message": data.get('message', ''), "time": time.time()})
                                    
                                    elif data.get('type') == 'final_result':
                                        final_result = data.get('data', {})
                                    
                                    elif data.get('type') == 'error':
                                        steps.append({"step": "❌", "message": f"Error: {data.get('message', '')}", "time": time.time()})
                                    
                                    # Update the display
                                    with steps_placeholder.container():
                                        st.markdown("### 📋 Reasoning Steps:")
                                        for i, step in enumerate(steps, 1):
                                            st.markdown(f"**Step {i}:** {step['step']} {step['message']}")
                                        
                                        if final_result:
                                            st.markdown("---")
                                            st.markdown("### 🎉 Final Result:")
                                            
                                            # Display the response
                                            response_text = final_result.get('response', 'No response')
                                            st.markdown(f"**Response:** {response_text}")
                                            
                                            # Display metadata
                                            st.markdown("**Metadata:**")
                                            col1, col2, col3 = st.columns(3)
                                            with col1:
                                                st.metric("Orchestrator", final_result.get('orchestrator_model', 'Unknown'))
                                            with col2:
                                                st.metric("Iterations", final_result.get('iterations', 0))
                                            with col3:
                                                st.metric("Workflow", final_result.get('workflow', 'Unknown'))
                                    
                                except json.JSONDecodeError:
                                    continue
                    
                else:
                    st.error(f"❌ Error: Server returned status {response.status_code}")
                    st.write(response.text)
                    
            except requests.exceptions.RequestException as e:
                st.error(f"❌ Connection error: {str(e)}")
                st.markdown("""
                **Troubleshooting:**
                1. Make sure the Smart Intent Router server is running on port 8000
                2. Check that LM Studio is running and has models loaded
                3. Verify the server configuration is correct
                """)

if __name__ == "__main__":
    main()
