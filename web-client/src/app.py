import gradio as gr
import asyncio
import httpx
import os
from dotenv import load_dotenv
import re

load_dotenv()
SERVER_URL = os.getenv("SERVER_URL", "http://localhost:8080")

def ensure_code_blocks(text: str) -> str:
    """
    Ensure proper Markdown formatting and handle single-line code blocks as multi-line.
    """
    # Normalize triple backticks if not typed correctly
    text = re.sub(r"```(python)?", "```python", text)

    # Split once at the first code block
    parts = text.split("```python")
    if len(parts) < 2:
        return text  # No code block found

    before_code = parts[0]
    rest = "```python".join(parts[1:])
    code_parts = rest.split("```", 1)

    code = code_parts[0].strip()
    after_code = code_parts[1] if len(code_parts) > 1 else ""

    # If code is a one-liner, format it as multiline
    if '\n' not in code and ';' not in code:
        # Try to split on common patterns like `def x(): return y`
        if "return " in code:
            code = code.replace("): ", "):\n    ", 1)

    multiline_code = f"```python\n{code}\n```"

    return f"{before_code}\n\n{multiline_code}\n\n{after_code.strip()}"

def format_conversation(messages):
    """Ensures newlines are preserved in the output"""
    formatted = []
    for msg in messages:
        if msg["role"] == "user":
            formatted.append(f"**You:** {msg['content']}\n")
        elif msg["role"] == "assistant":
            content = msg['content']
            
            # First preserve all newlines by replacing them with a special marker
            content = content.replace('\n', '⏎')
            
            # Apply code block formatting
            content = ensure_code_blocks(content)
            
            # Restore newlines (they'll now be inside proper markdown blocks)
            content = content.replace('⏎', '\n')
            
            formatted.append(f"**Assistant:**\n\n{content}\n")
    
    # Use TWO newlines between messages to ensure proper spacing
    return "\n\n---\n\n".join(formatted)

async def send_message_http(state, user_input):
    """Send message to server and stream the response."""
    if not user_input.strip():
        yield state, format_conversation(state["messages"]), ""
        return

    messages = state["messages"]
    messages.append({"role": "user", "content": user_input})
    reply = ""

    try:
        async with httpx.AsyncClient(timeout=100.0) as client:
            async with client.stream(
                "POST",
                f"{SERVER_URL}/route_request_stream",
                json={"messages": messages}
            ) as resp:
                
                if resp.status_code != 200:
                    reply = f"Server error: {resp.status_code}"
                    messages.append({"role": "assistant", "content": reply})
                    yield state, format_conversation(messages), ""
                    return
                
                async for chunk in resp.aiter_text():
                    reply += chunk
                    yield state, format_conversation(messages + [{"role": "assistant", "content": reply}]), ""

    except httpx.ConnectError:
        reply = "Error: Could not connect to server. Please check if it's running."
    except Exception as e:
        reply = f"Error: {str(e)}"
    
    messages.append({"role": "assistant", "content": reply})
    yield state, format_conversation(messages), ""

def clear_chat():
    """Clear the chat history."""
    return {"messages": []}, "", ""

def build_gradio_interface():
    """Build and configure the Gradio interface."""
    initial_state = {"messages": []}

    with gr.Blocks(css="""
    /* Force all whitespace to be preserved */
    .gr-markdown {
        white-space: pre-wrap !important;
    }
    
    /* Code blocks specifically */
    .gr-markdown pre {
        white-space: pre !important;
        display: block !important;
        margin: 1em 0 !important;
    }
    
    /* Regular text with preserved newlines */
    .gr-markdown p {
        white-space: pre-wrap !important;
        margin: 1em 0 !important;
    }
    
    /* Lists with preserved formatting */
    .gr-markdown ul, .gr-markdown ol {
        white-space: normal !important;
    }
""") as demo:

        gr.Markdown("## 💬 Smart Intent Router")

        state_obj = gr.State(value=initial_state)
        chat_display = gr.Markdown(value="", elem_id="chat-markdown", elem_classes="chat-box")
        user_input = gr.Textbox(
            placeholder="Type your message...",
            label="Enter text",
            show_label=False
        )

        with gr.Row():
            send_button = gr.Button("Submit", scale=1)
            clear_button = gr.Button("Clear", scale=1)

        send_button.click(
            fn=send_message_http,
            inputs=[state_obj, user_input],
            outputs=[state_obj, chat_display, user_input],
            concurrency_limit=1,
            queue=True,
        )

        user_input.submit(
            fn=send_message_http,
            inputs=[state_obj, user_input],
            outputs=[state_obj, chat_display, user_input],
            concurrency_limit=1,
            queue=True,
        )

        clear_button.click(
            fn=clear_chat,
            inputs=None,
            outputs=[state_obj, chat_display, user_input],
            show_progress=False
        )

        user_input.submit(fn=lambda: "", inputs=None, outputs=user_input)

    return demo

if __name__ == "__main__":
    demo = build_gradio_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)