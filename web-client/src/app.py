import gradio as gr
import asyncio
import httpx
import os
from dotenv import load_dotenv

load_dotenv()
SERVER_URL = os.getenv("SERVER_URL", "http://localhost:8080")

# Helper to format messages for gr.Chatbot
def format_conversation(messages):
    conversation = []
    user_msg = None
    for msg in messages:
        if msg["role"] == "user":
            user_msg = msg["content"]
        elif msg["role"] == "assistant":
            conversation.append((user_msg, msg["content"]))
            user_msg = None
    if user_msg is not None:
        conversation.append((user_msg, None))
    return conversation

# Async function to call the HTTP streaming endpoint
async def send_message_http(state, user_input):
    if not user_input.strip():
        yield state, format_conversation(state["messages"]), ""
        return
    messages = state["messages"]
    messages.append({"role": "user", "content": user_input})
    reply = ""
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", f"{SERVER_URL}/route_request_stream", json={"messages": messages}) as resp:
            async for chunk in resp.aiter_text():
                reply += chunk
                yield state, format_conversation(messages + [{"role": "assistant", "content": reply}]), ""
    try:
        import json
        data = json.loads(reply)
        if "error" in data:
            reply = f"[ERROR] {data['error']}"
    except Exception:
        pass
    messages.append({"role": "assistant", "content": reply})
    state["messages"] = messages
    yield state, format_conversation(messages), ""

# Handler to clear current chat
def clear_chat():
    return {"messages": []}, [], ""

# Build Gradio interface with sidebar and chat area
def build_gradio_interface():
    initial_state = {"messages": []}
    with gr.Blocks(title="Smart Intent Router Chat") as demo:
        gr.Markdown("## 💬 Smart Intent Router")
        state_obj = gr.State(value=initial_state)
        chatbot = gr.Chatbot(label=None, show_label=False, bubble_full_width=False)
        with gr.Row():
            user_input = gr.Textbox(
                placeholder="Type your message...", label=None, scale=8
            )
            send_button = gr.Button("Send", scale=1)
            clear_button = gr.Button("🗑️ Clear", scale=1)
        send_button.click(
            fn=send_message_http,
            inputs=[state_obj, user_input],
            outputs=[state_obj, chatbot, user_input],
            concurrency_limit=1,
            queue=True,
        )
        user_input.submit(
            fn=send_message_http,
            inputs=[state_obj, user_input],
            outputs=[state_obj, chatbot, user_input],
            concurrency_limit=1,
            queue=True,
        )
        clear_button.click(
            fn=clear_chat,
            inputs=None,
            outputs=[state_obj, chatbot, user_input],
            show_progress=False
        )
        user_input.submit(fn=lambda: "", inputs=None, outputs=user_input)
    return demo

if __name__ == "__main__":
    demo = build_gradio_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)