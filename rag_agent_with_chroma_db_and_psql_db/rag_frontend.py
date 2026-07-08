import streamlit as st
import websockets
import json
import asyncio

WS_URL = "ws://localhost:8000/ws/chat"

st.set_page_config(
    page_title="Agentic AI Chatbot",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <h3 style='text-align: center;'>
    Rag Agent
    </h3>
    """, 
    unsafe_allow_html = True
)


async def send_message_websocket(message:str, thread_id: str, user_id: str):
    try:
        async with websockets.connect(WS_URL) as websocket:
            await websocket.send(json.dumps({
                "message": message,
                "thread_id": thread_id,
                "user_id": user_id
            }))

            assistant_response = ""
            with st.chat_message("assistant"):
                placeholder = st.empty()
                placeholder.markdown("_**Thinking...**_")
            
                async for response in websocket:
                    data = json.loads(response)

                    if data["type"] == "content":
                        assistant_response += data["content"]
                        placeholder.markdown(assistant_response + "┃")
                        
                    elif data["type"] == "completed":
                        placeholder.markdown(assistant_response)

    except Exception as e:
        st.error(str(e))


message = st.chat_input(placeholder="Type here . . .", width='stretch')
if message:
    with st.chat_message('user'):
        st.write(message)
    asyncio.run(send_message_websocket(message=message, thread_id="thread_123", user_id="user_123"))