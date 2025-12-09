import streamlit as st
import uuid
import time
import markdown
import re
import asyncio
import websockets
import json
import requests
from typing import Dict, List

# Page configuration
st.set_page_config(
    page_title="Agentic AI Chatbot",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Configuration
API_BASE_URL = "http://localhost:8000"
WS_URL = "ws://localhost:8000/ws/chat"

# Functions
def generate_thread_id():
    """Generate a unique thread ID using UUID."""
    return str(uuid.uuid4())

def reset_chat():
    """Reset the chat session with a new thread ID."""
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    st.session_state['message_history'] = []

def add_thread_id(thread_id):
    """Add a thread ID to the chat threads list if it doesn't exist."""
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].insert(0, thread_id)

def fetch_threads() -> List[str]:
    """Fetch all threads from the API."""
    try:
        response = requests.get(f"{API_BASE_URL}/threads", timeout=5)
        response.raise_for_status()
        return response.json()["threads"]
    except Exception as e:
        st.error(f"Error fetching threads: {e}")
        return []

def fetch_conversation_title(thread_id: str) -> str:
    """Fetch conversation title from the API."""
    try:
        response = requests.get(f"{API_BASE_URL}/conversation/{thread_id}/title", timeout=5)
        response.raise_for_status()
        return response.json()["title"]
    except Exception:
        return "Empty chat"

def load_conversation(thread_id: str) -> List[Dict]:
    """Load conversation history from the API."""
    try:
        response = requests.get(f"{API_BASE_URL}/conversation/{thread_id}", timeout=5)
        response.raise_for_status()
        return response.json()["messages"]
    except Exception as e:
        st.error(f"Error loading conversation: {e}")
        return []

def display_message(message, role):
    """Display a message with appropriate styling based on role."""
    html_content = markdown.markdown(message)
    
    container_class = "message-container-user" if role == "user" else "message-container-assistant"
    message_class = "user-message" if role == "user" else "assistant-message"
    
    html = f"""
    <div class='{container_class}'>
        <div style='display:{"none" if role == "user" else "inline-block"};'>✨</div>
        <div class='{message_class}'>{html_content}</div>
        <div style='display:{"inline-block" if role == "user" else "none"};'>👤</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

async def send_message_websocket(message: str, thread_id: str, response_container):
    """Send message via WebSocket and handle streaming response."""
    full_response = ""
    tools_set = set()
    
    # Show thinking animation
    thinking_html = """
    <div class='message-container-assistant'>
        <div>✨</div>
        <div class='assistant-message'>
            <div class='thinking-container'>
                <span class='thinking-text'>Thinking</span>
                <div class='thinking-dots'>
                    <span class='dot'></span>
                    <span class='dot'></span>
                    <span class='dot'></span>
                </div>
            </div>
        </div>
    </div>
    """
    response_container.markdown(thinking_html, unsafe_allow_html=True)
    
    try:
        async with websockets.connect(WS_URL) as websocket:
            # Send the message
            await websocket.send(json.dumps({
                "message": message,
                "thread_id": thread_id
            }))
            
            # Receive streaming responses
            async for response in websocket:
                data = json.loads(response)
                
                if data["type"] == "tool_call":
                    # Handle tool calls
                    tools = data["tools"]
                    for tool in tools:
                        if tool == 'search':
                            tools_set.add(' 🔍Search')
                        elif tool == 'calculator':
                            tools_set.add(' 🧮Calculator')
                        else:
                            tools_set.add(f' 🔧{tool}')
                    
                    # Build tool status display
                    tools_html = f"<div class='tools'>Using:</div>"
                    tool_lines = ""
                    for tool in tools_set:
                        tool_lines += f"<div class='tool-line'>└─{tool} tool</div>"
                    
                    thinking_with_tools_html = f"""
                    <div class='message-container-assistant'>
                        <div>✨</div>
                        <div class='assistant-message'>
                            <div class='thinking-container'>
                                <span class='thinking-text'>Thinking</span>
                                <div class='thinking-dots'>
                                    <span class='dot'></span>
                                    <span class='dot'></span>
                                    <span class='dot'></span>
                                </div>
                            </div>
                            {tools_html}
                            {tool_lines}
                        </div>
                    </div>
                    """
                    response_container.markdown(thinking_with_tools_html, unsafe_allow_html=True)
                
                elif data["type"] == "content":
                    # Stream the content
                    new_content = data["content"]
                    if new_content and not new_content.isspace():
                        parts = re.split(r'(\s+)', new_content)
                        for part in parts:
                            full_response += part
                            html_content = markdown.markdown(full_response + '┃')
                            
                            message_container = f"""
                            <div class='message-container-assistant'>
                                <div>✨</div>
                                <div class='assistant-message'>{html_content}</div>
                            </div>
                            """
                            response_container.markdown(message_container, unsafe_allow_html=True)
                            await asyncio.sleep(0.006)
                
                elif data["type"] == "complete":
                    # Final update to remove cursor
                    if full_response:
                        html_content = markdown.markdown(full_response)
                        message_container = f"""
                        <div class='message-container-assistant'>
                            <div>✨</div>
                            <div class='assistant-message'>{html_content}</div>
                        </div>
                        """
                        response_container.markdown(message_container, unsafe_allow_html=True)
                    break
                
                elif data["type"] == "error":
                    st.error(f"Error: {data['content']}")
                    break
    
    except Exception as e:
        st.error(f"WebSocket error: {e}")
        full_response = "Sorry, I encountered an error processing your message."
    
    return full_response

# Session states
if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = fetch_threads()

if 'conversation_titles' not in st.session_state:
    st.session_state['conversation_titles'] = {}

# Custom CSS
st.markdown("""
<style>
    h2 {
        font-style: italic;
    }
    p {
        color: #FFFFFF;
        font-style: italic;        
    }
    .title {
        display: flex;
        justify-content: center;
        align-items: center;
        flex-direction: column;
        text-align: center;
    }
    .user-message {
        padding: 9px 16px 0 12px;
        border-radius: 18px 0 18px 18px;
        background-color: #262730;
        color: white;
        margin: 13px 0 15px;
        display: inline-block;
        max-width: 100%;
        text-align: left;
        word-wrap: break-word;
    }
    .assistant-message {
        padding: 9px 16px 9px 12px;
        border-radius: 0 18px 18px 18px;
        background-color: black;
        color: white;
        margin: 8px 0;
        margin-bottom: 50px;
        display: inline-block;
        max-width: 100%;
        text-align: left;
        word-wrap: break-word;
    }
    .message-container-user {
        display: flex;
        justify-content: flex-end;
        width: 100%;
    }
    .message-container-assistant {
        display: flex;
        justify-content: flex-start;
        width: 100%;
    }
    .thinking-container {
        display: flex;
        align-items: center;
        gap: 4px;
        margin-bottom: 2px;
    }
    .thinking-text {
        font-style: italic;
        color: #888;
    }
    .thinking-dots {
        padding-top:9px;
        display: flex;
        gap: 3px;
    }
    .dot {
        width: 4px;
        height: 4px;
        background-color: #888;
        border-radius: 50%;
        animation: typing 1.4s infinite ease-in-out;
    }
    .dot:nth-child(1) {
        animation-delay: -0.32s;
    }
    .dot:nth-child(2) {
        animation-delay: -0.16s;
    }
    .tools {
        color: #888;
        font-size: 12px;
        font-family: monospace;
        line-height: 1.2;
        padding-left: 16px;
    }
    .tool-line {
        color: #888;
        font-size: 10px;
        margin: 1px 0;
        font-family: monospace;
        line-height: 1.4;
        padding-left: 32px;
    }
    @keyframes typing {
        0%, 80%, 100% {
            transform: scale(0);
            opacity: 0.5;
        }
        40% {
            transform: scale(1);
            opacity: 1;
        }
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    if st.button('Start New Chat +') and st.session_state['message_history']:
        reset_chat()
    
    st.title('Recent Chats')
    st.markdown('___')
    
    for thread_id in st.session_state['chat_threads']:
        # Get or compute conversation title
        if thread_id not in st.session_state['conversation_titles']:
            st.session_state['conversation_titles'][thread_id] = fetch_conversation_title(thread_id)
        
        title = st.session_state['conversation_titles'][thread_id]
        
        if st.button(f"{title}", key=thread_id):
            st.session_state['thread_id'] = thread_id
            messages = load_conversation(thread_id)
            st.session_state['message_history'] = messages
            st.rerun()

# Main page
st.markdown("""
    <div class='title'>
        <h2>Agentic AI Chatbot 🤖</h2>
        <p>'Toqeer's personal assistant'</p>
    </div>
""", unsafe_allow_html=True)

# Display message history
for message in st.session_state.message_history:
    display_message(message['content'], message['role'])

# User input and processing
user_input = st.chat_input("Type here . . .")

if user_input:
    # Add user message to history
    st.session_state.message_history.append({"role": "user", "content": user_input})
    add_thread_id(st.session_state['thread_id'])
    display_message(user_input, 'user')

    # Update conversation title
    current_thread_id = st.session_state['thread_id']
    words = user_input.split()[:4]
    title = ' '.join(words)
    if len(user_input.split()) > 4:
        title += '...'
    st.session_state['conversation_titles'][current_thread_id] = title

    # Container for response
    response_container = st.empty()
    
    # Send message via WebSocket
    full_response = asyncio.run(send_message_websocket(
        user_input, 
        st.session_state['thread_id'],
        response_container
    ))
    
    # Add assistant response to history
    if full_response:
        st.session_state.message_history.append({"role": "assistant", "content": full_response})