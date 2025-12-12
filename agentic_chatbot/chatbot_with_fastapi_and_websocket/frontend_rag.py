import streamlit as st
import uuid
import time
import markdown
import asyncio
import websockets
import json
import requests
import re
from typing import Dict, List

st.set_page_config(
    page_title="Agentic AI Chatbot",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed"
)

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
    st.session_state['uploaded_docs'] = []

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

def fetch_uploaded_docs(thread_id: str) -> List[Dict]:
    """Fetch uploaded documents for a thread."""
    try:
        response = requests.get(f"{API_BASE_URL}/documents/{thread_id}", timeout=5)
        response.raise_for_status()
        return response.json()["documents"]
    except Exception:
        return []

def upload_files(files, thread_id: str) -> bool:
    """Upload multiple files to the API."""
    try:
        files_data = []
        for file in files:
            files_data.append(('files', (file.name, file.getvalue(), file.type)))
        
        response = requests.post(
            f"{API_BASE_URL}/upload",
            params={'thread_id': thread_id},
            files=files_data,
            timeout=300  # 5 minutes timeout for large files
        )
        response.raise_for_status()
        return True
    except Exception as e:
        st.error(f"Error uploading files: {e}")
        return False

def delete_document(thread_id: str, filename: str) -> bool:
    """Delete a document from the knowledge base."""
    try:
        response = requests.delete(
            f"{API_BASE_URL}/documents/{thread_id}/{filename}",
            timeout=10
        )
        response.raise_for_status()
        return True
    except Exception as e:
        st.error(f"Error deleting document: {e}")
        return False

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
            await websocket.send(json.dumps({
                "message": message,
                "thread_id": thread_id
            }))
            
            async for response in websocket:
                data = json.loads(response)
                
                if data["type"] == "tool_call":
                    tools = data["tools"]
                    for tool in tools:
                        if tool == 'search':
                            tools_set.add('Web Search')
                        elif tool == 'calculator':
                            tools_set.add('Calculator')
                        elif tool == 'retriever':
                            tools_set.add('Document Retriever')
                        else:
                            tools_set.add(tool.replace('_', ' ').title())
                    
                    tools_html = f"<div class='tools'>Using:</div>"
                    tool_lines = ""
                    for tool in tools_set:
                        tool_lines += f"<div class='tool-line'>└─ {tool} tool</div>"
                    
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

if 'uploaded_docs' not in st.session_state:
    st.session_state['uploaded_docs'] = fetch_uploaded_docs(st.session_state['thread_id'])

if 'show_upload' not in st.session_state:
    st.session_state['show_upload'] = False

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
        # border-radius: 0 18px 18px 18px;
        # background-color: black;
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
        padding-top: 9px;
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
    .section-header {
        font-size: 16px;
        font-weight: 600;
        color: #fff;
        margin: 16px 0 8px 0;
        padding: 0;
    }
    .doc-count {
        font-size: 12px;
        color: #888;
        margin-left: 4px;
    }
    .doc-item {
        padding: 6px 10px;
        margin: 4px 0;
        background-color: #1a1a1a;
        border-left: 2px solid #444;
        border-radius: 3px;
        font-size: 12px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .doc-item-display {
        padding: 3px 9px;
        background-color: #1a1a1a;
        border-left: 4px solid #933;
        border-radius: 9px;
        font-size: 12px;
    }
    .doc-name {
        color: #ddd;
        font-weight: 500;
    }
    .doc-meta {
        font-size: 10px;
        color: #777;
        margin-top: 2px;
    }
    .delete-btn {
        color: #ff4444;
        cursor: pointer;
        font-size: 16px;
        padding: 0 4px;
        opacity: 0.6;
    }
    .delete-btn:hover {
        opacity: 1;
    }
    .supported-formats {
        font-size: 14px;
        color: #666;
        margin-left: 4px;
        font-style: italic;
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
    if st.button('New Chat', use_container_width=True) and st.session_state['message_history']:
        reset_chat()
        st.rerun()
    
    st.markdown("___")
    
    # Knowledge Base Section
    st.markdown("<div class='section-header'>Knowledge Base</div>", unsafe_allow_html=True)
    
    if st.session_state['uploaded_docs']:
        st.markdown(
            f"<div class='doc-count'>{len(st.session_state['uploaded_docs'])} document(s) available</div>", 
            unsafe_allow_html=True
        )
        
        for idx, doc in enumerate(st.session_state['uploaded_docs']):
            col1, col2 = st.columns([6, 1])
            with col1:
                st.markdown(
                    f"<div class='doc-item-display'>"
                    f"<div class='doc-name'>{doc['filename']}</div>"
                    f"<div class='doc-meta'>{doc['chunks']} chunks • {doc['pages']} pages</div>"
                    f"</div>", 
                    unsafe_allow_html=True
                )
            with col2:
                if st.button("×", key=f"delete_{doc['filename']}_{idx}", help="Remove document"):
                    if delete_document(st.session_state['thread_id'], doc['filename']):
                        st.session_state['uploaded_docs'] = fetch_uploaded_docs(st.session_state['thread_id'])
                        time.sleep(0.5)
                        st.rerun()
    else:
        st.caption("No documents uploaded yet")
    
    # Upload toggle button
    if st.button('Upload Documents' if not st.session_state['show_upload'] else 'Cancel Upload', 
                 use_container_width=True, 
                 type='primary' if not st.session_state['show_upload'] else 'secondary'):
        st.session_state['show_upload'] = not st.session_state['show_upload']
        st.rerun()
    
    # File uploader (only show when toggled)
    if st.session_state['show_upload']:
        uploaded_files = st.file_uploader(
            "Select files",
            type=['pdf', 'txt', 'docx', 'doc', 'md', 'csv', 'ppt', 'pptx'],
            accept_multiple_files=True,
            key="file_uploader",
            label_visibility="collapsed"
        )
        
        st.markdown(
            "<div class='supported-formats'>Supports: PDF, TXT, DOCX, MD, CSV, PPT</div>", 
            unsafe_allow_html=True
        )
        
        if uploaded_files:
            if st.button("Process Files", use_container_width=True):
                with st.spinner("Processing..."):
                    if upload_files(uploaded_files, st.session_state['thread_id']):
                        st.success(f"{len(uploaded_files)} file(s) uploaded successfully")
                        st.session_state['uploaded_docs'] = fetch_uploaded_docs(st.session_state['thread_id'])
                        st.session_state['show_upload'] = False
                        time.sleep(1)
                        st.rerun()
    
    st.markdown("___")
    
    # Chat History Section
    st.markdown("<div class='section-header'>Chat History</div>", unsafe_allow_html=True)
    
    if not st.session_state['chat_threads']:
        st.caption("No previous conversations")
    else:
        for thread_id in st.session_state['chat_threads']:
            if thread_id not in st.session_state['conversation_titles']:
                st.session_state['conversation_titles'][thread_id] = fetch_conversation_title(thread_id)
            
            title = st.session_state['conversation_titles'][thread_id]
            
            if st.button(f"{title}", key=thread_id, use_container_width=True):
                st.session_state['thread_id'] = thread_id
                messages = load_conversation(thread_id)
                st.session_state['message_history'] = messages
                st.session_state['uploaded_docs'] = fetch_uploaded_docs(thread_id)
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
    st.session_state.message_history.append({"role": "user", "content": user_input})
    add_thread_id(st.session_state['thread_id'])
    display_message(user_input, 'user')

    current_thread_id = st.session_state['thread_id']
    words = user_input.split()[:4]
    title = ' '.join(words)
    if len(user_input.split()) > 4:
        title += '...'
    st.session_state['conversation_titles'][current_thread_id] = title

    response_container = st.empty()
    
    full_response = asyncio.run(send_message_websocket(
        user_input, 
        st.session_state['thread_id'],
        response_container
    ))
    
    if full_response:
        st.session_state.message_history.append({"role": "assistant", "content": full_response})