import streamlit as st
from chatbot_backend import chatbot, retrive_threads
from langchain_core.messages import HumanMessage
import uuid

st.set_page_config(
    page_title="Agentic AI Chatbot",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# *******************************************************Utility functions
def generate_thread_id():
    return uuid.uuid4()

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    st.session_state['message_history'] = []

def add_thread_id(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)

def load_recent_chat(thread_id):
    return chatbot.get_state(config={'configurable': {'thread_id': thread_id}}).values['messages']

# ***********************************************************Session state
if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

config = {'configurable': {'thread_id': st.session_state['thread_id']}}

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = retrive_threads()

# **************************************************************Sidebar
with st.sidebar:
    if st.button('Start New Chat + ') and st.session_state['message_history']:
        reset_chat()
    st.title('Recent Chats')
    st.markdown('___')
    for thread_id in st.session_state.chat_threads[::-1]:
        if st.button(f"{thread_id}"):
            st.session_state['thread_id'] = thread_id
            messages = load_recent_chat(thread_id)
            temp_messages = []
            for message in messages:
                role = 'user' if isinstance(message, HumanMessage) else 'assistant'
                temp_messages.append({"role": role, "content": message.content})
            st.session_state['message_history'] = temp_messages

# **************************************************************Main page
st.markdown(f"""
            <div class='title'>
            <h2>Agentic AI Chatbot 🤖</h2>
            <p>'Toqeer's personal assistant'</p>
            </div>
    """, unsafe_allow_html=True)

# Custom CSS
st.markdown("""
<style>
    h2 {
        font-style:italic;
    }
    p {
        color:gray;
        font-style:italic;        
    }
    .title{
        display: flex;
        justify-content: center;
        align-item:center;
        flex-direction:column;
        text-align:center;
    }
    .user-message {
        padding: 9px 16px 9px 12px;
        border-radius: 18px 0 18px 18px;
        background-color: #262730;
        color: white;
        margin: 8px 0 15px;
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
        margin-bottom:50px;
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
    .typing-dots {
        padding: 9px 16px 9px 12px;
        display: inline-flex;
        align-items: center;
    }
    .dot {
        width: 6px;
        height: 6px;
        background-color: #888;
        border-radius: 50%;
        margin: 0 2px;
        animation: typing 1.4s infinite ease-in-out;
    }
    .dot:nth-child(1) {
        animation-delay: -0.32s;
    }
    .dot:nth-child(2) {
        animation-delay: -0.16s;
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
# Functin to display message
def display_message(message, role):
    container_class = "message-container-user" if role == "user" else "message-container-assistant"
    message_class = "user-message" if role == "user" else "assistant-message"
    
    html = f"""
    <div class='{container_class}'>
        <div style='display:{"none" if role == "user" else "inline-block"};'>✨</div>
        <div class='{message_class}'>{message}</div>
        <div style='display:{"inline-block" if role == "user" else "none"};'>👤</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def show_typing_dots():
    dots_html = """
    <div class='message-container-assistant'>
        <div>✨</div>
        <div class='assistant-message'>
            <div class='typing-dots'>
                <div class='dot'></div>
                <div class='dot'></div>
                <div class='dot'></div>
            </div>
        </div>
    </div>
    """
    return st.markdown(dots_html, unsafe_allow_html=True)

for message in st.session_state.message_history:
    display_message(message['content'], message['role'])

user_input = st.chat_input("Type here . . .")

if user_input:
    st.session_state.message_history.append({"role": "user", "content": user_input})
    add_thread_id(st.session_state['thread_id'])
    display_message(user_input, 'user')

    typing_placeholder = st.empty()
    with typing_placeholder.container():
        show_typing_dots()
    
    msg = {'messages': [HumanMessage(content=user_input)]}
    
    assistant_container = st.empty()
    full_response = ""
    message_container = """
    <div class='message-container-assistant'>
        <div>✨</div>
        <div class='assistant-message'>
    """
    # Streaming functinality
    for message_chunk, metadata in chatbot.stream(msg, config=config, stream_mode='messages'):
        typing_placeholder.empty()
        if hasattr(message_chunk, 'content'):
            full_response += message_chunk.content
            assistant_container.markdown(
                f"{message_container}{full_response}</div></div>", 
                unsafe_allow_html=True
            )
    st.session_state.message_history.append({"role": "assistant", "content": full_response})
