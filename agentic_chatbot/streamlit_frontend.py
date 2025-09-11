# import streamlit as st
# from chatbot_backend import chatbot, retrive_threads
# from langchain_core.messages import HumanMessage
# import uuid

# st.set_page_config(
#     page_title="Agentic AI Chatbot",
#     page_icon="🤖",
#     layout="centered",
#     initial_sidebar_state="collapsed"
# )

# # functions # =====================================================
# def generate_thread_id():
#     return str(uuid.uuid4())

# def reset_chat():
#     thread_id = generate_thread_id()
#     st.session_state['thread_id'] = thread_id
#     st.session_state['message_history'] = []

# def add_thread_id(thread_id):
#     if thread_id not in st.session_state['chat_threads']:
#         st.session_state['chat_threads'].insert(0, thread_id)

# def load_recent_chat(thread_id):
#     state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
#     return state.values['messages']

# def display_message(message, role):
#     container_class = "message-container-user" if role == "user" else "message-container-assistant"
#     message_class = "user-message" if role == "user" else "assistant-message"
    
#     html = f"""
#     <div class='{container_class}'>
#         <div style='display:{"none" if role == "user" else "inline-block"};'>✨</div>
#         <div class='{message_class}'>{message}</div>
#         <div style='display:{"inline-block" if role == "user" else "none"};'>👤</div>
#     </div>
#     """
#     st.markdown(html, unsafe_allow_html=True)

# def show_typing_dots():
#     """Display typing indicator animation."""
#     dots_html = """
#     <div class='message-container-assistant'>
#         <div>✨</div>
#         <div class='assistant-message'>
#             <div class='typing-dots'>
#                 <div class='dot'></div>
#                 <div class='dot'></div>
#                 <div class='dot'></div>
#             </div>
#         </div>
#     </div>
#     """
#     return st.markdown(dots_html, unsafe_allow_html=True)

# # session states # =====================================================
# if 'thread_id' not in st.session_state:
#     st.session_state['thread_id'] = generate_thread_id()

# CONFIG = {'configurable': {'thread_id': st.session_state['thread_id']}}

# if 'message_history' not in st.session_state:
#     st.session_state['message_history'] = []

# if 'chat_threads' not in st.session_state:
#     st.session_state['chat_threads'] = retrive_threads()

# # sidebar # =====================================================
# with st.sidebar:
#     if st.button('Start New Chat +') and st.session_state['message_history']:
#         reset_chat()
    
#     st.title('Recent Chats')
#     st.markdown('___')
    
#     for thread_id in st.session_state['chat_threads']:
#         if st.button(f"{thread_id}"):
#             st.session_state['thread_id'] = thread_id
#             messages = load_recent_chat(thread_id)
#             temp_messages = []
            
#             for message in messages:
#                 role = 'user' if isinstance(message, HumanMessage) else 'assistant'
#                 temp_messages.append({"role": role, "content": message.content})
            
#             st.session_state['message_history'] = temp_messages

# # main page heading # =====================================================
# st.markdown("""
#     <div class='title'>
#         <h2>Agentic AI Chatbot 🤖</h2>
#         <p>'Toqeer's personal assistant'</p>
#     </div>
# """, unsafe_allow_html=True)

# # custom css # =====================================================
# st.markdown("""
# <style>
#     h2 {
#         font-style: italic;
#     }
#     p {
#         color: gray;
#         font-style: italic;        
#     }
#     .title {
#         display: flex;
#         justify-content: center;
#         align-items: center;
#         flex-direction: column;
#         text-align: center;
#     }
#     .user-message {
#         padding: 9px 16px 9px 12px;
#         border-radius: 18px 0 18px 18px;
#         background-color: #262730;
#         color: white;
#         margin: 13px 0 15px;
#         display: inline-block;
#         max-width: 100%;
#         text-align: left;
#         word-wrap: break-word;
#     }
#     .assistant-message {
#         padding: 9px 16px 9px 12px;
#         border-radius: 0 18px 18px 18px;
#         background-color: black;
#         color: white;
#         margin: 8px 0;
#         margin-bottom: 50px;
#         display: inline-block;
#         max-width: 100%;
#         text-align: left;
#         word-wrap: break-word;
#     }
#     .message-container-user {
#         display: flex;
#         justify-content: flex-end;
#         width: 100%;
#     }
#     .message-container-assistant {
#         display: flex;
#         justify-content: flex-start;
#         width: 100%;
#     }
#     .typing-dots {
#         padding: 9px 16px 9px 12px;
#         display: inline-flex;
#         align-items: center;
#     }
#      .dot {
#         width: 6px;
#         height: 6px;
#         background-color: #888;
#         border-radius: 50%;
#         margin: 0 2px;
#         animation: typing 1.4s infinite ease-in-out;
#     }
#     .dot:nth-child(1) {
#         animation-delay: -0.32s;
#     }
#     .dot:nth-child(2) {
#         animation-delay: -0.16s;
#     }
#     @keyframes typing {
#         0%, 80%, 100% {
#             transform: scale(0);
#             opacity: 0.5;
#         }
#         40% {
#             transform: scale(1);
#             opacity: 1;
#         }
#     }
# </style>
# """, unsafe_allow_html=True)

# # display recent messages # =====================================================
# for message in st.session_state.message_history:
#     display_message(message['content'], message['role'])

# # user input and processing # =====================================================
# user_input = st.chat_input("Type here . . .")

# if user_input:
#     # add user message to history
#     st.session_state.message_history.append({"role": "user", "content": user_input})
#     add_thread_id(st.session_state['thread_id'])
#     display_message(user_input, 'user')

#     # Show typing indicator
#     typing_placeholder = st.empty()
#     with typing_placeholder.container():
#         show_typing_dots()
    
#     msg = {'messages': [HumanMessage(content=user_input)]}
    
#     assistant_container = st.empty()
#     full_response = ""
#     message_container = """
#     <div class='message-container-assistant'>
#         <div>✨</div>
#         <div class='assistant-message'>
#     """
    
#     # streaming functinality
#     for message_chunk, metadata in chatbot.stream(msg, config=CONFIG, stream_mode='messages'):
#             typing_placeholder.empty()
            
#             if hasattr(message_chunk, 'content') and message_chunk.content:
#                 if hasattr(message_chunk, 'type') and 'AIMessageChunk' in str(type(message_chunk)):
#                     full_response += message_chunk.content
#                     assistant_container.markdown(
#                         f"{message_container}{full_response}</div></div>", 
#                         unsafe_allow_html=True
#                     )
    
#     # add assistant response to history
#     st.session_state.message_history.append({"role": "assistant", "content": full_response})


import streamlit as st
from chatbot_backend import chatbot, retrive_threads
from langchain_core.messages import HumanMessage
import uuid

# page configuration
st.set_page_config(
    page_title="Agentic AI Chatbot",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# functions # =====================================================
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

def load_recent_chat(thread_id):
    """Load messages from a specific thread."""
    state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
    return state.values['messages']

def display_message(message, role):
    """Display a message with appropriate styling based on role."""
    container_class = "message-container-user" if role == "user" else "message-container-assistant"
    message_class = "user-message" if role == "user" else "assistant-message"
    
    html = f"""
    <div class='{container_class}'>
        <div style='display:{"none" if role == "user" else "inline-block"};'>✨</div>
        <div class='{message_class}'>{message}</div>
        <div style='display:{"inline-block" if role == "user" else "none"};'>👤</div>
        <button class='copy-btn' style='display:{"none" if role == "user" else "inline-block"};'>
        <svg viewBox="0 0 24 24">
        <path d="M16 1H4c-1.1 0-2 .9-2 2v14h2V3h12V1zm3 4H8c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h11c1.1 0 2-.9 2-2V7c0-1.1-.9-2-2-2zm0 16H8V7h11v14z"/>
        </svg>
        </button>
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

# session states # =====================================================
if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

CONFIG = {'configurable': {'thread_id': st.session_state['thread_id']},
          'metadeta': {'thread_id': st.session_state['thread_id']},
          'run_name': 'chat_turn'}

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = retrive_threads()

# custom css # =====================================================
st.markdown("""
<style>
    h2 {
        font-style: italic;
    }
    p {
        color: gray;
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
        padding: 9px 16px 9px 12px;
        border-radius: 18px 0 18px 18px;
        background-color: #262730;
        color: white;
        margin: 18px 0 2px;
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
        margin: 0 0 0 22px;
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
        flex-direction: column;
        justify-content: flex-end;
        width: fit-content;
    }
    .copy-btn {
        width: 2rem;
        height: 2rem;
        margin: 1px 0 0 34px;
        border-radius: 100%;
        border: none;
        background-color: black;
    }
    svg {
        width: 16px;
        height: 16px;
        fill: currentColor;
    }
    .typing-dots {
        padding: 9px 16px 9px 12px;
        display: inline-flex;
        align-items: center;
        width: fit-content;
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

# sidebar # =====================================================
with st.sidebar:
    if st.button('Start New Chat +') and st.session_state['message_history']:
        reset_chat()
    
    st.title('Recent Chats')
    st.markdown('___')
    
    for thread_id in st.session_state['chat_threads']:
        if st.button(f"{thread_id}"):
            st.session_state['thread_id'] = thread_id
            messages = load_recent_chat(thread_id)
            temp_messages = []
            
            for message in messages:
                role = 'user' if isinstance(message, HumanMessage) else 'assistant'
                temp_messages.append({"role": role, "content": message.content})
            
            st.session_state['message_history'] = temp_messages

# main page # =====================================================
st.markdown("""
    <div class='title'>
        <h2>Agentic AI Chatbot 🤖</h2>
        <p>'Toqeer's personal assistant'</p>
    </div>
""", unsafe_allow_html=True)

# display message history
for message in st.session_state.message_history:
    display_message(message['content'], message['role'])

user_input = st.chat_input("Type here . . .")

if user_input:
    st.session_state.message_history.append({"role": "user", "content": user_input})
    add_thread_id(st.session_state['thread_id'])
    display_message(user_input, 'user')

    # typing dots animation
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
    

    # streaming functinality
    for message_chunk, metadata in chatbot.stream(msg, config=CONFIG, stream_mode='messages'):
            typing_placeholder.empty()
            
            if hasattr(message_chunk, 'content') and message_chunk.content:
                if hasattr(message_chunk, 'type') and 'AIMessageChunk' in str(type(message_chunk)):
                    full_response += message_chunk.content
                    assistant_container.markdown(
                        f"{message_container}{full_response + ' |'}</div></div>", 
                        unsafe_allow_html=True
                    )
    
    # add assistant response to history
    st.session_state.message_history.append({"role": "assistant", "content": full_response})