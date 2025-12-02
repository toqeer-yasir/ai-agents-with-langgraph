import streamlit as st
from chatbot_backend_sync_version import chatbot, retrieve_all_threads
from langchain_core.messages import HumanMessage, AIMessage
import uuid
import markdown

# page configuration
st.set_page_config(
    page_title="Agentic AI Chatbot",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# functions # =====================================================
def generate_thread_id():
    """generate a unique thread ID using UUID."""
    return str(uuid.uuid4())

def reset_chat():
    """reset the chat session with a new thread ID."""
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    st.session_state['message_history'] = []

def add_thread_id(thread_id):
    """add a thread ID to the chat threads list if it doesn't exist."""
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].insert(0, thread_id)


def load_recent_chat(thread_id):
    """realoading recent chats from the threaed id."""
    messages = chatbot.get_state(config={'configurable': {'thread_id': thread_id}}).values['messages']
    
    result = []
    
    for i, msg in enumerate(messages):
        if 'HumanMessage' in str(type(msg)):
            # adding human message:
            result.append(msg)
            
            ai_messages = []
            for future_msg in messages[i+1:]:
                if 'HumanMessage' in str(type(future_msg)):
                    break  # stop at next human message
                if 'AIMessage' in str(type(future_msg)):
                    ai_messages.append(future_msg)
            # adding last ai message:
            if ai_messages:
                result.append(ai_messages[-1])
    
    return result


def display_message(message, role):
    """Display a message with appropriate styling based on role."""
    container_class = "message-container-user" if role == "user" else "message-container-assistant"
    message_class = "user-message" if role == "user" else "assistant-message"
    html_content = markdown.markdown(message)

    html = f"""
    <div class='{container_class}'>
        <div style='display:{"none" if role == "user" else "inline-block"};'>✨</div>
        <div class='{message_class}'>{html_content}</div>
        <div style='display:{"inline-block" if role == "user" else "none"};'>👤</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

# session states # =====================================================
if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

CONFIG = {'configurable': {'thread_id': st.session_state['thread_id']}}

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = retrieve_all_threads()

# custom css # =====================================================
st.markdown("""
<style>
    h2 {
        font-style: italic;
    }
    p {
        color: #FFFFF;
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
        margin-bottom: 5px;
    }
    .thinking-text {
        font-style: italic;
        color: #888;
    }
    .thinking-dots {
        padding-top:8px;
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
    .tool-line {
        color: #888;
        font-size: 12px;
        margin: 1px 0;
        font-family: monospace;
        line-height: 1.2;
        padding-left: 10px;
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

# user input and processing # =====================================================
user_input = st.chat_input("Type here . . .")


if user_input:
    # add user message to history
    st.session_state.message_history.append({"role": "user", "content": user_input})
    add_thread_id(st.session_state['thread_id'])
    display_message(user_input, 'user')

    # container for animation and response
    response_container = st.empty()
    
    msg = {'messages': [HumanMessage(content=user_input)]}
    
    full_response = ""
    tool_emojis = []
    tool_names = []
    
    # show thinking animation
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
    
    tool_called = False
    # stream the response
    for message_chunk, metadata in chatbot.stream(msg, config=CONFIG, stream_mode='messages'):
        # handle tool messages
        if isinstance(message_chunk, AIMessage) and hasattr(message_chunk, 'tool_calls') and message_chunk.tool_calls:
            if tool_emojis:
                tool_emojis.clear()
                tool_names.clear()
            for tool_call in message_chunk.tool_calls:
                tool_name = tool_call["name"].lower()

                if 'duckduckgo' in tool_name or 'search' in tool_name:
                    tool_emojis.append('🔍')
                    tool_names.append('Searching in web')
                elif 'calculator' in tool_name or 'math' in tool_name:
                    tool_emojis.append('🧮')
                    tool_names.append('Calculating')
                elif 'rag' in tool_name or 'document' in tool_name:
                    tool_emojis.append('📚')
                    tool_names.append('Searching in documents')
                else:
                    tool_emojis.append('🔧')
                    tool_names.append('Processing')
                
                # build tool status display
                tool_lines = ""
                for emoji, name in zip(tool_emojis, tool_names):
                    tool_lines += f"<div class='tool-line'>└─{emoji}{name}</div>"
                
                # update container with tool status
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
                        {tool_lines}
                    </div>
                </div>
                """
                response_container.markdown(thinking_with_tools_html, unsafe_allow_html=True)
        
        # stream ai response text
        if isinstance(message_chunk, AIMessage) and message_chunk.content:
            full_response += message_chunk.content
            html_content = markdown.markdown(full_response + '┃')

            
            message_container = f"""
            <div class='message-container-assistant'>
                <div>✨</div>
                <div class='assistant-message'>{html_content}</div>
            </div>
            """
            
            response_container.markdown(message_container, unsafe_allow_html=True)

    # final update to remove cursor:
    if full_response:
        html_content = markdown.markdown(full_response)
        message_container = f"""
        <div class='message-container-assistant'>
            <div>✨</div>
            <div class='assistant-message'>{html_content}</div>
        </div>
        """
        response_container.markdown(message_container, unsafe_allow_html=True)
    
        # adding assistant response to history:
        st.session_state.message_history.append({"role": "assistant", "content": full_response})

