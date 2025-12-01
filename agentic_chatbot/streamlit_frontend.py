import streamlit as st
from chatbot_backend import chatbot, retrieve_all_threads, submit_async_task 
from langchain_core.messages import HumanMessage, AIMessage
import uuid
import queue
import time
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

def get_conversation_title(thread_id):
    """Get the first few words of the first user message for a thread."""
    try:
        messages = chatbot.get_state(config={'configurable': {'thread_id': thread_id}}).values['messages']
        
        # Find the first Human message in the conversation
        for msg in messages:
            if isinstance(msg, HumanMessage) and msg.content:
                # Get first 4 words and add ellipsis
                words = msg.content.split()[:4]
                title = ' '.join(words)
                if len(msg.content.split()) > 4:
                    title += '...'
                return title if title else "Empty message"
        
        # If no Human message found, return a default title
        return "Empty chat"
    except Exception as e:
        return "Empty chat"

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

# session states # =====================================================
if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

CONFIG = {'configurable': {'thread_id': st.session_state['thread_id']}}

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = retrieve_all_threads()

# Store conversation titles to avoid recomputation
if 'conversation_titles' not in st.session_state:
    st.session_state['conversation_titles'] = {}

# custom css # =====================================================
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

# sidebar # =====================================================
with st.sidebar:
    if st.button('Start New Chat +') and st.session_state['message_history']:
        reset_chat()
    
    st.title('Recent Chats')
    st.markdown('___')
    
    for thread_id in st.session_state['chat_threads']:
        # Get or compute conversation title
        if thread_id not in st.session_state['conversation_titles']:
            st.session_state['conversation_titles'][thread_id] = get_conversation_title(thread_id)
        
        title = st.session_state['conversation_titles'][thread_id]
        
        if st.button(f"{title}", key=thread_id):
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

    # Update conversation title for current thread with the first user message
    current_thread_id = st.session_state['thread_id']
    words = user_input.split()[:4]
    title = ' '.join(words)
    if len(user_input.split()) > 4:
        title += '...'
    st.session_state['conversation_titles'][current_thread_id] = title

    # container for animation and response
    response_container = st.empty()
    
    msg = {'messages': [HumanMessage(content=user_input)]}
    
    full_response = ""
    tools_set = set()
    
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
    
    event_queue = queue.Queue()

    async def run_async_stream():
        try:
            async for message_chunk, metadata in chatbot.astream(
                msg, 
                config=CONFIG, 
                stream_mode='messages'
            ):
                event_queue.put((message_chunk, metadata))
        except Exception as exc:
            event_queue.put(("error", exc))
        finally:
            event_queue.put(None)

    # async stream
    submit_async_task(run_async_stream())
    
    # Process events from the queue
    while True:
        item = event_queue.get()
        if item is None:
            break
            
        if item[0] == "error":
            raise item[1]
            
        message_chunk, metadata = item
        
        # handle tool messages 
        if isinstance(message_chunk, AIMessage) and hasattr(message_chunk, 'tool_calls') and message_chunk.tool_calls:
            if tools_set:
                tools_set.clear()
            for tool_call in message_chunk.tool_calls:
                tool_name = tool_call["name"].lower()

                if 'tavily' in tool_name:
                    tools_set.add(' 🔍Search')
                elif 'calculator' in tool_name or 'math' in tool_name:
                    tools_set.add(' 🧮Calculator')
                else:
                    tools_set.add(f' 🔧{tool_name}')
                
                # build tool status display
                tools= f"<div class='tools'>Using:</div>"
                tool_lines = ""
                for tool in tools_set:
                    tool_lines += f"<div class='tool-line'>└─{tool} tool</div>"
                
                
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
                        {tools}
                        {tool_lines}
                    </div>
                </div>
                """
                response_container.markdown(thinking_with_tools_html, unsafe_allow_html=True)
        
        # stream ai response text
        if isinstance(message_chunk, AIMessage) and message_chunk.content:
            new_content = message_chunk.content
            if new_content and not new_content.isspace():
                words = new_content.split()
                for word in words:
                    full_response += word + " "
                    
                    # Convert current content to HTML for proper markdown rendering
                    html_content = markdown.markdown(full_response + "┃")
                    
                    message_container = f"""
                    <div class='message-container-assistant'>
                        <div>✨</div>
                        <div class='assistant-message'>{html_content}</div>
                    </div>
                    """
                    response_container.markdown(message_container, unsafe_allow_html=True)
                    time.sleep(0.01)
    
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