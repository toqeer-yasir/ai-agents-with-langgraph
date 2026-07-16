# import streamlit as st
# import websockets
# import json
# import requests
# import asyncio
# from uuid import uuid4


# BASE_URL = "http://localhost:8000"
# WS_URL = "ws://localhost:8000/ws/chat"

# st.set_page_config(
#     page_title="Agentic AI Chatbot",
#     page_icon="🤖",
#     layout="centered",
#     initial_sidebar_state="collapsed",
# )

# st.markdown(
#     "<h3 style='text-align: center;'>Rag Agent</h3>",
#     unsafe_allow_html=True,
# )



# # session state:
# if "user" not in st.session_state:
#     st.session_state.user = None

# if "pending_interrupt" not in st.session_state:
#     st.session_state.pending_interrupt = None

# if "messages_history" not in st.session_state:
#     st.session_state.messages_history = []

# if "documents" not in st.session_state:
#     st.session_state.documents = {}

# if "threads_and_titles" not in st.session_state:
#     st.session_state.threads_and_titles = {}

# # hepler functions:

# def generate_uuid4():
#     """to generate uuid4"""
#     return str(uuid4())


# def new_chat():
#     st.session_state.messages_history = []


# USER_ID = st.session_state.user
# THREAD_ID = st.session_state.



# def fetch_threads_and_titles() -> list[dict]:
#     try:
#         response = requests.get(f"{BASE_URL}/users/{USER_ID}/chats", timeout=10)
#         response.raise_for_status()
#         return response["chats"]
    
#     except Exception:
#         return []
    

# def fetch_messages() -> list[dict]:
#     try:
#         response = requests.get(f"{BASE_URL}/chats/{thread_id}/messages", timeout=10)
#         response.raise_for_status()
#         return response["messages"]
    
#     except Exception:
#         return []
    

# def push_message(
#     user_id: str,
#     message_id: str,
#     chat_id: str,
#     user_content: str,
#     assistant_content: str,
# ):
#     response = requests.post(
#         f"{BASE_URL}/messages",
#         json={
#             "user_id": user_id,
#             "message_id": message_id,
#             "chat_id": chat_id,
#             "user_content": user_content,
#             "assistant_content": assistant_content,
#         },
#         timeout=10,
#     )

#     response.raise_for_status()
#     return response.json()


# if st.session_state.user is None:

#     st.title("Sign-in/Sign-up")

#     with st.form("registration"):
#         name = st.text_input("Name")
#         email = st.text_input("Email")

#         submitted = st.form_submit_button("Continue")

#     if submitted:

#         if not name or not email:
#             st.warning("Please enter both name and email.")
#             st.stop()

#         try:
#             response = requests.get(
#                 f"{BASE_URL}/users/by-email/{email}",
#                 timeout=10,
#             )
#             response.raise_for_status()

#             data = response.json()

#             if data["user_details"] is not None:
#                 st.session_state.user = data["user_details"]["id"]

#             else:
#                 response = requests.post(
#                     f"{BASE_URL}/users",
#                     json={
#                         "user_id": generate_uuid4(),
#                         "user_name": name,
#                         "user_email": email,
#                     },
#                     timeout=10,
#                 )
#                 response.raise_for_status()

#                 data = response.json()
#                 st.session_state.user = data["user_id"]

#             st.rerun()

#         except Exception as e:
#             st.error(f"Failed to connect to backend: {e}")

#     st.stop()

# async def stream_turn(payload: dict, placeholder):

#     assistant_response = ""

#     async with websockets.connect(WS_URL) as websocket:
#         await websocket.send(json.dumps(payload))

#         async for raw in websocket:
#             data = json.loads(raw)

#             if data["type"] == "content":
#                 assistant_response += data["content"]
#                 placeholder.markdown(assistant_response + "┃")

#             elif data["type"] == "interrupt":
#                 placeholder.markdown(assistant_response or "_Waiting for your approval..._")
#                 return assistant_response, data["content"]

#             elif data["type"] == "completed":
#                 placeholder.markdown(assistant_response)
#                 return assistant_response, None

#             elif data["type"] == "error":
#                 placeholder.markdown(f"⚠️ {data['content']}")
#                 return assistant_response, None

#     return assistant_response, None


# def run_turn_and_store(payload: dict):

#     try:
#         with st.chat_message("assistant"):
#             placeholder = st.empty()
#             placeholder.markdown("_Thinking..._")
#             assistant_response, interrupt_content = asyncio.run(
#                 stream_turn(payload, placeholder)
#             )
#     except Exception as e:
#         st.error(f"Connection error: {e}")
#         return

#     if assistant_response:
#         st.session_state.messages_history.append(
#             {"role": "assistant", "content": assistant_response}
#         )

#     st.session_state.pending_interrupt = interrupt_content
#     st.rerun()


# for turn in st.session_state.messages_history:
#     with st.chat_message(turn["role"]):
#         st.write(turn["content"])


# if st.session_state.pending_interrupt:
#     tools = st.session_state.pending_interrupt.get("tools", [])

#     with st.chat_message("assistant"):
#         st.warning("Approval required before I continue!", width= 270)

#         rejected_by_id = {}
#         for tc in tools:
#             st.write(f"**Tool:** {tc['name']}")
#             st.code(tc["command"], language="bash")
#             for risk, desc in zip(tc["risk_levels"], tc["descriptions"]):
#                 st.write(f"🔴 **{risk.upper()}** — {desc}")

#             approved = st.checkbox(
#                 "Approve this command",
#                 value=False,  # unchecked = rejected by default (fail-safe)
#                 key=f"approve_{tc['id']}",
#             )
#             rejected_by_id[tc["id"]] = not approved

#         if st.button("Submit decision"):
#             rejected_ids = [tid for tid, rejected in rejected_by_id.items() if rejected]
#             payload = {
#                 "type": "approval_response",
#                 "thread_id": THREAD_ID,
#                 "user_id": USER_ID,
#                 "rejected_tool_ids": rejected_ids,
#             }
#             run_turn_and_store(payload)

# message = st.chat_input(
#     placeholder="Type here . . .",
#     disabled=bool(st.session_state.pending_interrupt),
# )

# if message:
#     st.session_state.messages_history.append({"role": "user", "content": message})
#     with st.chat_message("user"):
#         st.write(message)

#     payload = {
#         "message": message,
#         "thread_id": THREAD_ID,
#         "user_id": USER_ID,
#     }
#     run_turn_and_store(payload)




import streamlit as st
import websockets
import json
import requests
import asyncio
from uuid import uuid4


BASE_URL = "http://localhost:8000"
WS_URL = "ws://localhost:8000/ws/chat"

st.set_page_config(
    page_title="Agentic AI Chatbot",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown(
    "<h3 style='text-align: center;'>Rag Agent</h3>",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# session state
# ---------------------------------------------------------------------------
if "user" not in st.session_state:
    st.session_state.user = None

if "thread_id" not in st.session_state:
    st.session_state.thread_id = None

if "pending_interrupt" not in st.session_state:
    st.session_state.pending_interrupt = None

if "messages_history" not in st.session_state:
    st.session_state.messages_history = []

if "documents" not in st.session_state:
    st.session_state.documents = {}

if "threads_and_titles" not in st.session_state:
    st.session_state.threads_and_titles = []


# ---------------------------------------------------------------------------
# helper functions
# ---------------------------------------------------------------------------
def generate_uuid4():
    """to generate uuid4"""
    return str(uuid4())


def new_chat():
    """Start a brand new, empty conversation thread."""
    st.session_state.thread_id = generate_uuid4()
    st.session_state.messages_history = []
    st.session_state.pending_interrupt = None


def fetch_threads_and_titles() -> list[dict]:
    if not st.session_state.user:
        return []
    try:
        response = requests.get(f"{BASE_URL}/users/{st.session_state.user}/chats", timeout=10)
        response.raise_for_status()
        return response.json().get("chats", [])
    except Exception:
        return []


def fetch_messages(thread_id: str) -> list[dict]:
    try:
        response = requests.get(f"{BASE_URL}/chats/{thread_id}/messages", timeout=10)
        response.raise_for_status()
        return response.json().get("messages", [])
    except Exception:
        return []


def push_message(
    user_id: str,
    message_id: str,
    chat_id: str,
    user_content: str,
    assistant_content: str,
):
    try:
        response = requests.post(
            f"{BASE_URL}/messages",
            json={
                "user_id": user_id,
                "message_id": message_id,
                "chat_id": chat_id,
                "user_content": user_content,
                "assistant_content": assistant_content,
            },
            timeout=10,
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.warning(f"Couldn't save this message to history: {e}")
        return None


def load_thread(thread_id: str):
    """Switch to an existing thread and pull its message history."""
    st.session_state.thread_id = thread_id
    st.session_state.pending_interrupt = None

    raw_messages = fetch_messages(thread_id)
    history = [
        {"role": msg["role"], "content": msg["content"]} 
        for msg in raw_messages
        ]
    st.session_state.messages_history = history


# ---------------------------------------------------------------------------
# sign-in / sign-up gate
# ---------------------------------------------------------------------------
if st.session_state.user is None:

    st.title("Sign-in / Sign-up")

    with st.form("registration"):
        name = st.text_input("Name")
        email = st.text_input("Email")
        submitted = st.form_submit_button("Continue")

    if submitted:
        if not name or not email:
            st.warning("Please enter both name and email.")
            st.stop()

        try:
            response = requests.get(f"{BASE_URL}/users/by-email/{email}", timeout=10)
            response.raise_for_status()
            data = response.json()

            if data["user_details"] is not None:
                st.session_state.user = data["user_details"]["id"]
            else:
                response = requests.post(
                    f"{BASE_URL}/users",
                    json={
                        "user_id": generate_uuid4(),
                        "user_name": name,
                        "user_email": email,
                    },
                    timeout=10,
                )
                response.raise_for_status()
                data = response.json()
                st.session_state.user = data["user_id"]

            new_chat()  # start on a fresh thread right after login
            st.rerun()

        except Exception as e:
            st.error(f"Failed to connect to backend: {e}")

    st.stop()


USER_ID = st.session_state.user

# make sure there is always an active thread once logged in
if st.session_state.thread_id is None:
    new_chat()

THREAD_ID = st.session_state.thread_id


# ---------------------------------------------------------------------------
# sidebar: chat history
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Chats")

    if st.button("➕ New chat", use_container_width=False):
        new_chat()
        st.rerun()

    st.session_state.threads_and_titles = fetch_threads_and_titles()

    st.divider()
    for thread in st.session_state.threads_and_titles:
        tid = thread.get("id")
        title = thread.get("title") or "Untitled chat"
        is_active = tid == st.session_state.thread_id
        if st.button(
            ("👉 " if is_active else "") + title,
            key=f"thread_{tid}",
            use_container_width=True,
        ):
            load_thread(tid)
            st.rerun()


# ---------------------------------------------------------------------------
# websocket streaming
# ---------------------------------------------------------------------------
async def stream_turn(payload: dict, placeholder):
    assistant_response = ""

    async with websockets.connect(WS_URL) as websocket:
        await websocket.send(json.dumps(payload))

        async for raw in websocket:
            data = json.loads(raw)

            if data["type"] == "content":
                assistant_response += data["content"]
                placeholder.markdown(assistant_response + "┃")

            elif data["type"] == "interrupt":
                placeholder.markdown(assistant_response or "_Waiting for your approval..._")
                return assistant_response, data["content"]

            elif data["type"] == "completed":
                placeholder.markdown(assistant_response)
                return assistant_response, None

            elif data["type"] == "error":
                placeholder.markdown(f"⚠️ {data['content']}")
                return assistant_response, None

    return assistant_response, None


def run_turn_and_store(payload: dict, user_content: str | None = None):
    try:
        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("_Thinking..._")
            assistant_response, interrupt_content = asyncio.run(
                stream_turn(payload, placeholder)
            )
    except Exception as e:
        st.error(f"Connection error: {e}")
        return

    if assistant_response:
        st.session_state.messages_history.append(
            {"role": "assistant", "content": assistant_response}
        )

    # only persist to the DB once the turn is fully done (no pending approval)
    if not interrupt_content and assistant_response and user_content:
        push_message(
            user_id=USER_ID,
            message_id=generate_uuid4(),
            chat_id=THREAD_ID,
            user_content=user_content,
            assistant_content=assistant_response,
        )

    st.session_state.pending_interrupt = interrupt_content
    st.rerun()


# ---------------------------------------------------------------------------
# render existing conversation
# ---------------------------------------------------------------------------
for turn in st.session_state.messages_history:
    with st.chat_message(turn["role"]):
        st.write(turn["content"])


# ---------------------------------------------------------------------------
# pending tool approval UI
# ---------------------------------------------------------------------------
if st.session_state.pending_interrupt:
    tools = st.session_state.pending_interrupt.get("tools", [])

    with st.chat_message("assistant"):
        st.warning("Approval required before I continue!")

        rejected_by_id = {}
        for tc in tools:
            st.write(f"**Tool:** {tc['name']}")
            st.code(tc["command"], language="bash")
            for risk, desc in zip(tc["risk_levels"], tc["descriptions"]):
                st.write(f"🔴 **{risk.upper()}** — {desc}")

            approved = st.checkbox(
                "Approve this command",
                value=False,  # unchecked = rejected by default (fail-safe)
                key=f"approve_{tc['id']}",
            )
            rejected_by_id[tc["id"]] = not approved

        if st.button("Submit decision"):
            rejected_ids = [tid for tid, rejected in rejected_by_id.items() if rejected]
            payload = {
                "type": "approval_response",
                "thread_id": THREAD_ID,
                "user_id": USER_ID,
                "rejected_tool_ids": rejected_ids,
            }
            # note: user_content=None here, so this resumed turn isn't
            # re-persisted as a new row; the original user message already
            # started the row for this exchange
            run_turn_and_store(payload, user_content=None)


# ---------------------------------------------------------------------------
# chat input
# ---------------------------------------------------------------------------
message = st.chat_input(
    placeholder="Type here . . .",
    disabled=bool(st.session_state.pending_interrupt),
)

if message:
    st.session_state.messages_history.append({"role": "user", "content": message})
    st.session_state.threads_and_titles = fetch_threads_and_titles()
    
    with st.chat_message("user"):
        st.write(message)

    payload = {
        "message": message,
        "thread_id": THREAD_ID,
        "user_id": USER_ID,
    }
    run_turn_and_store(payload, user_content=message)