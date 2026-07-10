import streamlit as st
import websockets
import json
import asyncio

WS_URL = "ws://localhost:8000/ws/chat"

st.set_page_config(
    page_title="Agentic AI Chatbot",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    "<h3 style='text-align: center;'>Rag Agent</h3>",
    unsafe_allow_html=True,
)

THREAD_ID = "thread_123"
USER_ID = "user_123"

if "pending_interrupt" not in st.session_state:
    st.session_state.pending_interrupt = None  # {"tools": [...]} or None
if "history" not in st.session_state:
    st.session_state.history = []  # [{"role": "user"|"assistant", "content": str}, ...]


async def stream_turn(payload: dict, placeholder):
    """Send one payload (a fresh message OR an approval_response) over a
    fresh websocket connection and stream the reply into `placeholder`.

    Returns (assistant_text, interrupt_content_or_None).
    """
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


def run_turn_and_store(payload: dict):
    """Runs one graph turn, updates chat history / pending-interrupt state,
    then reruns the script so the UI reflects the new state."""
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
        st.session_state.history.append(
            {"role": "assistant", "content": assistant_response}
        )

    st.session_state.pending_interrupt = interrupt_content
    st.rerun()


for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.write(turn["content"])


if st.session_state.pending_interrupt:
    tools = st.session_state.pending_interrupt.get("tools", [])

    with st.chat_message("assistant"):
        st.warning("Approval required before I continue")

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
            run_turn_and_store(payload)

message = st.chat_input(
    placeholder="Type here . . .",
    disabled=bool(st.session_state.pending_interrupt),
)

if message:
    st.session_state.history.append({"role": "user", "content": message})
    with st.chat_message("user"):
        st.write(message)

    payload = {
        "message": message,
        "thread_id": THREAD_ID,
        "user_id": USER_ID,
    }
    run_turn_and_store(payload)