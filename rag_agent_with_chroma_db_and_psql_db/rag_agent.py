import os
from contextlib import asynccontextmanager
from typing import Annotated

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langchain_openrouter import ChatOpenRouter

from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from psycopg_pool import AsyncConnectionPool

from typing_extensions import TypedDict

from tools import get_tools



load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]



def create_chat_node(model_with_tools):

    async def chat_node(state: ChatState):

        response = await model_with_tools.ainvoke(
            state["messages"]
        )

        return {
            "messages": [response]
        }

    return chat_node



def build_graph(model_with_tools, tools, checkpointer):

    builder = StateGraph(ChatState)

    chat_node = create_chat_node(model_with_tools)

    builder.add_node("chatnode", chat_node)
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "chatnode")

    builder.add_conditional_edges(
        "chatnode",
        tools_condition,
    )

    builder.add_edge(
        "tools",
        "chatnode",
    )

    return builder.compile(
        checkpointer=checkpointer
    )



@asynccontextmanager
async def lifespan(app: FastAPI):

    print("Starting server...")

    pool = AsyncConnectionPool(
        DATABASE_URL,
        min_size=4,
        max_size=10,
        open=False
    )

    await pool.open()

    checkpointer = AsyncPostgresSaver(pool)

    await checkpointer.setup()

    model = ChatOpenRouter(
        model="openrouter/free",
        api_key=OPENROUTER_API_KEY,
        streaming= True,
        temperature=0.7
    )

    tools, mcp_client = await get_tools()
    print(f"Tools Received: {len(tools)}")

    model_with_tools = model.bind_tools(tools)

    graph = build_graph(
        model_with_tools=model_with_tools,
        tools=tools,
        checkpointer=checkpointer,
    )

    app.state.pool = pool
    app.state.graph = graph
    app.state.model = model
    app.state.tools = tools
    app.state.mcp_client = mcp_client

    yield

    print("Stopping server...")

    await pool.close()

    if hasattr(app.state, "mcp_client"):
        await app.state.mcp_client.aclose()



app = FastAPI(
    title="Agentic RAG Backend",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            data = await websocket.receive_json()
            message = data.get("message")
            thread_id = data.get("thread_id")
            user_id = data.get("user_id")
            
            if not message or not thread_id:
                await websocket.send_json({
                    "type": "error",
                    "content": "Missing message or thread_id"
                })
                continue
            
            msg = {'messages': [HumanMessage(content=message)]}
            config = {"configurable": {'thread_id': thread_id}}
            context={"user_id": user_id}

            try:
                async for message_chunk, _ in websocket.app.state.graph.astream(
                    msg,
                    config= config,
                    context= context,
                    stream_mode= ["messages", "updates"]
                ):

                    if isinstance(message_chunk, AIMessage) and message_chunk.content:
                        await websocket.send_json({
                            "type": "content",
                            "content": message_chunk.content
                        })

                await websocket.send_json({
                    "type": "completed"
                })

            except Exception as e:
                await websocket.send_json({
                    "type": "error",
                    "content": str(e)
                })
    
    except WebSocketDisconnect:
        print("Client Disconnected")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)