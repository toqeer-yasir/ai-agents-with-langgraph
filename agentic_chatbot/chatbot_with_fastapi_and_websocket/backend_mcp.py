import os
import math
import asyncio
import aiosqlite
from typing import List, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool, BaseTool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_tavily import TavilySearch
from langchain_mcp_adapters.client import MultiServerMCPClient
from typing import TypedDict, Annotated

load_dotenv()

# Global variables
chatbot = None
CHECKPOINTER = None
client = None

# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    global chatbot, CHECKPOINTER, client
    
    # Startup
    print("Initializing chatbot...")
    chatbot, CHECKPOINTER, client = await initialize_chatbot()
    print("Chatbot initialized successfully!")
    
    yield
    
    # Shutdown
    print("Shutting down...")
    if client:
        await client.close()
    print("Shutdown complete")

app = FastAPI(title="Agentic Chatbot API", lifespan=lifespan)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class ChatRequest(BaseModel):
    message: str
    thread_id: str

class ThreadResponse(BaseModel):
    threads: List[str]

class ConversationMessage(BaseModel):
    role: str
    content: str

# Tool definitions
@tool()
def calculator_tool(expression: str) -> str:
    """Evaluate mathematical expressions."""
    try:
        allowed_names = {**math.__dict__}
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"

async def load_mcp_tools() -> list[BaseTool]:
    """Load MCP tools asynchronously."""
    try:
        mcp_client = MultiServerMCPClient({   
            'System info.': {
                'transport': 'stdio',
                'command': 'python',
                'args': ["/home/toqeer-yasir/Documents/repos/ai-agents-with-langgraph/agentic_chatbot/local_mcp_servers/system_info_mcp_server.py"]
            },
            'File System': {
                'transport': 'stdio',
                'command': 'python',
                'args': ["/home/toqeer-yasir/Documents/repos/ai-agents-with-langgraph/agentic_chatbot/local_mcp_servers/filesystem_mcp_server.py"]
            },
            'GitHub': {
                'transport': 'stdio',
                'command': 'python',
                'args': ["/home/toqeer-yasir/Documents/repos/ai-agents-with-langgraph/agentic_chatbot/local_mcp_servers/github_mcp_server.py"]
            },
            'Shell': {
                'transport': 'stdio',
                'command': 'python',
                'args': ["/home/toqeer-yasir/Documents/repos/ai-agents-with-langgraph/agentic_chatbot/local_mcp_servers/shell_mcp_server.py"]
            }
        })
        return await mcp_client.get_tools(), mcp_client
    except Exception as e:
        print(f"Error loading MCP tools: {e}")
        return [], None

# State definition
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

async def initialize_chatbot():
    """Initialize the chatbot with all components"""
    
    # Initialize LLM
    llm = ChatOpenAI(
        model="kwaipilot/kat-coder-pro:free",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        temperature=0.7,
        streaming=True,
        max_retries=2
    )
    
    # Initialize tools
    search_tool = TavilySearch(
        max_results=3,
        include_answer=True,
        search_depth="advanced",
        tavily_api_key=os.getenv("TAVILY_API_KEY")
    )
    
    mcp_tools, mcp_client = await load_mcp_tools()
    tools = [search_tool, calculator_tool, *mcp_tools]
    print("Abailable tools: ", len(tools))
    llm_with_tools = llm.bind_tools(tools=tools)
    
    # Initialize checkpointer
    conn = await aiosqlite.connect(database="database/chatbot.db")
    checkpointer = AsyncSqliteSaver(conn)
    
    # Custom node
    async def chat_node(state: ChatState):
        """Process chat messages using the LLM."""
        messages = state['messages']
        response = await llm_with_tools.ainvoke(messages)
        return {'messages': [response]}
    
    tool_node = ToolNode(tools=tools) if tools else None
    
    # Build graph
    graph = StateGraph(ChatState)
    graph.add_node('chat_node', chat_node)
    graph.add_edge(START, 'chat_node')
    
    if tool_node:
        graph.add_node('tools', tool_node)
        graph.add_conditional_edges('chat_node', tools_condition)
        graph.add_edge('tools', 'chat_node')
    else:
        graph.add_edge('chat_node', END)
    
    compiled_chatbot = graph.compile(checkpointer=checkpointer)
    
    return compiled_chatbot, checkpointer, mcp_client

# WebSocket endpoint for streaming chat
@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            message = data.get("message")
            thread_id = data.get("thread_id")
            
            if not message or not thread_id:
                await websocket.send_json({
                    "type": "error",
                    "content": "Missing message or thread_id"
                })
                continue
            
            config = {'configurable': {'thread_id': thread_id}}
            msg = {'messages': [HumanMessage(content=message)]}
            
            # Stream the response
            try:
                async for message_chunk, metadata in chatbot.astream(
                    msg, 
                    config=config, 
                    stream_mode='messages'
                ):
                    # Handle tool calls
                    if isinstance(message_chunk, AIMessage) and hasattr(message_chunk, 'tool_calls') and message_chunk.tool_calls:
                        tools_used = []
                        for tool_call in message_chunk.tool_calls:
                            tool_name = tool_call["name"].lower()
                            if 'tavily' in tool_name:
                                tools_used.append('search')
                            elif 'calculator' in tool_name or 'math' in tool_name:
                                tools_used.append('calculator')
                            else:
                                tools_used.append(tool_name)
                        
                        await websocket.send_json({
                            "type": "tool_call",
                            "tools": tools_used
                        })
                    
                    # Stream AI response text
                    if isinstance(message_chunk, AIMessage) and message_chunk.content:
                        await websocket.send_json({
                            "type": "content",
                            "content": message_chunk.content
                        })
                
                # Send completion signal
                await websocket.send_json({
                    "type": "complete"
                })
                
            except Exception as e:
                await websocket.send_json({
                    "type": "error",
                    "content": str(e)
                })
    
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")

# REST endpoints
@app.get("/threads", response_model=ThreadResponse)
async def get_threads():
    """Get all conversation threads"""
    all_threads = set()
    async for checkpoint in CHECKPOINTER.alist(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])
    return ThreadResponse(threads=list(all_threads))

@app.get("/conversation/{thread_id}")
async def get_conversation(thread_id: str):
    """Get conversation history for a thread"""
    try:
        state = await chatbot.aget_state(config={'configurable': {'thread_id': thread_id}})
        messages = state.values.get('messages', [])
        
        result = []
        for i, msg in enumerate(messages):
            if isinstance(msg, HumanMessage):
                result.append({"role": "user", "content": msg.content})
                
                # Find the next AI message
                for future_msg in messages[i+1:]:
                    if isinstance(future_msg, HumanMessage):
                        break
                    if isinstance(future_msg, AIMessage) and future_msg.content:
                        result.append({"role": "assistant", "content": future_msg.content})
                        break
        
        return {"messages": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/conversation/{thread_id}/title")
async def get_conversation_title(thread_id: str):
    """Get the title of a conversation based on first message"""
    try:
        state = await chatbot.aget_state(config={'configurable': {'thread_id': thread_id}})
        messages = state.values.get('messages', [])
        
        for msg in messages:
            if isinstance(msg, HumanMessage) and msg.content:
                words = msg.content.split()[:4]
                title = ' '.join(words)
                if len(msg.content.split()) > 4:
                    title += '...'
                return {"title": title if title else "Empty message"}
        
        return {"title": "Empty chat"}
    except Exception as e:
        return {"title": "Empty chat"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "chatbot_initialized": chatbot is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)