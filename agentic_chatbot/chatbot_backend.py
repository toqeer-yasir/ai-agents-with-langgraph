# Standard library imports
import os
import math
import asyncio
import threading
import aiosqlite
# import requests

# Third-party imports
from typing import TypedDict, Annotated
from dotenv import load_dotenv

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool, BaseTool
from langchain_core.messages import BaseMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

# Tool-specific imports
from langchain_tavily import TavilySearch
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

# Async loop setup
_ASYNC_LOOP = asyncio.new_event_loop()
_ASYNC_THREAD = threading.Thread(target=_ASYNC_LOOP.run_forever, daemon=True)
_ASYNC_THREAD.start()


def _submit_async(coro):
    """Submit a coroutine to the async event loop."""
    return asyncio.run_coroutine_threadsafe(coro, _ASYNC_LOOP)


def run_async(coro):
    """Run a coroutine and return the result."""
    return _submit_async(coro).result()


def submit_async_task(coro):
    """Schedule a coroutine on the backend event loop."""
    return _submit_async(coro)


llm = ChatOpenAI(
    model="x-ai/grok-4.1-fast:free",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0.6,
    streaming=True,
    max_retries=2,
    timeout=30.0
)

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
    

search_tool = TavilySearch(
    max_results=3,
    include_answer=True,
    search_depth="advanced",
    tavily_api_key=os.getenv("TAVILY_API_KEY")
)


client = MultiServerMCPClient(
    {   
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

    }
)


def load_mcp_tools() -> list[BaseTool]:
    """Load MCP tools asynchronously."""
    try:
        return run_async(client.get_tools())
    except Exception:
        return f"Error: Check servers config."


mcp_tools = load_mcp_tools()

tools = [search_tool, calculator_tool, *mcp_tools]
llm_with_tools = llm.bind_tools(tools=tools)



# State definition
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# Custom node
async def chat_node(state: ChatState):
    """Process chat messages using the LLM."""
    messages = state['messages']
    response = await llm_with_tools.ainvoke(messages)
    return {'messages': [response]}


tool_node = ToolNode(tools=tools) if tools else None


# Checkpointer initialization
async def _init_checkpointer():
    """Initialize the SQLite checkpointer."""
    conn = await aiosqlite.connect(database="database/chatbot.db")
    return AsyncSqliteSaver(conn)


CHECKPOINTER = run_async(_init_checkpointer())


# Graph definition
graph = StateGraph(ChatState)

graph.add_node('chat_node', chat_node)
graph.add_edge(START, 'chat_node')

if tool_node:
    graph.add_node('tools', tool_node)
    graph.add_conditional_edges('chat_node', tools_condition)
    graph.add_edge('tools', 'chat_node')
else:
    graph.add_edge('chat_node', END)

chatbot = graph.compile(checkpointer=CHECKPOINTER)


# Thread management
async def _alist_threads():
    """List all threads from the checkpointer."""
    all_threads = set()
    async for checkpoint in CHECKPOINTER.alist(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])
    return list(all_threads)


def retrieve_all_threads():
    """Retrieve all threads synchronously."""
    return run_async(_alist_threads())