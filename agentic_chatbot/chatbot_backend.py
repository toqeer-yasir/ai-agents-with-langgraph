# Standard library imports
import os
import math
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

# Third-party imports
from typing import TypedDict, Annotated
from dotenv import load_dotenv

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool, BaseTool
from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition

# imports for rag
import tempfile
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from typing import Dict, Optional, Any

# Tool-specific imports
from langchain_tavily import TavilySearch
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()


# initializing models-----------------------
llm = ChatOpenAI(
    model="x-ai/grok-4.1-fast:free",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0.6,
    streaming=True,
    max_retries=2
)

embeddings_model = ChatOpenAI(
    model="qwen/qwen3-embedding-0.6b",  # Or check OpenRouter docs for best model
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
)

# -------------------------------------------


# pdf retriever threads and functions---------
_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}


def _get_retriever(thread_id: Optional[str]):
    """to fetch the retriever for thread if available."""
    if thread_id and thread_id in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[thread_id]
    return None


def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    """to build a Faiss retriever from the uploaded pdf for the thread.
    returns a symmary dict that can be surfaced in the UI.
    """
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size= 1000, chunk_overlap= 200, separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(docs)

        vector_store = FAISS.from_documents(chunks, embeddings_model)
        retriever = vector_store.as_retriever(
            search_type= "similarity", search_kwargs={'k':3}
        )

        _THREAD_RETRIEVERS[str(thread_id)] = retriever
        _THREAD_METADATA[str(thread_id)] = {
            'filename': filename or os.path.basename(temp_path),
            'documents': len(docs),
            'chunks': len(chunks)
        }

        return {
            'filename': filename or os.path.basename(temp_path),
            'documents': len(docs),
            'chunks': len(chunks),
        }
    
    finally:
        # FAISS store keeps copies of the text, so the temp file is safe to remove.
        try:
            os.remove(temp_path)
        except OSError:
            pass
# --------------------------------------------



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


# defining @tools and related functions--------------------
@tool()
def calculator_tool(expression: str) -> str:
    """Evaluate mathematical expressions."""
    try:
        allowed_names = {**math.__dict__}
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"
    

@tool()
def rag_tool(query: str, thread_id: Optional[str] = None) -> dict:
    """retrieve relevant info from the uploaded pdf for chat thread."""
    retriever = _get_retriever(thread_id)
    if retriever is None:
         return {
             'error': "No document uploaded for this chat. Upload pdf first.",
             'query': query
         }    
    result = retriever.invoke(query)
    context = [doc.page_content for doc in result]
    metadata = [doc.page_metadata for doc in result]
    return {
        'query': query,
        'context': context,
        'metadata': metadata,
        'source_file': _THREAD_METADATA.get(str(thread_id), {}).get('filename')
    }


search_tool = TavilySearch(
    max_results=3,
    include_answer=True,
    search_depth="advanced",
    tavily_api_key=os.getenv("TAVILY_API_KEY")
)


def load_mcp_tools() -> list[BaseTool]:
    """Load MCP tools asynchronously."""
    try:
        return client.get_tools()
    except Exception:
        return f"Error: Check servers config."


mcp_tools = load_mcp_tools()

tools = [search_tool, calculator_tool, *mcp_tools]
llm_with_tools = llm.bind_tools(tools=tools)



# State definition
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState, config=None):
    """LLM node that answers or requests tool call."""
    thread_id = None
    if config and isinstance(config, dict):
        thread_id = config.get("configurable", {}).get("thread_id")

    system_message = SystemMessage(
        content=(
            "You are a helpful assistant. For questions about the uploaded PDF, call "
            "the `rag_tool` and include the thread_id "
            f"`{thread_id}`. You can also use the web search, and "
            "calculator and other tools when helpful. If no document is available, ask the user "
            "to upload a PDF."
        )
    )

    messages = [system_message, *state["messages"]]
    response = llm_with_tools.invoke(messages, config=config)
    return {"messages": [response]}



tool_node = ToolNode(tools)


CONN = sqlite3.connect(database="database/chatbot.db", check_same_thread=False)
CHECKPOINTER = SqliteSaver(conn=CONN)


# Graph definition
graph = StateGraph(ChatState)


graph.add_node('chat_node', chat_node)
graph.add_node('tools', tool_node)

graph.add_edge(START, 'chat_node')
graph.add_conditional_edges('chat_node', tools_condition)
graph.add_edge('tools', 'chat_node')

chatbot = graph.compile(checkpointer=CHECKPOINTER)


def retrieve_all_threads():
    all_threads = set()
    for checkpoint in CHECKPOINTER.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)

def thread_has_document(thread_id: str) -> bool:
    return str(thread_id) in _THREAD_RETRIEVERS


def thread_documents_metadata(thread_id: str) -> dict:
    return _THREAD_METADATA.get(str(thread_id), {})
