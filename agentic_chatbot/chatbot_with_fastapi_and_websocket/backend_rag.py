import os
import math
import asyncio
import aiosqlite
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager
import tempfile
import numpy as np
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Document processing
from langchain_community.document_loaders import (
    PyPDFLoader, 
    TextLoader, 
    Docx2txtLoader,
    UnstructuredMarkdownLoader,
    CSVLoader,
    UnstructuredPowerPointLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool, BaseTool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_tavily import TavilySearch
from langchain_mcp_adapters.client import MultiServerMCPClient
from typing import TypedDict, Annotated

# Sentence transformers for embeddings
from sentence_transformers import SentenceTransformer

load_dotenv()

# ==================== RAG Components ====================

class FastRAG:
    """Fast RAG implementation using sentence-transformers"""
    
    def __init__(self, model_name: str = 'BAAI/bge-small-en-v1.5'):
        self.model = SentenceTransformer(model_name)
        self.documents = []
        self.embeddings = None
        self.metadata = []
    
    def add_documents(self, texts: List[str], metadatas: List[Dict] = None):
        """Add documents to the vector store"""
        self.documents.extend(texts)
        
        if metadatas:
            self.metadata.extend(metadatas)
        else:
            self.metadata.extend([{}] * len(texts))
        
        # Compute embeddings
        new_embeddings = self.model.encode(texts, convert_to_numpy=True)
        
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
    
    def similarity_search(self, query: str, k: int = 4) -> List[Dict]:
        """Search for similar documents"""
        if not self.documents:
            return []
        
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        
        # Compute cosine similarity
        similarities = np.dot(self.embeddings, query_embedding.T).flatten()
        similarities = similarities / (np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding))
        
        # Get top k
        top_k_idx = np.argsort(similarities)[-k:][::-1]
        
        results = []
        for idx in top_k_idx:
            results.append({
                'content': self.documents[idx],
                'metadata': self.metadata[idx],
                'score': float(similarities[idx])
            })
        
        return results

# Thread-specific RAG stores
_THREAD_RAGS: Dict[str, FastRAG] = {}
_THREAD_DOCS_METADATA: Dict[str, List[Dict]] = {}

def get_rag_store(thread_id: str) -> Optional[FastRAG]:
    """Get RAG store for a thread"""
    return _THREAD_RAGS.get(thread_id)

async def ingest_file(file_bytes: bytes, filename: str, thread_id: str) -> Dict:
    """Process and ingest a file into the RAG store"""
    
    # Create temp file
    suffix = Path(filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name
    
    try:
        # Load document based on file type
        if suffix == '.pdf':
            loader = PyPDFLoader(temp_path)
        elif suffix == '.txt':
            loader = TextLoader(temp_path)
        elif suffix in ['.docx', '.doc']:
            loader = Docx2txtLoader(temp_path)
        elif suffix == '.md':
            loader = UnstructuredMarkdownLoader(temp_path)
        elif suffix == '.csv':
            loader = CSVLoader(temp_path)
        elif suffix in ['.ppt', '.pptx']:
            loader = UnstructuredPowerPointLoader(temp_path)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")
        
        docs = loader.load()
        
        # Split documents
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(docs)
        
        # Get or create RAG store for thread
        if thread_id not in _THREAD_RAGS:
            _THREAD_RAGS[thread_id] = FastRAG()
            _THREAD_DOCS_METADATA[thread_id] = []
        
        rag_store = _THREAD_RAGS[thread_id]
        
        # Add to RAG store
        texts = [chunk.page_content for chunk in chunks]
        metadatas = [{'filename': filename, **chunk.metadata} for chunk in chunks]
        
        rag_store.add_documents(texts, metadatas)
        
        # Store file metadata
        file_metadata = {
            'filename': filename,
            'pages': len(docs),
            'chunks': len(chunks),
            'file_type': suffix
        }
        _THREAD_DOCS_METADATA[thread_id].append(file_metadata)
        
        return file_metadata
        
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass

# ==================== Global Variables ====================
chatbot = None
CHECKPOINTER = None
client = None

# ==================== Lifespan ====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    global chatbot, CHECKPOINTER, client
    
    print("🚀 Initializing chatbot...")
    chatbot, CHECKPOINTER, client = await initialize_chatbot()
    print("✅ Chatbot initialized successfully!")
    
    yield
    
    print("🔄 Shutting down...")
    if client:
        await client.close()
    print("👋 Shutdown complete")

app = FastAPI(title="Agentic Chatbot API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== Pydantic Models ====================
class ChatRequest(BaseModel):
    message: str
    thread_id: str

class ThreadResponse(BaseModel):
    threads: List[str]

class ConversationMessage(BaseModel):
    role: str
    content: str

# ==================== Tools ====================
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
def retriever_tool(query: str, thread_id: str) -> Dict:
    """
    Retrieve relevant information from uploaded documents.
    Use this when user asks questions about their files.
    """
    
    rag_store = get_rag_store(thread_id)
    
    if not rag_store:
        return {
            'error': 'No documents available',
            'query': query
        }
    
    results = rag_store.similarity_search(query, k=4)
    
    if not results:
        return {
            'error': 'No relevant information found',
            'query': query
        }
    
    context = [r['content'] for r in results]
    metadata = [r['metadata'] for r in results]
    
    return {
        'query': query,
        'context': context,
        'metadata': metadata,
        'num_results': len(results)
    }

async def load_mcp_tools() -> tuple[list[BaseTool], Any]:
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

# ==================== State ====================
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# ==================== Initialize Chatbot ====================
async def initialize_chatbot():
    """Initialize the chatbot with all components"""
    
    llm = ChatOpenAI(
        model="kwaipilot/kat-coder-pro:free",
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0.7,
        streaming=True,
        max_retries=2
    )
    
    search_tool = TavilySearch(
        max_results=3,
        include_answer=True,
        search_depth="advanced",
        tavily_api_key=os.getenv("TAVILY_API_KEY")
    )
    
    mcp_tools, mcp_client = await load_mcp_tools()
    tools = [search_tool, calculator_tool, retriever_tool, *mcp_tools]
    print(f"Available tools: {len(tools)}")
    
    llm_with_tools = llm.bind_tools(tools=tools)
    
    conn = await aiosqlite.connect(database="database/chatbot.db")
    checkpointer = AsyncSqliteSaver(conn)
    
    async def chat_node(state: ChatState, config=None):
        """Process chat messages using the LLM."""
        thread_id = None
        if config:
            thread_id = config.get('configurable', {}).get('thread_id')
        
        # Check if documents are uploaded
        has_docs = thread_id and thread_id in _THREAD_RAGS
        
        system_prompt = "You are a helpful AI assistant."
        if has_docs:
            doc_info = _THREAD_DOCS_METADATA.get(thread_id, [])
            filenames = [d['filename'] for d in doc_info]
            system_prompt += f"\n\nUser has uploaded: {', '.join(filenames)}. "
            system_prompt += f"For questions about these documents, use retriever_tool with thread_id='{thread_id}'."
        
        messages = [SystemMessage(content=system_prompt)] + state['messages']
        response = await llm_with_tools.ainvoke(messages)
        return {'messages': [response]}
    
    tool_node = ToolNode(tools=tools) if tools else None
    
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

# ==================== Endpoints ====================

@app.post("/upload")
async def upload_files(
    thread_id: str,
    files: List[UploadFile] = File(...)
):
    """Upload multiple files for RAG"""
    try:
        results = []
        for file in files:
            file_bytes = await file.read()
            metadata = await ingest_file(file_bytes, file.filename, thread_id)
            results.append(metadata)
        
        return {
            'status': 'success',
            'files': results,
            'total_files': len(results)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/documents/{thread_id}/{filename}")
async def delete_document(thread_id: str, filename: str):
    """Delete a specific document from the knowledge base"""
    try:
        if thread_id not in _THREAD_DOCS_METADATA:
            raise HTTPException(status_code=404, detail="Thread not found")
        
        # Find and remove the document
        docs = _THREAD_DOCS_METADATA[thread_id]
        doc_to_remove = None
        for i, doc in enumerate(docs):
            if doc['filename'] == filename:
                doc_to_remove = i
                break
        
        if doc_to_remove is None:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Remove from metadata
        _THREAD_DOCS_METADATA[thread_id].pop(doc_to_remove)
        
        # If no documents left, clear the RAG store
        if not _THREAD_DOCS_METADATA[thread_id]:
            if thread_id in _THREAD_RAGS:
                del _THREAD_RAGS[thread_id]
            del _THREAD_DOCS_METADATA[thread_id]
        else:
            # Rebuild RAG store without this document
            # Note: This is a simple implementation. For production, you'd want to track chunks by document
            pass
        
        return {'status': 'success', 'message': f'{filename} removed'}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents/{thread_id}")
async def get_documents(thread_id: str):
    """Get uploaded documents for a thread"""
    docs = _THREAD_DOCS_METADATA.get(thread_id, [])
    return {'documents': docs}

@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()
    
    try:
        while True:
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
            
            try:
                async for message_chunk, metadata in chatbot.astream(
                    msg, 
                    config=config, 
                    stream_mode='messages'
                ):
                    if isinstance(message_chunk, AIMessage) and hasattr(message_chunk, 'tool_calls') and message_chunk.tool_calls:
                        tools_used = []
                        for tool_call in message_chunk.tool_calls:
                            tool_name = tool_call["name"].lower()
                            if 'tavily' in tool_name:
                                tools_used.append('search')
                            elif 'calculator' in tool_name or 'math' in tool_name:
                                tools_used.append('calculator')
                            elif 'retriever' in tool_name:
                                tools_used.append('retriever')
                            else:
                                tools_used.append(tool_name)
                        
                        await websocket.send_json({
                            "type": "tool_call",
                            "tools": tools_used
                        })
                    
                    if isinstance(message_chunk, AIMessage) and message_chunk.content:
                        await websocket.send_json({
                            "type": "content",
                            "content": message_chunk.content
                        })
                
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