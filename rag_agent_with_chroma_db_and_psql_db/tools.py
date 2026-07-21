import os

from dotenv import load_dotenv
load_dotenv()

from langchain_mcp_adapters.client import MultiServerMCPClient

from langgraph.prebuilt import ToolRuntime

from langchain_tavily import TavilySearch

from langchain.tools import tool


async def load_mcp_tools():
    """Load MCP tools asynchronously."""
    try:
        mcp_client = MultiServerMCPClient({   
            'System info.': {
                'transport': 'stdio',
                'command': 'python',
                'args': ["/home/toqeer-yasir/Documents/repos/ai-agents-with-langgraph/agentic_chatbot/local_mcp_servers/system_info_mcp_server.py"]
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
        raise RuntimeError(f"Error loading MCP tools: {e}")


@tool
async def rag_search(query: str, runtime: ToolRuntime):
    """ Search documents using the RAG retriever. """
    user_id = runtime.context["user_id"]
    rag = runtime.context["rag"]
    retriever = rag.get_retriever(user_id= user_id)
    docs = await retriever.ainvoke(query)
    return str(docs)


online_search = TavilySearch(
    max_results=3,
    include_answer=True,
    search_depth="advanced",
    tavily_api_key=os.getenv("TAVILY_API_KEY")
)


async def get_tools():
    mcp_tools, mcp_client = await load_mcp_tools()
    return list(mcp_tools) + [rag_search, online_search], mcp_client