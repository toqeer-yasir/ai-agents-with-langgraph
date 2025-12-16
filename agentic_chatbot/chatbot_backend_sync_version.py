# basic libraries:
import os
import math

from langchain_openai import ChatOpenAI

# library to create state_class
from typing import TypedDict, Annotated
from langchain_core.messages import AIMessage, BaseMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START
from dotenv import load_dotenv

# from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

# libraries to add tools:
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_tavily import TavilySearch


load_dotenv()

llm = ChatOpenAI(
     model= "tngtech/tng-r1t-chimera:free",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    temperature=0.1
    )



# creating tools:
@tool
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



tools = [calculator_tool, search_tool]
llm_with_tools = llm.bind_tools(tools=tools)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    message = state['messages']
    response = llm_with_tools.invoke(message)
    return {'messages': [response]}
    # return {'messages': [AIMessage(content=response.content)]}

tool_node = ToolNode(tools=tools)

# define graph:
graph = StateGraph(ChatState)

# add nodes:
graph.add_node('chat_node', chat_node)
graph.add_node('tools', tool_node)

# add edges:
graph.add_edge(START, 'chat_node')
graph.add_conditional_edges('chat_node', tools_condition)
graph.add_edge('tools', 'chat_node')


CONN = sqlite3.connect(database='database/chatbot.db', check_same_thread=False)

CHECKPOINTER = SqliteSaver(conn=CONN)
chatbot = graph.compile(checkpointer= CHECKPOINTER)

def retrieve_all_threads():
    all_threads = set()
    for checkpoint in CHECKPOINTER.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])
    return  list(all_threads)
