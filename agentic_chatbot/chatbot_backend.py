# basic libraries:
import os
import math

from langchain_openai import ChatOpenAI

# library to create state_class
from typing import TypedDict, Annotated
from langchain_core.messages import AIMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

# from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

# libraries to add tools:
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun



load_dotenv()

llm = ChatOpenAI(
     model= "x-ai/grok-4.1-fast:free",
    # model= "kwaipilot/kat-coder-pro:free",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0.1
    )

# creating tools:
@tool
def calculator(expression: str) -> str:
    """Evaluate mathematical expressions."""
    try:
        allowed_names = {**math.__dict__}
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"



search_tool = DuckDuckGoSearchRun(region= 'us-en')
calculator_tool = calculator
tools = [calculator_tool, search_tool]
llm_with_tools = llm.bind_tools(tools=tools)

class ChatState(TypedDict):
    messages: Annotated[list, add_messages]

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

def retrive_threads():
    all_threads = set()
    for checkpoint in CHECKPOINTER.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])
    return  list(all_threads)
