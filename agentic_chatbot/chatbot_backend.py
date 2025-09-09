from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langgraph.graph.message import add_messages
# from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
import os


load_dotenv()

model4title = model = ChatOpenAI(
    # model="mistralai/mistral-7b-instruct", # Free model
    model="mistralai/Mistral-7B-Instruct-v0.1", # Free model
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0.1
    )

class ChatState(TypedDict):
    messages: Annotated[list, add_messages]

def chat_node(state: ChatState):
    message = state['messages']
    response = model.invoke(message)
    return {'messages': [AIMessage(content=response.content)]}

# Define graph:
graph = StateGraph(ChatState)
# Add nodes:
graph.add_node('chat_node', chat_node)
# Add edges:
graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

CONN = sqlite3.connect(database='database/chatbot.db', check_same_thread=False)

CHECKPOINTER = SqliteSaver(conn=CONN)
chatbot = graph.compile(checkpointer= CHECKPOINTER)

def retrive_threads():
    all_threads = set()
    for checkpoint in CHECKPOINTER.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])
    return  list(all_threads)