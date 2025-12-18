from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, BaseMessage
from langchain.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from typing import TypedDict, Annotated, Literal, List
import os
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt

load_dotenv()

llm = ChatOpenAI(
    model="tngtech/tng-r1t-chimera:free",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    temperature=0.6
)

@tool
def read_file(path: str):
    """perform basic arithmetic operations."""
    "Before performing any operation this tools will take permission from the user to do so or not (Yes/No)."
    decission = interrupt(f"Allow permission, do you really wanna open it? Yes/No")
    
    if isinstance(decission, str) and decission.lower() == 'yes':
        with open(path, 'r') as f:
            return {'result': f.read()}
    else:
        return {'status': "You canceled the operation to read file."}
         


tools = [read_file]
llm_with_tools = llm.bind_tools(tools)

class ToolState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

graph = StateGraph(ToolState)

def chat_node(state: ToolState):
        response = llm_with_tools.invoke(state["messages"])
        return {'messages': [response]}

tool_node = ToolNode(tools=tools)

graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

CHECKPOINTER = MemorySaver()
chatbot = graph.compile(CHECKPOINTER)

if __name__ == "__main__":
    while True:
        user_input = input("Type here: ")

        CONFIG = {'configurable': {'thread_id': 244}}
        
        result = chatbot.invoke(
            {'messages': [HumanMessage(content=user_input)]},
            config=CONFIG
        )
        
        interrupt_result = result.get("__interrupt__", [])
        
        if interrupt_result:
            print(f"HITL Permission {interrupt_result[0].value}")
            u_decission = input("Give permission: ").strip()

            result = chatbot.invoke(
                  Command(resume=u_decission),
                  config=CONFIG
                  ) 
            print("\nUser:")
            print(user_input)
            print("Assistant:")
            print(result["messages"][-1].content)