from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, BaseMessage, ToolMessage
from langchain.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from typing import TypedDict, Annotated, Literal, List
import os
from langchain_mcp_adapters.client import MultiServerMCPClient
import asyncio

load_dotenv()

# to use fastMCP tools we must use async technique to execute tools because it returns structured tools insted of normal tools.

llm = ChatOpenAI(
    model="tngtech/tng-r1t-chimera:free",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    temperature=0.6
)

from langchain_mcp_adapters.tools import load_mcp_tools

async def setup_with_session():
    client = MultiServerMCPClient({
        'Calculator': {
            'transport': 'stdio',
            'command': 'python',
            'args': ["/home/toqeer-yasir/Documents/repos/ai-agents-with-langgraph/advance_concepts/demo_mcp_server.py"]
        }
    })
    
    async with client.session("Calculator") as session:
        tools = await load_mcp_tools(session)
        return tools

# In your main code
tools = asyncio.run(setup_with_session())
print(f"Loaded {tools=}")
llm_with_tools = llm.bind_tools(tools)

class ToolState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

graph = StateGraph(ToolState)

def chat_node(state: ToolState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

tool_node = ToolNode(tools=tools)

graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile()

if __name__ == "__main__":
    user_input = input("Type here: ")

    state = chatbot.invoke(
        {"messages": [HumanMessage(content=user_input)]}
    )

    print("\nUser:")
    print(user_input)

    print("Assistant:")
    print(state["messages"][-1].content)
    for i, key in enumerate(state['messages']):
        if isinstance(key, ToolMessage):
            print('\n', state['messages'][i])
            break
    else:
        print("Dosen't hasattr ToolMessage.")