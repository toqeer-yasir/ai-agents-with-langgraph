from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage
from typing import TypedDict, Optional, Literal
import os

load_dotenv()

# LLMs
master_llm = ChatOpenAI(
    model="tngtech/tng-r1t-chimera:free",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    temperature=0.6
)
general_llm = ChatOpenAI(
    model="tngtech/tng-r1t-chimera:free",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    temperature=0.2
)
coding_llm = ChatOpenAI(
    model="kwaipilot/kat-coder-pro:free",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    temperature=0.2
)

# state
class AgentState(TypedDict):
    user_query: str
    task_type: Optional[Literal['general', 'coding']]
    general_task: Optional[str]
    coding_task: Optional[str]
    general_response: Optional[str]
    coding_response: Optional[str]
    final_response: Optional[str]


# ===== general subgraph =====
general_graph = StateGraph(AgentState)

def chat_node(state: AgentState):
    response = general_llm.invoke(state["general_task"])
    return {'general_response': response.content + "[used general_subgaph]"}


general_graph.add_node("chat_node", chat_node)

general_graph.add_edge(START, "chat_node")
general_graph.add_edge("chat_node", END)

general_subgraph = general_graph.compile()

# ===== coding subgraph =====
coding_graph = StateGraph(AgentState)

def coding_node(state: AgentState):
    response = coding_llm.invoke(state['coding_task'])
    return {'coding_response': response.content + "[used coding_subgraph]"}

coding_graph.add_node('coding_node', coding_node)
coding_graph.add_edge(START, 'coding_node')
coding_graph.add_edge('coding_node', END)

coding_subgraph = coding_graph.compile()

# ===== master graph =====
master_graph = StateGraph(AgentState)

# task type
def task_classifier(state: AgentState):
    messages = [
        SystemMessage(content="""You are a task classifier. Read the user query and classify it as:
        1. "general" - for explanations, summaries, general knowledge questions, calculations
        2. "coding" - for programming, code generation, debugging
        Respond with ONLY one word: "general" or "coding"."""),
        HumanMessage(content=state['user_query'])
    ]
    response = master_llm.invoke(messages).content.strip().lower()
    return {'task_type': response}

# assign task node
def assign_task(state: AgentState):
    if state['task_type'] == 'coding':
        return {'coding_task': state['user_query']}
    else:
        return {'general_task': state['user_query']}

# router function
def route_to_subgraph(state: AgentState):
    if state['task_type'] == 'coding':
        return "coding_subgraph"
    else:
        return "general_subgraph"

# collecting final response node
def collect_response(state: AgentState):
    if state['task_type'] == 'coding':
        final = state.get('coding_response', 'No response generated')
    else:
        final = state.get('general_response', 'No response generated')
    return {'final_response': final}

# nodes
master_graph.add_node("classifier", task_classifier)
master_graph.add_node("assign_task", assign_task)
master_graph.add_node("general_subgraph", general_subgraph)
master_graph.add_node("coding_subgraph", coding_subgraph)
master_graph.add_node("collect_response", collect_response)

# edges
master_graph.add_edge(START, "classifier")
master_graph.add_edge("classifier", "assign_task")
master_graph.add_conditional_edges("assign_task", route_to_subgraph)
master_graph.add_edge("general_subgraph", "collect_response")
master_graph.add_edge("coding_subgraph", "collect_response")
master_graph.add_edge("collect_response", END)

agent = master_graph.compile()

if __name__ == "__main__":
    result1 = agent.invoke({"user_query": "What is deep-learning give a proper defination?"})
    print("Query 1:", result1['user_query'])
    print("Type:", result1['task_type'])
    print("Response:", result1['final_response'])
    print("\n" + "="*50 + "\n")
    
    result2 = agent.invoke({"user_query": "Write a Python code to reverse a string"})
    print("Query 2:", result2['user_query'])
    print("Type:", result2['task_type'])
    print("Response:", result2['final_response'])