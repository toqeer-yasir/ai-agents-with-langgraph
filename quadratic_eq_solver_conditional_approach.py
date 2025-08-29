from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from IPython.display import Image
class QuadraticEq(TypedDict):
    a: int
    b: int
    c: int
    equation: str
    discriminant: float
    roots: str

graph = StateGraph(QuadraticEq)

def show_eq(state: QuadraticEq):
    sign_b = '+' if state['b'] >= 0 else ''
    sign_c = '+' if state['c'] >= 0 else ''
    eq_str = f"{state['a']}x² {sign_b}{state['b']}x {sign_c}{state['c']} = 0"
    return {"equation": eq_str}

def discriminant(state: QuadraticEq):
    disc = (state["b"]**2) - (4*(state["a"] * state["c"]))
    return {"discriminant": disc}

def real_roots(state: QuadraticEq):
    root1 = (-state["b"] + (state["discriminant"]**0.5)) / (2*state["a"])
    root2 = (-state["b"] - (state["discriminant"]**0.5)) / (2*state["a"])
    return {"roots": f"Real roots of the equation are: \nroot1: {root1:.2f}\nroot2: {root2:.2f}"}

def repeated_roots(state: QuadraticEq):
    root1 = -state["b"] / (2*state["a"])
    return {"roots": f"Single repeated root of the given equation is: \nroot: {root1:.2f}"}

def no_roots(state: QuadraticEq):
    return {"roots": "No real roots exist."}

def conditional_struct(state: QuadraticEq):
    if state["discriminant"] > 0:
        return "real_roots"
    elif state["discriminant"] == 0:
        return "repeated_roots"
    else:
        return "no_roots"

# adding_nodes:
graph.add_node("show_eq", show_eq)
graph.add_node("discriminant", discriminant)
graph.add_node("real_roots", real_roots)
graph.add_node("repeated_roots", repeated_roots)
graph.add_node("no_roots", no_roots)

# adding_edges:
graph.set_entry_point("show_eq")
graph.add_edge("show_eq", "discriminant")
graph.add_conditional_edges(
    "discriminant",
    conditional_struct,
    {
        "real_roots": "real_roots",
        "repeated_roots": "repeated_roots",
        "no_roots": "no_roots"
    }
)
graph.add_edge("real_roots", END)
graph.add_edge("repeated_roots", END)
graph.add_edge("no_roots", END)

workflow = graph.compile()


input_dict = {
    "a": int(input("Input value of 'a': ")),
    "b": int(input("Input value of 'b': ")),
    "c": int(input("Input value of 'c': "))
}

result = workflow.invoke(input_dict)

print("Input equatin: ", result["equation"])
print(result["roots"])
Image(workflow.get_graph().draw_mermaid_png())
