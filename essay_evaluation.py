from langgraph.graph import StateGraph, START, END
from IPython.display import Image
from typing import TypedDict, List
from langchain_openai import OpenAI
from dotenv import load_dotenv
from annotated_types import Annotated
from pydantic import BaseModel, Field
import operator

load_dotenv()

model = OpenAI(model="gpt-4o-mini")

# describing a schema to get output as we want in the form of a string and a score in int value:
class EvaluationSchema(BaseModel):
    feedback: str = Field(description="Detailed feedback for the essay.")
    score: int = Field(description="Score out of ten.", ge=0, le=10)

# new model to get structured output:
structured_model = model.with_structured_output(EvaluationSchema)

class EssayEvaluation(TypedDict):
    essay: str
    language_feedback: str
    clarity_feedback: str
    analysis_feedback: str
    individual_scores: Annotated[List[int], operator.add]
    summarized_feedback: str
    avg_score: float

# graph object:
graph = StateGraph(EssayEvaluation)

def Language_feedback(state: EssayEvaluation):
    prompt = f"Evaluate the language quality of the following essay: {state['essay']}"
    output = structured_model.invoke(prompt)
    # returning the required outputs:
    return {"language_feedback": output.feedback, "individual_scores": [output.score]}

def Clarity_feedback(state: EssayEvaluation):
    prompt = f"Evaluate the clarity of the following essay: {state['essay']}"
    output = structured_model.invoke(prompt)
    # returning the required outputs:
    return {"clarity_feedback": output.feedback, "individual_scores": [output.score]}

def Analysis_feedback(state: EssayEvaluation):
    prompt = f"Evaluate the depth of analysis of the following essay: {state['essay']}"
    output = structured_model.invoke(prompt)
    # returning the required outputs:
    return {"analysis_feedback": output.feedback, "individual_scores": [output.score]}

def Final_feedback(state: EssayEvaluation):
    prompt = f"""
    On the basis of the following feedbacks generate a new summarized feedback:
    Language feedback: {state["language_feedback"]}
    Clarity feedback: {state["clarity_feedback"]}
    Analysis feedback: {state["analysis_feedback"]}
    
    Please provide a comprehensive summary that highlights the key strengths and weaknesses.
    """
    output = model.invoke(prompt)
    # Calculate average score
    avg = sum(state["individual_scores"]) / len(state["individual_scores"]) if state["individual_scores"] else 0
    return {"summarized_feedback": output.content, "avg_score": avg}

# nodes:
graph.add_node("language_feedback", Language_feedback)
graph.add_node("clarity_feedback", Clarity_feedback)
graph.add_node("analysis_feedback", Analysis_feedback)
graph.add_node("final_feedback", Final_feedback)

# edges:
graph.add_edge(START, "language_feedback")
graph.add_edge(START, "clarity_feedback")
graph.add_edge(START, "analysis_feedback")
graph.add_edge("language_feedback", "final_feedback")
graph.add_edge("clarity_feedback", "final_feedback")
graph.add_edge("analysis_feedback", "final_feedback")
graph.add_edge("final_feedback", END)

# compile graph
workflow = graph.compile()

# Display the graph
# Image(workflow.get_graph().draw_mermaid_png())

if __name__ == "__main__":
    # user input:
    essay = """
    The 20th century, an era defined by unprecedented technological acceleration, witnessed the birth of an idea so profound it would forever alter humanity’s trajectory: the creation of artificial intelligence (AI). The rise of AI was not a sudden event but a gradual, intellectual crescendo, built upon centuries of philosophical inquiry and propelled by the practical necessities of a world recovering from global war and racing into a new technological age. The journey from theoretical concept to tangible reality throughout the 1900s represents one of the most ambitious and transformative scientific endeavors in human history.
    The seeds of AI were sown long before the century began, in the realms of mythology and philosophy with tales of artificial beings endowed with consciousness. However, the essential prerequisite for its rise was the development of the modern computer during the mid-20th century. Figures like Alan Turing provided the theoretical bedrock, moving the conversation from the mythical to the mathematical. His 1950 paper, "Computing Machinery and Intelligence," posed the seminal question, "Can machines think?" and proposed the famous Turing Test as a measure of machine intelligence. This period also saw the creation of the first electronic, programmable computers, such as ENIAC, which provided the physical hardware necessary to move from abstract theory to concrete experimentation.
    This fertile ground gave way to the formal birth of AI as an academic discipline at the now-legendary Dartmouth Conference of 1956. Organized by John McCarthy, who coined the term "artificial intelligence," the workshop brought together the field's founding figures, including Marvin Minsky, Claude Shannon, and Nathaniel Rochester. They were united by an audacious and optimistic belief: that "every aspect of learning or any other feature of intelligence can in principle be so precisely described that a machine can be made to simulate it." This conference marked the official launch of AI research, igniting a period of immense optimism and generous funding, now often referred to as the "first summer of AI."
    The subsequent decades were a rollercoaster of breakthroughs and setbacks, periods of exuberant "AI summers" and skeptical "AI winters." Early successes fueled the initial hype. Programs like the Logic Theorist and the General Problem Solver demonstrated that machines could mimic certain aspects of human problem-solving and logical reasoning. Joseph Weizenbaum’s ELIZA, a simple natural language processing program that simulated a Rogerian psychotherapist, astonished the public by creating the illusion of understanding, even though its mechanics were based on simplistic pattern matching.
    However, the field soon collided with the immense complexity of human cognition. Early AI, largely based on symbolic reasoning or "good old-fashioned AI" (GOFAI), struggled with tasks humans find trivial, such as visual recognition, commonsense reasoning, and navigating unstructured environments. The failure of machines to live up to the inflated promises, combined with critical reports like the Lighthill Report in the UK, led to a sharp reduction in funding and the first major "AI winter" in the 1970s.
    The resilience of the AI community, however, proved to be one of its greatest strengths. The latter part of the century witnessed a critical paradigm shift away from pure symbolic logic. The revival of connectionism—modeling AI on the neural networks of the human brain—and the development of machine learning algorithms offered a new path. Instead of being explicitly programmed for every rule, systems could now learn from data. Key developments, such as the backpropagation algorithm in the 1980s, provided a practical method for training multi-layer neural networks. Furthermore, the rise of expert systems in the 1980s, which captured the knowledge of human experts in specific domains like medicine or geology, provided commercial success and helped pull the field out of its winter, restoring credibility and investment.
    By the century’s close, AI was no longer a mere academic curiosity. In 1997, IBM’s Deep Blue defeated world chess champion Garry Kasparov, a milestone that captured the global imagination and demonstrated the sheer computational power of AI in a defined, rule-based domain. This event was a powerful symbol of how far the field had come since its theoretical beginnings.
    In conclusion, the 20th-century rise of artificial intelligence was a journey of monumental ambition, characterized by cycles of visionary hope and sobering reality checks. It evolved from a philosophical question into a formal scientific discipline, navigated through periods of disillusionment, and adapted by embracing new models like neural networks and machine learning. The century laid the essential groundwork—the theoretical frameworks, the algorithmic foundations, and the hardware—setting the stage for the explosive, data-driven AI revolution of the 21st century. It was a century of dawning, a prolonged and often tumultuous sunrise that heralded the arrival of a new form of intelligence, forever changing our relationship with technology and our understanding of our own minds.
    """

    # run workflow:
    result = workflow.invoke({"essay": essay})
    print("Summarized Feedback:", result["summarized_feedback"])
    print("Average Score:", result["avg_score"])
    print("Individual Scores:", result["individual_scores"])