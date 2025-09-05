from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict, Literal
from pydantic import BaseModel, Field
import os
import json
import re

print(load_dotenv())

model = ChatOpenAI(
    model="mistralai/mistral-7b-instruct", # Free model
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0.1,
    max_tokens=200

)
class ReviewState(TypedDict):
    review: str
    sentiment: str
    response: str
    diagnosis: dict

def find_sentiment(state: ReviewState):
    prompt = f"Perform sentiment analysis on the following text: {state['review']}"    
    response = model.invoke(prompt).content.lower()
    sentiment = 'positive' if 'positive' in response.split('. ')[0] else 'negative'
    return {'sentiment': sentiment}

def check_sentiment(state: ReviewState):
    return 'positive_response' if state['sentiment' ] == 'positive' else 'run_diagnosis'

def positive_response(state: ReviewState):
    prompt = f"""write a warm thank-you message in response to that review:
    \n\n{state['review']}\n\n
    Also, kindly ask the user to leve feedback on our website."""
    response = model.invoke(prompt).content
    return {'response': response}

def run_diagnosis(state: ReviewState):
    prompt = f"""Diagnose this negative review:\n\n{state['review']}\n\n
    Return a JSON object with issue_type, tone and urgency fields."""
    response = model.invoke(prompt).content
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if json_match:
        json_str = json_match.group(0)
        data = json.loads(json_str)
    return {'diagnosis': data}

def negative_response(state: ReviewState):
    prompt = f"""You are an assistant write a helpful resolution message for our user on the basis of the review:\n\n{state['review']}\n\n
    Users issue_type is: {state['diagnosis']['issue_type']}\n\n
    Users tone: {state['diagnosis']['tone']}\n\n
    users urency: {state['diagnosis']['urgency']}"""
    response = model.invoke(prompt).content
    return {'response': response}

graph = StateGraph(ReviewState)

# Add nodes to the graph:
graph.add_node('find_sentiment', find_sentiment)
graph.add_node('positive_response', positive_response)
graph.add_node('run_diagnosis', run_diagnosis)
graph.add_node('negative_response', negative_response)

# Add nodes to the graph:
graph.add_edge(START, 'find_sentiment')
graph.add_conditional_edges('find_sentiment', check_sentiment, ['run_diagnosis', 'positive_response'])
graph.add_edge('run_diagnosis', 'negative_response')
graph.add_edge('negative_response', END)
graph.add_edge('positive_response', END)

workflow = graph.compile()

# negative_review = "i've been using this app for about a month now, and i must say, the user interface is incredibly clean and intutive."
# positive_review = "i've been trying to login for over an hour now, and the app keeps freezing on the authentication screen. I enven tried reinstalling it, but no luck. this kind of bug is unacceptable, especially affects basic functionality."

def user_email():
    text = input("Input review:")
    result = workflow.invoke({'review': text})
    print(result['sentiment'])
    print(result['response'])




if __name__ == "__main__":
    user_email()