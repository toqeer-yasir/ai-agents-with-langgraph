import asyncio
import os
from typing import Annotated
from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool
from typing_extensions import TypedDict


load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")


model = ChatOpenRouter(
    model="openrouter/free",
    api_key=OPENROUTER_API_KEY,
    temperature=0.7,
)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


async def chatbot(state: ChatState):

    response = await model.ainvoke(state["messages"])

    return {
        "messages": [response]
    }


builder = StateGraph(ChatState)

builder.add_node("chatbot", chatbot)

builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)


async def main():

    async with AsyncConnectionPool(
        DATABASE_URL,
        min_size=4,
        max_size=10
    ) as pool:

        checkpointer = AsyncPostgresSaver(pool)
        await checkpointer.setup()

        graph = builder.compile(checkpointer=checkpointer)

        thread_id = "chat_2"

        print("\nType 'exit' to quit.\n")

        while True:

            query = input("You : ")

            if query.lower() == "exit":
                break

            result = await graph.ainvoke({"messages": [HumanMessage(content= query)]},
                config={"configurable": {"thread_id": thread_id,}},)

            result = "\nAssistant :", result["messages"][-1].content
            print(f"{result[0]} {result[1]}")


if __name__ == "__main__":
    asyncio.run(main())