"""Conditional LangGraph: weather node, then rainy or sunny.

Coin-flip branch, not an LLM. Wanted to see add_conditional_edges
before putting a real Think node behind it.
"""

import random
from typing import Literal

from langgraph.graph import END, START, MessagesState, StateGraph


def weather(state):
    return {"messages": [{"role": "assistant", "content": "Hi! Well.. I have no idea... But... "}]}


def rainy_weather(state):
    return {
        "messages": [
            {"role": "assistant", "content": "Its going to rain today. Carry an umbrella."}
        ]
    }


def sunny_weather(state):
    return {
        "messages": [
            {"role": "assistant", "content": "Its going to be sunny today. Wear sunscreen."}
        ]
    }


def forecast_weather(str) -> Literal["rainy", "sunny"]:
    if random.random() < 0.5:
        return "rainy"
    else:
        return "sunny"


workflow = StateGraph(MessagesState)

workflow.add_node("weather", weather)
workflow.add_node("sunny", sunny_weather)
workflow.add_node("rainy", rainy_weather)

workflow.add_edge(START, "weather")
workflow.add_conditional_edges("weather", forecast_weather)
workflow.add_edge("rainy", END)
workflow.add_edge("sunny", END)

app = workflow.compile()

result = app.invoke(
    {"messages": [{"role": "user", "content": "Hi! What does the weather look like?"}]}
)
for msg in result["messages"]:
    print(f"{msg.type}: {msg.content}")
