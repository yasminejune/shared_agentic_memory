import subprocess
from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class State(TypedDict):
    message: str


def main():
    workflow = StateGraph(State)

    workflow.add_node("node_1", node1)
    workflow.add_node("node_2", node2)

    workflow.add_edge(START, "node_1")
    workflow.add_edge("node_1", "node_2")
    workflow.add_edge("node_2", END)

    app = workflow.compile()

    result = app.invoke({"message": "hello"})
    print(result)

    with open("graph.png", "wb") as f:
        f.write(app.get_graph().draw_mermaid_png())
    subprocess.run(["open", "graph.png"])  # macOS: opens in Preview


def node1(state: State) -> State:
    return {"message": state["message"] + " I reached Node1."}


def node2(state: State) -> State:
    return {"message": state["message"] + " I reached Node2."}


if __name__ == "__main__":
    main()
