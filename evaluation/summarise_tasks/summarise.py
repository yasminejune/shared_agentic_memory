"""Count how many WebArena tasks share each intent_template_id."""

import json

import pandas as pd


def count_instantiations(data: list[dict[str, str]]) -> dict[str, int]:
    """How many tasks share each intent_template_id."""
    counts: dict[str, int] = {}
    for entry in data:
        template_id = entry["intent_template_id"]
        counts[template_id] = counts.get(template_id, 0) + 1
    return counts


def save_to_csv(counts: dict[str, int], path: str) -> None:
    csv = pd.Series(counts).reset_index()
    csv.columns = pd.Index(["intent_template_id", "num_instantiations"])
    csv.to_csv(path, index=False)


if __name__ == "__main__":
    with open("../task_descriptions_all.json") as file:
        data: list[dict[str, str]] = json.load(file)

    num_instantiations = count_instantiations(data)

    for item, count in num_instantiations.items():
        print(item, ": ", count)

    print("Min instantiations: ", min(num_instantiations.values()))
    print("Max instantiations: ", max(num_instantiations.values()))
    print("Length of instantiations: ", len(num_instantiations))
    print(f"Length of json: {len(data)}")

    save_to_csv(num_instantiations, "instantiations.csv")
