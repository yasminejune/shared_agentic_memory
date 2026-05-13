"""This script counts the number of task instantiations from the task descriptions json"""

import json

import pandas as pd

num_instantiations: dict[str, int] = {}

with open("../task_descriptions_all.json") as file:
    data: list[dict[str, str]] = json.load(file)

# Count and print num instantiations
for entry in data:
    if entry["intent_template_id"] not in num_instantiations.keys():
        num_instantiations[entry["intent_template_id"]] = 1
    else:
        num_instantiations[entry["intent_template_id"]] += 1

for item in num_instantiations:
    print(item, ": ", num_instantiations[item])

print("Min instantiations: ", min(num_instantiations.values()))
print("Max instantiations: ", max(num_instantiations.values()))
print("Length of instantations: ", len(num_instantiations))
print(f"Length of json: {len(data)}")

# Save as csv
csv = pd.Series(num_instantiations).reset_index()
csv.columns = ["intent_template_id", "num_instantiations"]
csv.to_csv("instantiations.csv", index=False)
