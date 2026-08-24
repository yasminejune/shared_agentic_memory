"""Peek at nayohan/multi_session_chat.

Loaded this while looking for a dialogue corpus that might stand in
for multi-user memory text. Prints schema, a few dialogue rows, and
the Python type. Not wired into anything.
"""

from datasets import load_dataset

dataset = load_dataset("nayohan/multi_session_chat")
df = dataset["train"].to_pandas()
print(df.info())
print(df["dialogue"].head())
print(type(df["dialogue"]))
print([str(t) for t in df["dialogue"]][:5])
