from datasets import load_dataset

dataset = load_dataset("nayohan/multi_session_chat")
df = dataset["train"].to_pandas()
print(df.info())
print(df["dialogue"].head())
print(type(df["dialogue"]))
print([str(t) for t in df["dialogue"]][:5])
