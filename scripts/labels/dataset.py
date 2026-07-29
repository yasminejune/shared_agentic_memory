from datasets import load_dataset

# Load the automatically generated parquet data directly
dataset = load_dataset(
    "json",
    data_files={
        "test": "hf://datasets/tner/mit_movie_trivia/dataset/test.json",
    },
)
df = dataset["test"].to_pandas()
print(df.head())
print(df.info())
print(df["tokens"][0])
