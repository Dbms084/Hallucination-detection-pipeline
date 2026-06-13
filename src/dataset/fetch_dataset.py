from datasets import load_dataset

def load_truthfulqa():
    dataset = load_dataset("truthfulqa/truthful_qa", "generation")
    return dataset["validation"]
# print(dataset["validation"][0])
if __name__ == "__main__":
    data = load_truthfulqa()

    print(f"Total Questions: {len(data)}")

    sample = data[0]

    print("\nQuestion:")
    print(sample["question"])

    print("\nAnswer:")
    print(sample["best_answer"])
 