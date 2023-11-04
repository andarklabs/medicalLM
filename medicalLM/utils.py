import json

from datasets import Dataset, DatasetDict, load_dataset


def preprocess_function(examples, tokenizer):
    # create new inputs:
    inputs = [
        # TODO: might want to add <A> as a special token
        " ".join(["<Q>", x, "<A>", y])
        for x, y in zip(examples["instruction"], examples["label"])
    ]
    # pad the new Q..A.. sequence
    inputs = tokenizer(inputs, trunction=True, padding="max_length", max_length=512)

    # create labels:
    labels = inputs["input_ids"].copy()
    # shift the input to the right (so that the model predicts the next token, not wholesale
    # answer) to get the labels we will use for training
    labels = [[-100] + label[:-1] for label in labels]
    inputs["labels"] = labels
    return inputs


def load_sample_data(file: str = "data/know_merge.json", sample=3):
    with open(file, "r") as f:
        data = json.load(f)

    data = data[:sample]
    x_train = [d["instruction"] for d in data]
    y_train = [d["output"] for d in data]
    return x_train, y_train


def create_dataset(x_train, y_train):
    # create new dataset
    dataset = DatasetDict(
        # must be called "label" for loss to properly work in HF
        {
            "train": Dataset.from_dict({"label": y_train, "instruction": x_train}),
            "validation": Dataset.from_dict({"label": y_train, "instruction": x_train}),
        }
    )
    return dataset
