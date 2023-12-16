import json
from typing import Any

import evaluate
import fire
import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BioGptForCausalLM,
    BioGptTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    pipeline,
    set_seed,
)


def infer(
    model_directory: str = "causal-models/preprocessed",
    prompt: str = "What are the 23 amino acids?",
    num_return_sequences: int = 1,
    max_length: int = 200,
) -> Any:
    # # Load the trained model
    model_directory = model_directory
    model = BioGptForCausalLM.from_pretrained(model_directory)
    tokenizer = BioGptTokenizer.from_pretrained(model_directory)
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    set_seed(42)
    prompt = prompt
    out = generator(
        f"Question: {prompt} Answer:",
        max_length=50,
        num_return_sequences=1,
        do_sample=True,
    )
    return out


if __name__ == "__main__":
    fire.Fire(infer)
