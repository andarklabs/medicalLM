# not working yet, see traceback below

import json

import evaluate
import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_dataset
from peft import LoraConfig, PeftConfig, PeftModel, TaskType, get_peft_model
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

# # Load the trained model
model_directory = "causal-models/preprocessed"


model = BioGptForCausalLM.from_pretrained(model_directory)
tokenizer = BioGptTokenizer.from_pretrained(model_directory)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
set_seed(42)
out = generator(
    "Question: Why does my elbow hurt? Answer:",
    # "Question: List the 23 Amino Acids below: Answer:",
    max_length=50,
    num_return_sequences=5,
    do_sample=True,
)

print(out)

# model = AutoModelForCausalLM.from_pretrained(model_directory)

# # Load the tokenizer (assuming it's the same one used for training)
# tokenizer = AutoTokenizer.from_pretrained(model_directory)

# # Prepare the text you want to encode and generate from
# text = "<Q>Why does my elbow hurt?<A>"
# text = "<Q>I feel a twinge in my elbow after climbing, should I see a doctor?<A>"

# # Encode the text into input IDs
# input_ids = tokenizer.encode(text, return_tensors="pt")

# # Generate output using the model
# output = model.generate(input_ids, max_length=50)  # max_length is just an example

# # Decode the output to a human-readable text
# decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)

# print(decoded_output)
# def load_data(file: str = "data/sample.json"):
#     with open(file, "r") as f:
#         data = json.load(f)

#     x_train = [d["instruction"] for d in data]
#     y_train = [d["output"] for d in data]
#     return x_train, y_train


# x_train, y_train = load_data()

# # # create new dataset
# dataset = DatasetDict(
#     # must be called "label" for loss to properly work in HF
#     {
#         "train": Dataset.from_dict({"label": y_train, "instruction": x_train}),
#         "validation": Dataset.from_dict({"label": y_train, "instruction": x_train}),
#     }
# )

# from transformers import (
#     AutoModelForCausalLM,
#     DataCollatorForLanguageModeling,
#     Trainer,
#     TrainingArguments,
# )

# tokenizer = BioGptTokenizer.from_pretrained("microsoft/BioGPT")

# data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


# def preprocess_function(examples):
#     inputs = tokenizer(
#         examples["instruction"], truncation=True, padding="max_length", max_length=512
#     )
#     # outputs = tokenizer(examples["label"], truncation=True, padding='max_length', max_length=512) # Not needed as shown previously

#     # Adjusting to the correct approach
#     labels = inputs["input_ids"].copy()  # Clone the input_ids to use as labels
#     # We need to shift the labels to the right to prepare them for a language modeling objective
#     labels = [
#         [-100 if token == tokenizer.pad_token_id else token for token in label]
#         for label in labels
#     ]
#     # The labels should be shifted inside the preprocessing function as we need to pad the labels before the shift
#     labels = [[-100] + label[:-1] for label in labels]  # shift labels to the right

#     inputs["labels"] = labels

#     return inputs


# def preprocess_function_naive_seq2seq(examples):
#     inputs = [
#         " ".join([x, y]) for x, y in zip(examples["instruction"], examples["label"])
#     ]
#     return tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")


# # tokenized_dataset = dataset.map(preprocess_function, batched=True)
# tokenized_dataset = dataset.map(
#     preprocess_function,
#     batched=True,
#     remove_columns=["instruction", "label"],
# )
# del dataset
# # loading the model you previously trained
# model = AutoModelForSequenceClassification.from_pretrained("causal-models/")

# test_dataset = tokenized_dataset["validation"]
# # arguments for Trainer
# test_args = TrainingArguments(
#     output_dir="causal-models/",
#     do_train=False,
#     do_predict=True,
#     per_device_eval_batch_size=4,
#     dataloader_drop_last=False,
# )

# # init trainer
# trainer = Trainer(model=model, args=test_args)  # , compute_metrics=compute_metrics)

# test_results = trainer.predict(test_dataset)

# """
# Traceback (most recent call last):
#   File "/home/arelius/workspace/medicalLM/medicalLM/ft/sample_inf.py", line 159, in <module>
#     test_results = trainer.predict(test_dataset)
#   File "/home/arelius/miniconda3/envs/biogpt/lib/python3.10/site-packages/transformers/trainer.py", line 3142, in predict
#     output = eval_loop(
#   File "/home/arelius/miniconda3/envs/biogpt/lib/python3.10/site-packages/transformers/trainer.py", line 3255, in evaluation_loop
#     loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
#   File "/home/arelius/miniconda3/envs/biogpt/lib/python3.10/site-packages/transformers/trainer.py", line 3474, in prediction_step
#     loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
#   File "/home/arelius/miniconda3/envs/biogpt/lib/python3.10/site-packages/transformers/trainer.py", line 2801, in compute_loss
#     outputs = model(**inputs)
#   File "/home/arelius/miniconda3/envs/biogpt/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
#     return self._call_impl(*args, **kwargs)
#   File "/home/arelius/miniconda3/envs/biogpt/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1527, in _call_impl
#     return forward_call(*args, **kwargs)
#   File "/home/arelius/miniconda3/envs/biogpt/lib/python3.10/site-packages/transformers/models/biogpt/modeling_biogpt.py", line 956, in forward
#     loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
#   File "/home/arelius/miniconda3/envs/biogpt/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
#     return self._call_impl(*args, **kwargs)
#   File "/home/arelius/miniconda3/envs/biogpt/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1527, in _call_impl
#     return forward_call(*args, **kwargs)
#   File "/home/arelius/miniconda3/envs/biogpt/lib/python3.10/site-packages/torch/nn/modules/loss.py", line 1179, in forward
#     return F.cross_entropy(input, target, weight=self.weight,
#   File "/home/arelius/miniconda3/envs/biogpt/lib/python3.10/site-packages/torch/nn/functional.py", line 3053, in cross_entropy
#     return torch._C._nn.cross_entropy_loss(input, target, weight, _Reduction.get_enum(reduction), ignore_index, label_smoothing)
# ValueError: Expected input batch_size (2) to match target batch_size (1024).


# """
