import json

import evaluate
import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_dataset
from peft import LoraConfig, PeftConfig, PeftModel, TaskType, get_peft_model
from transformers import (
    AutoConfig,
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

"""
[{
  "instruction": "My daughter ( F, 18 y\/o, 5'5', 165lbs) has been feeling poorly for a 6-8 months. She had COVID a couple of months ago and symptoms have are much worse in the last month or so. Symptoms seem POTS-like. She feels light headed, breathless, dizzy, HR goes from ~65 lying down to ~155-160 on standing. Today she tells me HR has been around 170 all day and she feels really lousy. (She using an OTC pulse ox to measure.) She has a cardiology appt but not until March and a PCP appt but not until April since she's at school and it's a new provider. What to do? Is this a on call nurse sort of issue? Or a trip to the ED? Or wait till tomorrow and try for an early appt? Try a couple of Valsalvas? Wait it out until her cardio appt? Or? She's away at school if Boston, what to do? Thank you",
  "output": "If she actually has a HR of 170 that is accurate, ongoing and persistent, she needs to be seen in the ED immediately."
},
]

"""


def load_data(file: str = "data/sample.json"):
    with open(file, "r") as f:
        data = json.load(f)

    x_train = [d["instruction"] for d in data]
    y_train = [d["output"] for d in data]
    return x_train, y_train


x_train, y_train = load_data()

# # create new dataset
dataset = DatasetDict(
    # must be called "label" for loss to properly work in HF
    {
        "train": Dataset.from_dict({"label": y_train, "instruction": x_train}),
        "validation": Dataset.from_dict({"label": y_train, "instruction": x_train}),
    }
)

from transformers import (
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

# model = BioGptForCausalLM.from_pretrained("microsoft/BioGPT-Large")
# tokenizer = BioGptTokenizer.from_pretrained("microsoft/BioGPT-Large")
model = BioGptForCausalLM.from_pretrained("microsoft/BioGPT")
tokenizer = BioGptTokenizer.from_pretrained("microsoft/BioGPT")
# tokenizer = AutoTokenizer.from_pretrained("microsoft/BioGPT-Large")
tokenizer.pad_token = tokenizer.eos_token
# data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# TODO: does changing between these make a difference?
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
# from transformers import DataCollatorForSeq2Seq
# data_collator = DataCollatorForSeq2Seq(tokenizer)


# TODO: this trains, but is it actually using instruction as input and label as goal prediction (ie QA)
# i cant tell if this combines the instruction and answer appropriately.
# is the value of QA dataset so this preprocess can cleanly combine the two?
def preprocess_function(examples):
    inputs = tokenizer(
        examples["instruction"], truncation=True, padding="max_length", max_length=512
    )
    # outputs = tokenizer(examples["label"], truncation=True, padding='max_length', max_length=512) # Not needed as shown previously

    # Adjusting to the correct approach
    labels = inputs["input_ids"].copy()  # Clone the input_ids to use as labels
    # We need to shift the labels to the right to prepare them for a language modeling objective
    labels = [
        [-100 if token == tokenizer.pad_token_id else token for token in label]
        for label in labels
    ]
    # The labels should be shifted inside the preprocessing function as we need to pad the labels before the shift
    labels = [[-100] + label[:-1] for label in labels]  # shift labels to the right

    inputs["labels"] = labels

    return inputs


def preprocess_function_naive_seq2seq(examples):
    inputs = [
        " ".join([x, y]) for x, y in zip(examples["instruction"], examples["label"])
    ]
    return tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")
    # return tokenizer([
    #         " ".join([x, y]) for x, y in zip(examples["instruction"], examples["label"])
    #     ])


# tokenized_dataset = dataset.map(preprocess_function, batched=True)
tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=["instruction", "label"],
)
del dataset

# we can also try using int8 loading
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=4,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["q_proj"],
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

training_args = TrainingArguments(
    output_dir="causal-models/",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    # does it not see my gpu?
    # fp16=True,  # if this doesn't work try regular size model
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],  # lm_dataset["train"],
    # we use train and eval as same ds for now
    eval_dataset=tokenized_dataset["validation"],  # lm_dataset["test"],
    data_collator=data_collator,
)

# sigkilled as is (too big i guess to train as is)
trainer.train()
model_to_save = (
    trainer.model.module if hasattr(trainer.model, "module") else trainer.model
)  # Take care of distributed/parallel training
model_to_save.save_pretrained("causal-models/")


# model = BioGptForCausalLM.from_pretrained("microsoft/BioGPT-Large")
# print(model.config)
# tokenizer = BioGptTokenizer.from_pretrained("microsoft/BioGPT-Large")
# # generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
# # set_seed(42)
# # out = generator(
# #     "My elbow hurts after rock climbing is a symptom of",
# #     max_length=20,
# #     num_return_sequences=5,
# #     do_sample=True,
# # )


# # from transformers import AutoModelForQuestionAnswering, Trainer, TrainingArguments
# BioGptConfig {
#   "_name_or_path": "microsoft/BioGPT-Large",
#   "activation_dropout": 0.0,
#   "architectures": [
#     "BioGptForCausalLM"
#   ],
#   "attention_probs_dropout_prob": 0.1,
#   "bos_token_id": 0,
#   "eos_token_id": 2,
#   "hidden_act": "gelu",
#   "hidden_dropout_prob": 0.1,
#   "hidden_size": 1600,
#   "initializer_range": 0.02,
#   "intermediate_size": 6400,
#   "layer_norm_eps": 1e-12,
#   "layerdrop": 0.0,
#   "max_position_embeddings": 2048,
#   "model_type": "biogpt",
#   "num_attention_heads": 25,
#   "num_hidden_layers": 48,
#   "pad_token_id": 1,
#   "scale_embedding": true,
#   "torch_dtype": "float32",
#   "transformers_version": "4.34.1",
#   "use_cache": true,
#   "vocab_size": 57717
# }


# how dataset was generated

# load imdb data
# imdb_dataset = load_dataset("imdb")

# # define subsample size
# N = 1000
# # generate indexes for random subsample
# rand_idx = np.random.randint(24999, size=N)

# # extract train and test data
# x_train = imdb_dataset["train"][rand_idx]["text"]
# y_train = imdb_dataset["train"][rand_idx]["label"]
# print(x_train[0], y_train[0])

# x_test = imdb_dataset["test"][rand_idx]["text"]
# y_test = imdb_dataset["test"][rand_idx]["label"]
# print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# create new dataset
# dataset = DatasetDict(
#     {
#         "train": Dataset.from_dict({"label": y_train, "text": x_train}),
#         "validation": Dataset.from_dict({"label": y_test, "text": x_test}),
#     }
# )
