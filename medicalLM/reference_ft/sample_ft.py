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
    DataCollatorForLanguageModeling,
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

# MAKE A PROJECT
# FIX THE PYTOML and pip install editable
# clean up the project
from datasets import Dataset, DatasetDict, load_dataset
from utils import create_dataset, load_sample_data, preprocess_function

# model = BioGptForCausalLM.from_pretrained("microsoft/BioGPT-Large")
# tokenizer = BioGptTokenizer.from_pretrained("microsoft/BioGPT-Large")
model = BioGptForCausalLM.from_pretrained("microsoft/BioGPT")
tokenizer = BioGptTokenizer.from_pretrained("microsoft/BioGPT")
tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


def preprocess_function(examples):
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


x_train, y_train = load_sample_data()
dataset = create_dataset(x_train, y_train)

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
tokenizer.save_pretrained("causal-models/")
model.config.save_pretrained("causal-models/")
