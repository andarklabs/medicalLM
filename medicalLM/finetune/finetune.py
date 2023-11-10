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

set_seed(42)

from datasets import Dataset, DatasetDict, load_dataset

from utils import (
    create_dataset,
    load_preprocessed_data,
    load_sample_data,
    preprocess_function,
)

model = BioGptForCausalLM.from_pretrained("microsoft/BioGPT")
tokenizer = BioGptTokenizer.from_pretrained("microsoft/BioGPT")
tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

'''we need to have more unique end tokens
        self.model_input = """
<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant""".format(
            system_message=self.system_message,
            prompt=self.prompt,
        )
'''


def causal_preprocess_function(examples) -> dict:
    return NotImplementedError


def qa_preprocess_function(examples):
    inputs = ["<im_start> " + q + " <im_end>" for q in examples["instruction"]]
    targets = ["<im_start> " + a + " <im_end>" for a in examples["label"]]
    inputs = [f"Question: {q} Answer:" for q in examples["instruction"]]
    targets = [a for a in examples["label"]]
    model_inputs = tokenizer(
        inputs, max_length=512, padding="max_length", truncation=True
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets, max_length=512, padding="max_length", truncation=True
        )

    labels["input_ids"] = [
        [
            (label if label != tokenizer.pad_token_id else -100)
            for label in label_example
        ]
        for label_example in labels["input_ids"]
    ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


x_train, y_train = load_preprocessed_data()
dataset = create_dataset(x_train, y_train)

tokenized_dataset = dataset.map(
    qa_preprocess_function,
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
    output_dir="causal-models/preprocessed-2",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    per_device_train_batch_size=8,  # TODO: make preprocess handle variable pad lengths
    per_device_eval_batch_size=8,
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
model_to_save.save_pretrained("causal-models/preprocessed-2")
tokenizer.save_pretrained("causal-models/preprocessed-2")
model.config.save_pretrained("causal-models/preprocessed-2")
