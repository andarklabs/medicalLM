"""
clark you are being kinda silly.
this is so extra
we get an err bc biogpt apparently has some conflict with HF stuff?
i say we just move on and only do this sort of thing with torch

TODO:
    just clean up the files
    dataset in one place
    model in another place
    fine tune script in one place
    better way to save to not overwrite (just try overwrite=False)

    then test w andrew dataset and try to get a validation framework
    to see if shit is actually training as desired

"""
from typing import Any, Final

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


class BioGPT(BioGptForCausalLM):
    # this allows statically checking later i.e. Diffuser.DEFAULT_BATCH_SIZE
    DEFAULT_PATH = "causal-models/"
    DEFAULT_CONFIG = AutoConfig.from_pretrained(DEFAULT_PATH)
    DEFAULT_TASK_TYPE = TaskType.CAUSAL_LM
    DEFAULT_RANK = 4
    DEFAULT_ALPHA = 16
    DEFAULT_DROPOUT = 0.1
    DEFAULT_TARGET_MODULES = ["q_proj"]
    INFERENCE_MODE = False

    METADATA_KEY: Final[str] = "metadata"

    __slots__ = ["_metadata", "config", "_model"]

    _metadata: Final[dict[str, Any]]

    def __init__(
        self,
        path: str = DEFAULT_PATH,
        config: AutoConfig = DEFAULT_CONFIG,
        task_type: TaskType = DEFAULT_TASK_TYPE,
        r: int = DEFAULT_RANK,
        lora_alpha: float = DEFAULT_ALPHA,
        lora_dropout: float = DEFAULT_DROPOUT,
        target_modules: list[str] = DEFAULT_TARGET_MODULES,
        inference_mode: bool = False,
        *model_args,
        **model_kwargs,
    ):
        super().__init__(config, *model_args, **model_kwargs)
        self._metadata = {
            "path": path,
            "task_type": task_type,
            "inference_mode": inference_mode,
            "r": r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "target_modules": target_modules,
        }
        self._model = None
        # TODO: fix
        """
        Traceback (most recent call last):
          File "/home/arelius/workspace/medicalLM/medicalLM/model_ops/biogpt2.py", line 216, in <module>
            model.finetune(dataset=tokenized_dataset, collator=data_collator)
          File "/home/arelius/workspace/medicalLM/medicalLM/model_ops/biogpt2.py", line 104, in finetune
            peft_model = get_peft_model(self._model, peft_config)
          File "/home/arelius/miniconda3/envs/biogpt/lib/python3.10/site-packages/peft/mapping.py", line 110, in get_peft_model
            peft_config.base_model_name_or_path = model.__dict__.get("name_or_path", None)
        AttributeError: 'NoneType' object has no attribute '__dict__'. Did you mean: '__dir__'?
        """

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    @property
    def lora_config(self) -> LoraConfig:
        return LoraConfig(
            task_type=self.metadata["task_type"],
            inference_mode=self.metadata["inference_mode"],
            r=self.metadata["r"],
            lora_alpha=self.metadata["lora_alpha"],
            lora_dropout=self.metadata["lora_dropout"],
            target_modules=self.metadata["target_modules"],
        )

    @staticmethod
    def get_tokenizer(path) -> BioGptTokenizer:
        # tokenizer = BioGptTokenizer.from_pretrained("microsoft/BioGPT")
        # return BioGptTokenizer.from_pretrained(self._metadata["path"])
        return BioGptTokenizer.from_pretrained(path)

    def finetune(self, dataset, collator):
        peft_config = PeftConfig(self.lora_config)
        peft_model = get_peft_model(self._model, peft_config)
        # add default params for trainer
        trainer = Trainer(
            model=peft_model,
            args=TrainingArguments(
                output_dir=self._metadata["path"],
                overwrite_output_dir=True,
                num_train_epochs=1,
                per_device_train_batch_size=1,
                per_device_eval_batch_size=1,
            ),
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            data_collator=collator,
        )
        trainer.train()

    def save(self) -> None:
        self.save_pretrained(self._metadata["path"])
        self.tokenizer.save_pretrained(self._metadata["path"])
        self.config.save_pretrained(self._metadata["path"])

    @classmethod
    def pretrained(cls, path: str = DEFAULT_PATH, *model_args, **kwargs):
        # config = AutoConfig.from_pretrained(path)
        # # Initialize the model with pretrained weights.
        # # model = super().from_pretrained(path, config=config, *model_args, **kwargs)
        # model =
        # # tokenizer = BioGptTokenizer.from_pretrained(path)
        # cls._model = model
        # return cls
        config = AutoConfig.from_pretrained(path)
        # # cls._model = cls(path, config=config, *model_args, **kwargs)
        # model = super(BioGPT, cls).from_pretrained(
        #     path, config=config, *model_args, **kwargs
        # )
        # return model
        # model = cls(**data[cls.METADATA_KEY], model=data[cls.MODEL_KEY])
        # model = BioGptForCausalLM.from_pretrained("microsoft/BioGPT")
        underlying_model = BioGptForCausalLM.from_pretrained(path)
        # tokenizer = BioGptTokenizer.from_pretrained("microsoft/BioGPT")
        model = cls(path, config=config, *model_args, **kwargs)
        model._model = underlying_model
        return model
        # model._model.load_state_dict(data[cls.MODEL_KEY])

        # model._optimizer.load_state_dict(data[cls.OPTIMIZER_KEY])


if __name__ == "__main__":
    # data
    import json

    from datasets import Dataset, DatasetDict, load_dataset

    def load_data(file: str = "data/know_merge.json"):
        with open(file, "r") as f:
            data = json.load(f)

        data = data[:3]
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

    # model = BioGPT.from_pretrained("microsoft/BioGPT")c
    model = BioGPT.pretrained("causal-models/")
    print(model._model)
    import sys

    sys.exit()
    tokenizer = model.get_tokenizer("causal-models/")
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

    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=["instruction", "label"],
    )
    del dataset

    # ft
    # this is the main entry point when running `python biogpt2.py`
    model.finetune(dataset=tokenized_dataset, collator=data_collator)

    # inf
    # import pipeline

    # generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    # out = generator(
    #     "<Q>Why does my elbow hurt?<A>",
    #     max_length=20,
    #     num_return_sequences=5,
    #     do_sample=True,
    # )
    # print(out)

    # tokenizer = BioGptTokenizer.from_pretrained("causal-models/")
