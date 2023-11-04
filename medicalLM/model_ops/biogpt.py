from os import PathLike
from typing import Any, Final, TypeAlias

import torch
from torch.nn import Module
from torch.optim import Optimizer
from transformers import (AutoConfig, AutoModelForSequenceClassification,
                          AutoTokenizer, BioGptForCausalLM, BioGptTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments,
                          pipeline, set_seed)

# TypeAlias is a type hint that can be used in type hints
FilePath: TypeAlias = PathLike[Any] | str

DEFAULT_DEVICE = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)
DEFAULT_DTYPE = torch.float32


class BioGPT(Module):
    # this allows statically checking later i.e. Diffuser.DEFAULT_BATCH_SIZE
    DEFAULT_TASK_TYPE = TaskType.CAUSAL_LM
    DEFAULT_RANK = 4
    DEFAULT_ALPHA = 16
    DEFAULT_DROPOUT = 0.1
    DEFAULT_TARGET_MODULES = ["q_proj"]
    DEFAULT_PEFT_CONFIG = LoraConfig(
        task_type=DEFAULT_TASK_TYPE,
        inference_mode=False,
        r=DEFAULT_RANK,
        lora_alpha=DEFAULT_ALPHA,
        lora_dropout=DEFAULT_DROPOUT,
        target_modules=DEFAULT_TARGET_MODULES,
    )
    # model = get_peft_model(model, peft_config)
    # model.print_trainable_parameters()

    # training_args = TrainingArguments(
    #     output_dir="causal-models/",
    #     evaluation_strategy="epoch",
    #     learning_rate=2e-5,
    #     weight_decay=0.01,
    #     per_device_train_batch_size=1,
    #     per_device_eval_batch_size=1,
    # does it not see my gpu?
    # fp16=True,  # if this doesn't work try regular size model
    # )

    MODEL_KEY: Final[str] = "model"
    OPTIMIZER_KEY: Final[str] = "optimizer"
    METADATA_KEY: Final[str] = "metadata"

    __slots__ = [
        "_metadata",
        "_model",
        "_optimizer",
        "device",
    ]

    _metadata: Final[dict[str, Any]]
    _model: Final[Module]
    _optimizer: Final[Optimizer]
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

    def __init__(
        self,
        *,
        epochs_trained: int = 0,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        weight_decay: float = DEFAULT_WEIGHT_DECAY,
        batch_size: int = DEFAULT_BATCH_SIZE,
        model: ,
        optimizer: Optimizer,
        device: torch.device = DEFAULT_DEVICE,
        dtype: torch.dtype = DEFAULT_DTYPE,
    ) -> None:
        self._metadata = {
            "epochs_trained": epochs_trained,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "dtype": dtype,
        }

        self._model = model
        self._optimizer = optimizer
        self.device = device

    @property
    def epochs_trained(self) -> int:
        # extra line for type hinting + dot notation access
        epochs_trained: int = self._metadata["epochs_trained"]
        return epochs_trained

    @property
    def learning_rate(self) -> float:
        learning_rate: float = self._metadata["learning_rate"]
        return learning_rate

    @property
    def weight_decay(self) -> float:
        weight_decay: float = self._metadata["weight_decay"]
        return weight_decay

    @property
    def dtype(self) -> torch.dtype:
        dtype: torch.dtype = self._metadata["dtype"]
        return dtype

    def train(
        self,
        dataset: torch.utils.data.Dataset,
        *,
        epochs: int = DEFAULT_TRAINING_EPOCHS,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        raise NotImplementedError

    def save(self, path: FilePath) -> None:
        torch.save(
            {
                self.MODEL_KEY: self._model.state_dict(),
                self.OPTIMIZER_KEY: self._optimizer.state_dict(),
                self.METADATA_KEY: self._metadata,
            },
            path,
        )

    @classmethod
    def load(cls, path: FilePath, *, device: torch.device = DEFAULT_DEVICE) -> "BioGPT":
        data = torch.load(path, map_location=device)
        model = cls(**data[cls.METADATA_KEY], model=data[cls.MODEL_KEY])
        model._model.load_state_dict(data[cls.MODEL_KEY])
        model._optimizer.load_state_dict(data[cls.OPTIMIZER_KEY])
        return model
