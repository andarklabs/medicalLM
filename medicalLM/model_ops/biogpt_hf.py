from os import PathLike

import torch
import transformers

from biogpt import BioGPT


class BioGPT_HF(BioGPT):
    DEFAULT_TASK_TYPE = TaskType.CAUSAL_LM
    DEFAULT_RANK=4
    DEFAULT_ALPHA=16
    DEFAULT_DROPOUT=0.1
    DEFAULT_TARGET_MODULES=["q_proj"]
    DEFAULT_PEFT_CONFIG = LoraConfig(
        task_type=DEFAULT_TASK_TYPE,
        inference_mode=False,
        r=DEFAULT_RANK,
        lora_alpha=DEFAULT_ALPHA,
        lora_dropout=DEFAULT_DROPOUT,
        target_modules=DEFAULT_TARGET_MODULES,
    )

    # OPTIMIZER_KEY: Final[str] = "optimizer"
    # MODEL_KEY: Final[str] = "model"
    # METADATA_KEY: Final[str] = "metadata"

    # _metadata: Final[dict[str, Any]]
    # _model: Final[Module]
    # _optimizer: Final[Optimizer]
    # ) -> None:
    #     self._metadata = {
    #         "epochs_trained": epochs_trained,
    #         "learning_rate": learning_rate,
    #         "weight_decay": weight_decay,
    #         "batch_size": batch_size,
    #         "dtype": dtype,
    #     }

    #     self._model = model
    #     self._optimizer = optimizer
    #     self.device = device
    def __init__(
        self,

        *,


        **kwargs  # This will capture all the existing params for BioGPT
    ) -> None:
        # Initialize the base class with all the parameters it expects
        super().__init__(**kwargs)
        # Now handle the initialization of the new functionality
        self._metadata["extra_optimizer_setting"] = extra_optimizer_setting

    def additional_preprocess_function(self, examples):
        # Implementation of the additional preprocessing steps
        pass  # Replace with actual implementation

    def finetune(self, dataset):j
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
        trainer.train()

    def save(self, path: PathLike):
        model_to_save = (
            trainer.model.module if hasattr(trainer.model, "module") else trainer.model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(path)
        tokenizer.save_pretrained(path)
