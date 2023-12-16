from typing import Any

from transformers import BioGptForCausalLM, BioGptTokenizer, pipeline, set_seed


def infer(
    # model_directory: str = "./causal-models",
    model_directory: str = "causal-models/preprocessed/checkpoint-94500/",
    prompt: str = "What are the 23 amino acids?",
    num_return_sequences: int = 1,
    max_length: int = 200,
) -> Any:
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
    # set model_directory to the dir you download this: https://huggingface.co/arelius/MedicalLM/tree/main
    # at
    out = infer()
    print(out)
