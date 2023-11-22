from transformers import BioGptForCausalLM, BioGptTokenizer, pipeline, set_seed

model = BioGptForCausalLM.from_pretrained("microsoft/BioGPT")
tokenizer = BioGptTokenizer.from_pretrained("microsoft/BioGPT")
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
set_seed(42)
out = generator(
    "<Q>Why does my elbow hurt?<A>",
    max_length=180,
    num_return_sequences=5,
    do_sample=True,
)

print(out)


# TODO: add transformers to requirements.txt
# TODO: rip apart what I want to be able to get the tokenizer, model loader, and generator so that
# TODO: add the bin and whatever it takes to open without dependency to our medicalLM app
# so we can use in an app with no dependency on hf
