# Load model directly
import json
from transformers import AutoTokenizer, AutoModelForCausalLM

class Model():
    def __init__(self, model) -> None:
        f = open('/Users/andrewceniccola/Desktop/medicalLM/medicalLM/frontend/models.json')
        self.mod_data = json.load(f)[model]
        self.tokenizer = AutoTokenizer.from_pretrained(self.mod_data['url'])
        self.model = AutoModelForCausalLM.from_pretrained(self.mod_data['url'])

    def infer(self, prompt, context = ''):
        inputs = self.tokenizer(prompt, return_tensors = "pt", padding = True, truncation=True, max_length=512)
        outputs = self.model.generate(input_ids = inputs['input_ids'], attention_mask = inputs['attention_mask'], max_length=100)
        reply = self.tokenizer.decode(outputs[0])
        return reply
    
mod = Model("biogpt")
print(mod.infer("what do your kidneys do?"))