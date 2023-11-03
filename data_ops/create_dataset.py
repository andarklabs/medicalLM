from datasets import load_dataset
from pprint import pprint
import pandas as pd
from transformers import AutoTokenizer
import pyarrow

def dropper(key, dropees):
    datasets[key] = datasets[key].drop(dropees, axis = 1)

datasets = {}

datasets['medqa'] = load_dataset("bigbio/med_qa", split="train+test+validation") # multiple choice with minimal context
datasets['pubmedqa'] = load_dataset("pubmed_qa", "pqa_artificial", split = "train") # yes or no with long anwser (and context)
datasets['medicationqa'] = load_dataset("truehealth/medicationqa", split = "train") # short response given question
#datasets['mmlu'] = load_dataset("cais/mmlu", "all", split = "auxiliary_train+test+validation+dev") # selection of multiplechoice questions
#datasets['medmcqa'] = load_dataset("medmcqa", split = "train+test+validation") # multiple choice questions with explaination
#datasets['liveqa'] = load_dataset("liveqa", split = "train") # 'passages' only 

for key in datasets.keys():
    datasets[key] = datasets[key].to_pandas()
    
dropper('medqa', ['meta_info','answer_idx','options'])
dropper('pubmedqa', ['pubid', 'context'])
dropper('medicationqa', ['Focus (Drug)','Question Type','Section Title','URL'])
datasets['medicationqa'].dropna(subset = ['Answer'], inplace = True)


for key in datasets.keys():
    print(datasets[key].info())

print(datasets)

# to turn back to arrows https://www.youtube.com/watch?v=tfcY1067A5Q