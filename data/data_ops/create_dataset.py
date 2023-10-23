from datasets import load_dataset
from pprint import pprint
import pandas as pd


datasets = {}

#datasets['medqa'] = load_dataset("bigbio/med_qa") # multiple choice with minimal context
#datasets['medmcqa'] = load_dataset("medmcqa") # multiple choice questions with explaination
datasets['pubmedqa'] = load_dataset("pubmed_qa", "pqa_artificial") # yes or no with long anwser (and context)
#datasets['liveqa'] = load_dataset("liveqa") # 'passages' only 
#datasets['medicationqa'] = load_dataset("truehealth/medicationqa") # short response given question
#datasets['mmlu'] = load_dataset("cais/mmlu", "all") # selection of multiplechoice questions


datasets['pubmedqa']['train'].set_format("pandas")
print(len(datasets['pubmedqa']['train']))
