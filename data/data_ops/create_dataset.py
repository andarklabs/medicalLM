from datasets import load_dataset
from pprint import pprint
import pandas as pd
from transformers import AutoTokenizer

datasets = {}

datasets['medqa'] = load_dataset("bigbio/med_qa") # multiple choice with minimal context
datasets['medmcqa'] = load_dataset("medmcqa") # multiple choice questions with explaination
datasets['pubmedqa'] = load_dataset("pubmed_qa", "pqa_artificial") # yes or no with long anwser (and context)
datasets['liveqa'] = load_dataset("liveqa") # 'passages' only 
datasets['medicationqa'] = load_dataset("truehealth/medicationqa") # short response given question
datasets['mmlu'] = load_dataset("cais/mmlu", "all") # selection of multiplechoice questions

datasets['medqa']['train'].set_format("pandas")
datasets['medqa']['test'].set_format("pandas")
datasets['medqa']['validation'].set_format("pandas")

datasets['medmcqa']['train'].set_format("pandas")
datasets['medmcqa']['test'].set_format("pandas")
datasets['medmcqa']['validation'].set_format("pandas")

datasets['pubmedqa']['train'].set_format("pandas")

datasets['liveqa']['train'].set_format("pandas")

datasets['medicationqa']['train'].set_format("pandas")

datasets['mmlu']['auxiliary_train'].set_format("pandas")
datasets['mmlu']['test'].set_format("pandas")
datasets['mmlu']['validation'].set_format("pandas")
datasets['mmlu']['dev'].set_format("pandas")


