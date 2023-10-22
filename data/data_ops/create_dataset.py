from datasets import list_datasets, load_dataset
from pprint import pprint


medqa_dataset = load_dataset("bigbio/med_qa")  
medmcqa_dataset = load_dataset("medmcqa")
pubmedqa_dataset = load_dataset("pubmed_qa", "pqa_artificial") 
liveqa_dataset = load_dataset("liveqa")
medicationqa_dataset = load_dataset("truehealth/medicationqa")
medicationqa_dataset = load_dataset("cais/mmlu", "all")