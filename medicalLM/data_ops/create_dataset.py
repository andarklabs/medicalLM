import numpy as np
import pandas as pd
from datasets import load_dataset


''' dropper:
* removes columns from a dataset in a dictionary

* params:
* - key: (str, key of the dictonary of datasets that points us to the right dataset)
* - dropees: (list, list of columns we want to remove from our dataset)
'''
def dropper(key, dropees):
    datasets[key] = datasets[key].drop(dropees, axis=1)


''' remover: 
* removes unwanted tokens from our string

* params
* - string: str
* - unwanted_tokens: (list, list of substrings that we want
*   to remove from our string)
'''
def remover(string, unwanted_tokens):
    string = str(string)
    for token in unwanted_tokens:
<<<<<<< HEAD
        # removes token by replacing it with the empty string
        string = string.replace(token, '')
    string = string.replace('aery','artery')
    string = string.replace('.o', '.')
    string = string.replace('.*', '.')
    string = string.replace('  ', ' ')
    string = string.replace('  ', ' ')
=======
        string = string.replace(token, "")
    string = string.replace("aery", "artery")
    string = string.replace(".o", ".")
    string = string.replace(".*", ".")
    string = string.replace("  ", " ")
    string = string.replace("  ", " ")
>>>>>>> f6af0b1ff8d387d295918a455789e213ad56cba1
    return string


datasets = {}

<<<<<<< HEAD
# populate our dict `datasets` with our hugging face datasets. Each split is being combine before entering our dataset. 
datasets['medqa'] = load_dataset("bigbio/med_qa", split="train+test+validation") # multiple choice with minimal context
datasets['pubmedqa'] = load_dataset("pubmed_qa", "pqa_artificial", split = "train") # yes or no with long anwser (and context)
datasets['medicationqa'] = load_dataset("truehealth/medicationqa", split = "train") # short response given question
datasets['mmlu'] = load_dataset("cais/mmlu", "all", split = "auxiliary_train+test+validation+dev") # selection of multiplechoice questions
datasets['medmcqa'] = load_dataset("medmcqa", split = "train+test+validation") # multiple choice questions with explaination

# human validation data - RL : to be used later in our process
load_dataset("liveqa", split = "train").to_json(path_or_buf=r'data/data_unlabeled.json') # 'passages' only 

=======
datasets["medqa"] = load_dataset(
    "bigbio/med_qa", split="train+test+validation"
)  # multiple choice with minimal context
datasets["pubmedqa"] = load_dataset(
    "pubmed_qa", "pqa_artificial", split="train"
)  # yes or no with long anwser (and context)
datasets["medicationqa"] = load_dataset(
    "truehealth/medicationqa", split="train"
)  # short response given question
datasets["mmlu"] = load_dataset(
    "cais/mmlu", "all", split="auxiliary_train+test+validation+dev"
)  # selection of multiplechoice questions
datasets["medmcqa"] = load_dataset(
    "medmcqa", split="train+test+validation"
)  # multiple choice questions with explaination

# human validation data - RL
# datasets['liveqa'] = load_dataset("liveqa", split = "train") # 'passages' only
>>>>>>> f6af0b1ff8d387d295918a455789e213ad56cba1

for key in datasets.keys():
    datasets[key] = datasets[key].to_pandas()


datasets["pubmedqa"]["final_decision"] = (
    datasets["pubmedqa"]["final_decision"][0][0].upper()
    + datasets["pubmedqa"]["final_decision"][0][1:]
)
datasets["pubmedqa"]["answer"] = (
    datasets["pubmedqa"]["final_decision"] + ". " + datasets["pubmedqa"]["long_answer"]
)

datasets["medicationqa"]["question"] = datasets["medicationqa"]["Question"]
datasets["medicationqa"]["answer"] = datasets["medicationqa"]["Answer"]
datasets["medicationqa"].dropna(subset=["answer"], inplace=True)

mmlu_keepers = [
    "anatomy",
    "clinical_knowledge",
    "college_biology",
    "college_medicine",
    "high_school_biology",
    "high_school_psychology",
    "human_aging",
    "human_sexuality",
    "medical_genetics",
    "nutrition",
    "professional_medicine",
    "professional_psychology",
    "virology",
]
datasets["mmlu"] = datasets["mmlu"][datasets["mmlu"]["subject"].isin(mmlu_keepers)]
datasets["mmlu"]["answer"] = datasets["mmlu"].apply(
    lambda row: row["choices"][row["answer"]], axis=1
)

op_list = ["opa", "opb", "opc", "opd"]
datasets["medmcqa"]["answer"] = datasets["medmcqa"].apply(
    lambda row: row[op_list[row["cop"]]], axis=1
)
datasets["medmcqa"]["answer"] = (
    datasets["medmcqa"]["answer"] + ". " + datasets["medmcqa"]["exp"]
)
unwanted_tokens = [
    "Ans. A ",
    "Ans. B ",
    "Ans. C ",
    "Ans. D ",
    " B ",
    " C ",
    " D ",
    " b ",
    " c ",
    " d ",
    "is 'a'",
    "is 'b'",
    "is 'c'",
    "is 'd'",
    "'a'",
    "'b'",
    "'c'",
    "'d'",
    "is b ",
    "is c ",
    "is d ",
    "(a)",
    "(b)",
    "(c)",
    "(d)",
    "(A)",
    "(B)",
    "(C)",
    "(D)",
    "ans.",
    "Ans.",
    "i.e.",
    "i.e.,",
    "Ref:",
    " , ",
    " :-i)",
    " A. ",
    " B. ",
    " C. ",
    " D. ",
    "Answer- A ",
    "Answer- B ",
    "Answer- C ",
    "Answer- D ",
]
datasets["medmcqa"]["answer"] = datasets["medmcqa"].apply(
    lambda row: remover(row["answer"], unwanted_tokens), axis=1
)

dropper("medqa", ["meta_info", "answer_idx", "options"])
dropper("pubmedqa", ["pubid", "context", "final_decision", "long_answer"])
dropper(
    "medicationqa",
    ["Focus (Drug)", "Question Type", "Section Title", "URL", "Answer", "Question"],
)
dropper("mmlu", ["subject", "choices"])
dropper(
    "medmcqa",
    [
        "id",
        "opa",
        "opb",
        "opc",
        "opd",
        "cop",
        "choice_type",
        "exp",
        "subject_name",
        "topic_name",
    ],
)

dataframe = pd.concat(
    [
        datasets["medqa"],
        datasets["pubmedqa"],
        datasets["medicationqa"],
        datasets["mmlu"],
        datasets["medmcqa"],
    ],
    ignore_index=True,
)

# shuffle our data
dataframe = dataframe.sample(frac=1, random_state=42)

# We use 60% of our data to train our model, 20% to validate, and 20% to test
data_train, data_validate, data_test = np.split(
    dataframe, [int(0.6 * len(dataframe)), int(0.8 * len(dataframe))]
)


data_dict = {"train": data_train, "vaildate": data_validate, "test": data_test}

data_train.to_json(path_or_buf=r"data/data_train.json")
data_validate.to_json(path_or_buf=r"data/data_validate.json")
data_test.to_json(path_or_buf=r"data/data_test.json")

# to turn back to arrows https://www.youtube.com/watch?v=tfcY1067A5Q
