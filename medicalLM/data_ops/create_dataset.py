from datasets import load_dataset
import pandas as pd
import numpy as np


###################### Helper Functions ######################

''' dropper:
* removes columns from a dataset in a dictionary

* params:
* - key: (str, key of the dictonary of datasets that points us to the right dataset)
* - dropees: (list, list of columns we want to remove from our dataset)
'''
def dropper(key, dropees):
    datasets[key] = datasets[key].drop(dropees, axis = 1)

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
        # removes token by replacing it with the empty string
        string = string.replace(token, '')
    string = string.replace('aery','artery')
    string = string.replace('.o', '.')
    string = string.replace('.*', '.')
    string = string.replace('  ', ' ')
    string = string.replace('  ', ' ')
    return string


######################## Data Loading ########################

datasets = {}

# populate our dict `datasets` with our hugging face datasets. Each split is being combine before entering our dataset. 
datasets['medqa'] = load_dataset("bigbio/med_qa", split="train+test+validation") # multiple choice with minimal context
datasets['pubmedqa'] = load_dataset("pubmed_qa", "pqa_artificial", split = "train") # yes or no with long anwser (and context)
datasets['medicationqa'] = load_dataset("truehealth/medicationqa", split = "train") # short response given question
datasets['mmlu'] = load_dataset("cais/mmlu", "all", split = "auxiliary_train+test+validation+dev") # selection of multiplechoice questions
datasets['medmcqa'] = load_dataset("medmcqa", split = "train+test+validation") # multiple choice questions with explaination


######################## Data Cleaning #######################

# turns our mutable datasets into pandas so we can work with them
for key in datasets.keys():
    datasets[key] = datasets[key].to_pandas()

# creating a anwser column that is the concat of the short answer and the explaination
datasets['pubmedqa']['final_decision'] = datasets['pubmedqa']['final_decision'][0][0].upper() \
                                         + datasets['pubmedqa']['final_decision'][0][1:]
datasets['pubmedqa']['answer'] = datasets['pubmedqa']['final_decision'] + '. ' + datasets['pubmedqa']['long_answer']

# create properly named question and anwser columns and drop our na's
datasets['medicationqa']['question'] = datasets['medicationqa']['Question']
datasets['medicationqa']['answer'] = datasets['medicationqa']['Answer']
datasets['medicationqa'].dropna(subset = ['answer'], inplace = True)

# give our model a course list for its mmlu education
mmlu_keepers = ['anatomy', 'clinical_knowledge', 'college_biology', 'college_medicine', 'high_school_biology', \
    'high_school_psychology', 'human_aging', 'human_sexuality', 'medical_genetics', 'nutrition', 'professional_medicine', \
        'professional_psychology', 'virology']
datasets['mmlu'] = datasets['mmlu'][datasets['mmlu']['subject'].isin(mmlu_keepers)]
# properly format mmlu answers
datasets['mmlu']['answer'] = datasets['mmlu'].apply(lambda row: row['choices'][row['answer']], axis = 1)

# turn columns of choices into anwsers
op_list = ['opa', 'opb', 'opc', 'opd']
datasets['medmcqa']['answer'] = datasets['medmcqa'].apply(lambda row: row[op_list[row['cop']]], axis = 1)
datasets['medmcqa']['answer'] = datasets['medmcqa']['answer'] + '. ' + datasets['medmcqa']['exp']
# eliminate unwanted tokens that have to do with the initial multiple choice style
unwanted_tokens = ['Ans. A ', 'Ans. B ', 'Ans. C ', 'Ans. D ', ' B ', ' C ', ' D ', ' b ', ' c ', \
                    ' d ', "is 'a'","is 'b'", "is 'c'","is 'd'", "'a'","'b'", "'c'","'d'", 'is b ', \
                      'is c ', 'is d ', '(a)', '(b)', '(c)', '(d)', '(A)', '(B)', '(C)', '(D)' ,'ans.', \
                        'Ans.', 'i.e.', 'i.e.,',  'Ref:', ' , ', " :-i)", " A. ", " B. ", " C. ", " D. ",  \
                            'Answer- A ', 'Answer- B ', 'Answer- C ', 'Answer- D ']
datasets['medmcqa']['answer'] = datasets['medmcqa'].apply(lambda row: remover(row['answer'], unwanted_tokens), axis = 1)

# drop unwanted columns from our datasets before merging them
dropper('medqa', ['meta_info','answer_idx','options'])
dropper('pubmedqa', ['pubid', 'context', 'final_decision', 'long_answer'])
dropper('medicationqa', ['Focus (Drug)','Question Type','Section Title','URL', 'Answer', 'Question'])
dropper('mmlu', ['subject', 'choices'])
dropper('medmcqa', ['id', 'opa', 'opb', 'opc', 'opd', 'cop', 'choice_type', 'exp', 'subject_name', 'topic_name'])


################### Data Collation/Storage ###################

# merge all datasets into one large dataframe
dataframe = pd.concat([datasets['medqa'], datasets['pubmedqa'], datasets['medicationqa'], datasets['mmlu'], datasets['medmcqa']], ignore_index=True)

# shuffle our data
dataframe = dataframe.sample(frac = 1, random_state = 42)

# We use 60% of our data to train our model, 20% to validate, and 20% to test
data_train, data_validate, data_test = np.split(dataframe, [int(.6*len(dataframe)), int(.8*len(dataframe))])

# convert our data to json files and save the files in our data directory
data_train.to_json(path_or_buf = r'data/data_train.json')
data_validate.to_json(path_or_buf = r'data/data_validate.json')
data_test.to_json(path_or_buf = r'data/data_test.json')

# human validation data - RL : to be used later in our process
load_dataset("liveqa", split = "train").to_json(path_or_buf=r'data/data_unlabeled.json') # 'passages' only 