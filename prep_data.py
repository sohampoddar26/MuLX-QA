import os, json, re
from transformers import AutoTokenizer
from modules.data_utils import get_conditional_data, conditional_union_data, get_test_data

# CHOOSE THE PARAMS
DATASET_NAME = 'caves' 

# tokenizer = AutoTokenizer.from_pretrained('roberta-base')
tokenizer = AutoTokenizer.from_pretrained('digitalepidemiologylab/covid-twitter-bert-v2')


train = '../caves-data/labelled_tweets/json_labels_explanations/train.json'
test = '../caves-data/labelled_tweets/json_labels_explanations/test.json'
val = '../caves-data/labelled_tweets/json_labels_explanations/val.json'


DATADIR = './data_ctb5_' + DATASET_NAME 

if not os.path.exists(DATADIR):
    print("Directory created:", DATADIR)
    os.makedirs(DATADIR)



data = get_conditional_data(train, dataset=DATASET_NAME)

# max_span_only: 'whether to consider the maximum span length as explanation' , True (default)
# is_bert: True for 'bert' model including the epidemiology model, False for 'roberta' and other mdoels
# use_none: Only applicable for 'caves' dataset otherwise False, if True then one of the negative/contrasting samples always contains 'none' 
# num_neg_samples: Number of negative examples per samples to generate, Default = 2 

train_data = conditional_union_data(data, 
                                    tokenizer,
                                    dataset=DATASET_NAME,
                                    use_none=True,
                                    num_neg_samples=5
                                    )

with open(os.path.join(DATADIR, 'train_data.json'), 'w') as f:
    tmp = json.dumps(train_data, indent = 8)
    pos = tmp.index('"text":')
    print(re.sub(r'(\d+,)\s+', r'\1 ', tmp[:pos]) + tmp[pos:] , file = f)





data = get_conditional_data(val, dataset=DATASET_NAME)
val_data = conditional_union_data(data, 
                                    tokenizer,
                                    dataset=DATASET_NAME,
                                    use_none=True,
                                    num_neg_samples=5
                                    )

with open(os.path.join(DATADIR, 'val_data.json'), 'w') as f:
    tmp = json.dumps(val_data, indent = 8)
    pos = tmp.index('"text":')
    print(re.sub(r'(\d+,)\s+', r'\1 ', tmp[:pos]) + tmp[pos:] , file = f)





test_samples = get_test_data(test, # TEST CSV FILE LOCATION
                            tokenizer, 
                            dataset=DATASET_NAME) 

with open(os.path.join(DATADIR, 'test_data.json'), 'w') as f:
    tmp = json.dumps(test_samples, indent = 8)
    print(re.sub(r'(\d+,)\s+', r'\1 ', tmp) , file = f)

