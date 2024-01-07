import os, json
from transformers import AutoTokenizer
from modules.data_utils import get_conditional_data, conditional_union_data, get_test_data

# CHOOSE THE PARAMS
DATASET_NAME = 'caves' 

#tokenizer = AutoTokenizer.from_pretrained('roberta-large')
tokenizer = AutoTokenizer.from_pretrained('digitalepidemiologylab/covid-twitter-bert-v2')


train= '../CAVES_data/train.csv'
test = '../CAVES_data/test.csv'
val = '../CAVES_data/val.csv'


DATADIR = './data_' + DATASET_NAME 
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
                                    max_span_only=True, 
                                    is_bert=True,
                                    dataset=DATASET_NAME,
                                    use_none=True,
                                    num_neg_samples=5
                                    )

with open(os.path.join(DATADIR, 'train_data.json'), 'w') as f:
    json.dump(train_data, f, indent = 8)




data = get_conditional_data(val, dataset=DATASET_NAME)
val_data = conditional_union_data(data, 
                                    tokenizer,
                                    max_span_only=True, 
                                    is_bert=True,
                                    dataset=DATASET_NAME,
                                    use_none=True,
                                    num_neg_samples=5
                                    )

with open(os.path.join(DATADIR, 'val_data.json'), 'w') as f:
    json.dump(val_data, f, indent = 8)




test_samples = get_test_data(test, # TEST CSV FILE LOCATION
                            tokenizer, 
                            is_testing=True, # ALWAYS SET TO TRUE
                            is_bert=True, # TRUE FOR BERT AND FALSE FOR ROBERTA AND OTHER MODELS.
                            dataset=DATASET_NAME) 

with open(os.path.join(DATADIR, 'test_data.json'), 'w') as f:
    json.dump(test_samples, f, indent = 8)

