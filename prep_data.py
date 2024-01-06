import os, json
from transformers import AutoTokenizer
from modules.data_utils import get_conditional_data, conditional_union_data

# CHOOSE THE PARAMS
DATASET_NAME = 'caves' 

#tokenizer = AutoTokenizer.from_pretrained('roberta-large')
tokenizer = AutoTokenizer.from_pretrained('digitalepidemiologylab/covid-twitter-bert-v2')


train= '../CAVES_data/train.csv'
test = '../CAVES_data/test.csv'
val = '../CAVES_data/val.csv'


TRAINDIR = './data_' + DATASET_NAME + '/train/'
VALDIR = './data_' + DATASET_NAME + '/val/'
os.makedirs(TRAINDIR)
os.makedirs(VALDIR)



data = get_conditional_data(train, dataset=DATASET_NAME)

# max_span_only: 'whether to consider the maximum span length as explanation' , True (default)
# is_bert: True for 'bert' model including the epidemiology model, False for 'roberta' and other mdoels
# use_none: Only applicable for 'caves' dataset otherwise False, if True then one of the negative/contrasting samples always contains 'none' 
# num_neg_samples: Number of negative examples per samples to generate, Default = 2 

tokenized_tweet_with_question, tweet_text, tweet_labels, tokenized_keywords, start_token, end_token = conditional_union_data(data, 
                                                                                                                             tokenizer,
                                                                                                                             max_span_only=True, 
                                                                                                                             is_bert=True,
                                                                                                                             dataset=DATASET_NAME,
                                                                                                                             use_none=True,
                                                                                                                             num_neg_samples=5,
                                                                                                                             joint_qna=True)
with open(os.path.join(TRAINDIR, 'train_labels.json'), 'w') as f:
    json.dump(tweet_labels, f)

with open(os.path.join(TRAINDIR, 'train_keywords.json'), 'w') as f:
    json.dump(tokenized_keywords, f)

with open(os.path.join(TRAINDIR, 'train_start_tokens.json'), 'w') as f:
    json.dump(start_token, f)

with open(os.path.join(TRAINDIR, 'train_end_tokens.json'), 'w') as f:
    json.dump(end_token, f)

with open(os.path.join(TRAINDIR, 'train_tokenized_twt.json'), 'w') as f:
    json.dump(tokenized_tweet_with_question, f)


data = get_conditional_data(val, dataset=DATASET_NAME)
tokenized_tweet_with_question, tweet_text, tweet_labels, tokenized_keywords, start_token, end_token = conditional_union_data(data, 
                                                                                                                             tokenizer,
                                                                                                                             max_span_only=True, 
                                                                                                                             is_bert=True,
                                                                                                                             dataset=DATASET_NAME,
                                                                                                                             use_none=True,
                                                                                                                             num_neg_samples=5,
                                                                                                                             joint_qna=True)
with open(os.path.join(VALDIR, 'val_labels.json'), 'w') as f:
    json.dump(tweet_labels, f)

with open(os.path.join(VALDIR, 'val_keywords.json'), 'w') as f:
    json.dump(tokenized_keywords, f)

with open(os.path.join(VALDIR, 'val_start_tokens.json'), 'w') as f:
    json.dump(start_token, f)

with open(os.path.join(VALDIR, 'val_end_tokens.json'), 'w') as f:
    json.dump(end_token, f)

with open(os.path.join(VALDIR, 'val_tokenized_twt.json'), 'w') as f:
    json.dump(tokenized_tweet_with_question, f)

