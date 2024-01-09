import pandas as pd
import numpy as np
import re
import json
import html
import itertools
import random
import torch
#from keras.preprocessing.sequence import pad_sequences
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler




def get_question(lab):
    #return f'Why {lab} ?'
    #return f'How {lab} ?'
    #return f'{lab}'
    return f'Why is "{lab}" a reason for not taking vaccines?'
    



def tokenize_and_map(text_words, tokenizer):
    
    assert text_words[0] == tokenizer.cls_token
    tokenized_text = [tokenizer.cls_token_id]
    i = 1
    idx_map = []

    for j, word in enumerate(text_words[1:]):
        word = re.sub("https://\S+", "HTTPURL", word)

        encoded_words = tokenizer.encode(word, add_special_tokens = False)
        tokenized_text += encoded_words

        idx_map.append(i)
        i += len(encoded_words)

    return tokenized_text, idx_map
        



def get_conditional_data(filename, dataset):
    with open(filename) as fp:
        json_data = json.load(fp)
    
    
    data = []
    for row in json_data: 
        temp = {}

        temp['text'] = row['text_tokens']
        temp['labels'] = []
        temp['tuples'] = []
        
        for lab, spans in row['labels'].items():
            temp['labels'].append(lab)
            temp['tuples'].append([lab, spans])
        
        data.append(temp)
    return data


def conditional_union_data(data:list, 
                           tokenizer,
                        #    max_span_only=True, 
                        #    is_bert=False,
                           num_neg_samples=2,
                           dataset='caves',
                           use_none=False
                           ):

    if dataset == 'caves':
        ALL_LABELS = np.array(['none'] + sorted(['unnecessary', 'side-effect', 'ineffective', 'mandatory', 'pharma', 'political', 'conspiracy', 'rushed', 'country', 'ingredients', 'religious']))
    elif dataset =='hatexplain':
        ALL_LABELS = np.array(['normal', 'hatespeech', 'offensive'])
    else:
        ALL_LABELS = np.array(['positive', 'neutral', 'negative'])

    tokenized_tweet_with_question, tweet_text, tweet_labels, tokenized_keywords, start_token, end_token = [], [], [], [], [], []
    
    for idx, dat in enumerate(data):

        temp = [tokenizer.cls_token] + dat['text'] + [tokenizer.unk_token, tokenizer.sep_token]

        tweet = ' '.join(temp) 
        labels = dat['labels']
        tokenized_tweet, idx_map = tokenize_and_map(temp, tokenizer)

        null_token_id = idx_map[len(dat['text'])]
        
        
        if labels[0] != 'none' and labels[0] != 'normal':           
            
            original_tokenized_tweet = tokenized_tweet.copy()
            tokenized_tweet = tokenized_tweet.copy()
            
            for lab in labels:
                
                for tup in dat['tuples']:
                    if tup[0] != lab:
                        continue

                    # print(tup)

                    question = get_question(lab)
                    # question = f'Why {lab} ?'
                    question = tokenizer(question).input_ids[1:-1]

                    for span in tup[1]:
                        tokenized_tweet.extend(question)
                        
                        tweet_text.append(tokenizer.decode(tokenized_tweet))
                        
                        tokenized_tweet_with_question.append(tokenized_tweet)
                        
                        
                        tokenized_keywords.append(tokenizer.decode(tokenized_tweet[idx_map[span['start']]:idx_map[span['end']]]))
                        # if re.sub("[^\w\d]", "", tokenized_keywords[-1].lower()) != re.sub("[^\w\d]", "", span["terms"].lower()) and '[UNK]' not in tokenized_keywords:
                        #     print(tokenized_keywords[-1] + "\n" + span["terms"], end = "\n\n")
                        
                        tweet_labels.append(lab)
                        
                        start_token.append(idx_map[span['start']])
                        end_token.append(idx_map[span['end']] - 1)
                        
                        tokenized_tweet = original_tokenized_tweet.copy()


            if dataset == 'caves' and use_none:
                negative_labels = ['none']
            else:
                negative_labels = []
            
            if dataset == 'caves':
                if use_none:
                    labels_ = [i for i in ALL_LABELS if i!='none' and i not in labels]
                else:
                    labels_ = [i for i in ALL_LABELS if i not in labels]
            else:
                labels_ = [i for i in ALL_LABELS if i not in labels]

            if use_none:
                negative_labels.extend(np.random.choice(labels_, num_neg_samples - 1, False).tolist())
            else:
                negative_labels.extend(np.random.choice(labels_, num_neg_samples, False).tolist())
            
            for neg in negative_labels:
                question = get_question(neg)
                # question = f'Why {neg} ?'

                question = tokenizer(question).input_ids[1:-1]
                
                tokenized_tweet.extend(question)
                
                tweet_text.append(tokenizer.decode(tokenized_tweet))
                
                tokenized_tweet_with_question.append(tokenized_tweet)
                
                tokenized_keywords.append(tokenizer.decode(tokenized_tweet[null_token_id: null_token_id + 1]))
                
                tweet_labels.append(neg)
                
                start_token.append(null_token_id)
                end_token.append(null_token_id)
                
                tokenized_tweet = original_tokenized_tweet.copy()
        else:
            original_tokenized_tweet = tokenized_tweet.copy()
            tokenized_tweet = tokenized_tweet.copy()
            
            if dataset == 'caves':
                question = get_question('none')
                
            else:
                question = get_question('normal')
                

            question = tokenizer(question).input_ids[1:-1]
            
            tokenized_tweet.extend(question)
            
            tweet_text.append(tokenizer.decode(tokenized_tweet))
            
            tokenized_tweet_with_question.append(tokenized_tweet)
            
            tokenized_keywords.append(tokenizer.decode(tokenized_tweet[0: 1]))
            
            tweet_labels.append(labels[0])
            
            start_token.append(0)
            end_token.append(0)
            
            tokenized_tweet = original_tokenized_tweet.copy()
            
            if dataset == 'caves':
                labels_ = [i for i in ALL_LABELS if i!='none']
            else:
                labels_ = [i for i in ALL_LABELS if i not in labels]
            
            negative_labels = np.random.choice(labels_, num_neg_samples, False)
            
            for neg in negative_labels:
                question = get_question(neg)
                
                question = tokenizer(question).input_ids[1:-1]
                
                tokenized_tweet.extend(question)
                
                tweet_text.append(tokenizer.decode(tokenized_tweet))
                
                tokenized_tweet_with_question.append(tokenized_tweet)
                
                tokenized_keywords.append(tokenizer.decode(tokenized_tweet[null_token_id: null_token_id + 1]))
                
                tweet_labels.append(neg)
                
                start_token.append(null_token_id)
                end_token.append(null_token_id)
                
                tokenized_tweet = original_tokenized_tweet.copy()

    
    json_data = {'tokenized_text': tokenized_tweet_with_question, 'text': tweet_text, 'labels': tweet_labels, 
                 'tokenized_keywords': tokenized_keywords, 'start_tokens': start_token, 'end_tokens': end_token}
    return json_data





def get_test_data(testfile, 
                       tokenizer,  
                    #    is_bert=False, 
                       data=None,
                       dataset='caves',
                       ALL_LABELS=None):
    
    if not data:
        data = get_conditional_data(testfile, dataset)
    else:
        pass

    if dataset == 'caves':
        ALL_LABELS = np.array(['none'] + sorted(['unnecessary', 'side-effect', 'ineffective', 'mandatory', 'pharma', 'political', 'conspiracy', 'rushed', 'country', 'ingredients', 'religious']))
    elif dataset =='hatexplain':
        ALL_LABELS = np.array(['normal', 'hatespeech', 'offensive'])
    else:
        ALL_LABELS = np.array(['positive', 'neutral', 'negative'])
    
    total_samples = []

    for dat in data:
        
        temp = [tokenizer.cls_token] + dat['text'] + [tokenizer.unk_token, tokenizer.sep_token]

        tweet = ' '.join(temp) 
        labels = dat['labels']
        tokenized_tweet, idx_map = tokenize_and_map(temp, tokenizer)

        null_token_id = idx_map[len(dat['text'])]
        

        original_tokenized_tweet = tokenized_tweet.copy()
        tokenized_tweet = tokenized_tweet.copy()
        
        questions_per_sample = {}
        questions_per_sample['tokenized_question'] = []
        questions_per_sample['question_text'] = []
        questions_per_sample['gt_labels'] = []
        questions_per_sample['gt_keywords'] = []
        questions_per_sample['labels'] = []
        questions_per_sample['start_token'] = None
        questions_per_sample['end_token'] = None

        for lab in ALL_LABELS:
            question = get_question(lab)
            # question = f'Why {lab} ?'

            question = tokenizer(question).input_ids[1:-1]
            
            ## if lab is actually a ground truth label
            if lab in labels and (lab != 'none' and lab != 'normal'):
                
                    for tup in dat['tuples']:
                        if tup[0] != lab:
                            continue

                        union_key = []
                        for span in tup[1]:
                            union_key.extend(tokenized_tweet[idx_map[span['start']]: idx_map[span['end']]])
                            
                        questions_per_sample['gt_keywords'].append(tokenizer.decode(union_key))    
                            
                        tokenized_tweet.extend(question)
                        questions_per_sample['question_text'].append(tokenizer.decode(tokenized_tweet))
                        questions_per_sample['tokenized_question'].append(tokenized_tweet)
                        questions_per_sample['labels'].append(lab)
                        questions_per_sample['gt_labels'].append(lab)
                        tokenized_tweet = original_tokenized_tweet.copy()
        
            elif lab in labels and (lab == 'none' or lab == 'normal'):
                questions_per_sample['gt_keywords'].append('')
                tokenized_tweet.extend(question)
                questions_per_sample['question_text'].append(tokenizer.decode(tokenized_tweet))
                questions_per_sample['tokenized_question'].append(tokenized_tweet)
                questions_per_sample['labels'].append(lab)
                questions_per_sample['gt_labels'].append(lab)
                
               
                tokenized_tweet = original_tokenized_tweet.copy()
            
            else: # when framing contrasting questions
                tokenized_tweet.extend(question)
                questions_per_sample['question_text'].append(tokenizer.decode(tokenized_tweet))
                questions_per_sample['tokenized_question'].append(tokenized_tweet)
                questions_per_sample['labels'].append(lab)
                tokenized_tweet = original_tokenized_tweet.copy()
      
        total_samples.append(questions_per_sample)             

    return total_samples




# %% CUSTOM DATALOADER


def attention_vector(tweets, attention):
    corrected = []
    for twt, attn in zip(tweets, attention):
        temp = [i for i in attn]
        max = len(twt)-len(attn)
        for i in range(0, max):
            temp.append(0)
        corrected.append(temp)
    return  torch.tensor(corrected)

def dataloader(tweets, keywords=None, start_tokens=None, end_tokens=None, tokenizer=None, params=None, is_train=False):
    
    attn = [[1 for i in range(len(tweets[j]))] for j in range(len(tweets))]
    
    tweets = pad_sequence([torch.LongTensor(x) for x in tweets], batch_first = True, padding_value = tokenizer.pad_token_id)
    
    keywords_final = []
    
    if keywords is not None:
        
        for k in keywords:
            
            if len(k) == 0 or k =='':
                keywords_final.append([-1])
            else:
                keywords_final.append(k)
        
        keywords_final = pad_sequence([torch.LongTensor(x) for x in keywords_final], batch_first = True, padding_value = tokenizer.pad_token_id)
        
        keywords_final = torch.tensor(keywords_final)
    
    attn_mask = attention_vector(tweets, attn)
    
    tweets = torch.tensor(tweets)
    
    if start_tokens is not None:
        
        start_tokens = torch.tensor(start_tokens).unsqueeze(1)
        
        end_tokens = torch.tensor(end_tokens).unsqueeze(1)
        
        print(f'tweet shape:{tweets.shape} \n attn_mask:{attn_mask.shape} \n start_tokens:{start_tokens.shape} \n end_tokens:{end_tokens.shape}')
        
        dataset = TensorDataset(tweets,
                                attn_mask, 
                                start_tokens,
                                end_tokens)
    else:
        dataset = TensorDataset(tweets,
                                attn_mask, 
                                keywords_final)
    
    if is_train:
        sampler = RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)

    dataloader = DataLoader(dataset, sampler=sampler, batch_size=params['batch_size'])

    return dataloader
