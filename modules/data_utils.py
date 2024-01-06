import pandas as pd
import numpy as np
import re
import json
import html
import itertools
import torch
# from keras.preprocessing.sequence import pad_sequences
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

# %% DATA PREP


def keyword_pos(text, keywords):
    """
    text: list of words
    keywords: list of words (from one annotator only)
    
    returns numpy array of size len(text), with 1s where the keywords are, and 0 elsewhere
    """
    assert type(text) is list and type(keywords) is list

    keypairs = []
    i = 0
    while i < len(keywords):
        j = -1
        # find all places in text where this keyword is present, and try to get max keywords matching from there
        while True:
            try:
                j = text.index(keywords[i], j + 1)
                l = 0
                while keywords[i + l] == text[j + l]:
                    l += 1                                        
                    if (i + l) >= len(keywords) or (j + l) >= len(text):
                        break
                keypairs.append([i, j, l])
            except ValueError:
                    break
        i += 1
    keypairs.sort(key = lambda x: x[2], reverse = True)
    # form final vector
    # assumption: if the starting word in a keyset is not yet selected, it was not part of any larger keysets
    vector = np.zeros(len(text))
    remaining = np.ones(len(keywords))
    
    for i, j, l in keypairs:
            if remaining[i]:
                    remaining[i : i + l] = 0
                    vector[j : j + l] = 1

    return vector


def span_finder(highlights, use_only_max_length=False):
    spans = []
    for val in highlights:
        sample_span = []
        if len(val)!=0:
            for tup in val:
                t = tup[1].tolist()
                start_idx = 0
                indices = []
                while start_idx < len(t):
                    l = 1
                    try:
                        start_idx = t.index(1, start_idx)
                        while True:
                            if start_idx+l < len(t) and t[start_idx+l] == 1:
                                l+=1
                            else:
                                end_idx = start_idx + l
                                indices.append((start_idx, end_idx))
                                start_idx = end_idx
                                break
                    except:
                        if start_idx == 0:
                            indices.append((0,0))
                        break
                if use_only_max_length:
                    max_length_index = np.argmax([tup[1]-tup[0] for tup in indices])
                    indices = [indices[max_length_index]]
                sample_span.append(indices)
            spans.append(sample_span)
        else:
            indices = [(0,0)]
            sample_span.append(indices)
            spans.append(sample_span)
    return spans


def keyword_position(keywords, tokenizer):
    new_keywords = []
    for keylist in keywords:
        temp1 = [] # temp1  = [[k1_l1, k2_l1], [k1_l2, k2_l2]]
        if len(keylist)!=0:
            for key in keylist:
                keys = key.split(';')
                processed_keys = [preprocess_and_tokenize(k, tokenizer)[1:-1] for k in keys]
                temp1.append(processed_keys)
        else:
            pass
        new_keywords.append(temp1)
    return new_keywords


# In[ ]:


def preprocess(txt):
    # print(txt)
    txt = html.unescape(txt.lower())
    txt = re.sub("https://\S+", "HTTPURL", txt)
    
    #removing everything except alphanumeric and spaces
    txt = re.sub('[^\w\s]', ' ', txt)
    txt = re.sub('\s+', ' ', txt)
    
    return txt

def preprocess_and_tokenize(txt, tokenizer):
    # print(txt)
    txt = html.unescape(txt.lower())
    txt = re.sub("https://\S+", "URL", txt)
    
    #removing everything except alphanumeric and spaces
    txt = re.sub('[^\w\s]', ' ', txt)
    txt = re.sub('\s+', ' ', txt)
    
    return tokenizer(txt).input_ids


def get_label_highlight(text_ids, keywords, labels, ALL_LABELS):        
    highlights = []
    # for each data point / tweet        
    for txt, key, lab in zip(text_ids, keywords, labels):
        lab_key = {}    
        # for each label in a data point
        for k, l in zip(key, lab):
            veclist = []         
            # keywords for different annotators
            for x in k:
                tmp = keyword_pos(txt, x)
                veclist.append(tmp)                  
            # COMBINE vectors
            vector = np.zeros(len(txt))
            for vec in veclist:
                  vector = vector + vec                   
            #tot = len(veclist) if len(veclist) == 1 or l in ['none'] else 1
            tot = len(veclist) if len(veclist) == 1 else 1
            lab_key[l] = (vector >= tot)             
        tmp = [(l, 1*lab_key[l].astype(bool)) for l in ALL_LABELS if l in lab_key]
        highlights.append(tmp)
    
    return highlights


# In[]

def fix_tokenizer_issue(keywords):
    corrected_keys = []
    for keys in keywords:
        temp = []
        for k in keys:
            #if k is not np.nan:
            m = ' '
            m += k
            temp.append(m)
            #else:
                #temp.append(None)
        corrected_keys.append(temp)
    return corrected_keys

def get_conditional_data(filename, dataset):
    
    df = pd.read_csv(filename)
    
    if dataset == 'caves':
        labels = df[['label1', 'label2', 'label3']].values.tolist()
        labels = [[i for i in sample_labels if i is not np.nan] for sample_labels in labels]
    else:
        labels = df['label'].tolist()
        labels = [[i] for i in labels]
    
    if dataset == 'caves':
        keywords = df[['Keyword for Sentiment 1', 'Keyword for Sentiment 2', 'Keyword for Sentiment 3']].values.tolist()
        keywords = [[i for i in sublist if i is not np.nan] for sublist in keywords]
    else:
        keywords = df['keyword'].tolist()
        keywords = [[i] if i is not np.nan else [] for i in keywords]
    
    keywords = fix_tokenizer_issue(keywords)
    if dataset == 'caves':
        tweets = df['temp'].tolist()
    else: 
        tweets = df['temp'].tolist()
    
    if dataset == '14lap':
        aspects = df['aspect'].tolist()
    else:
        aspects = [None for i in range(len(tweets))]

    data = []
    for lab, key, twt, a in zip(labels, keywords, tweets, aspects):
        temp = {}
        temp['tweet_text'] = twt
        temp['keyword'] = key
        temp['labels'] = lab
        if a is not None:
            temp['aspect'] = a
        data.append(temp)
    return data

def conditional_union_data(data:list, 
                           tokenizer,
                           max_span_only=True, 
                           is_bert=False,
                           num_neg_samples=2,
                           dataset='14lap',
                           use_none=False,
                           joint_qna=True):
    '''
    data: list of dict with keys = ['tweet_text', 'spans', 'lables']
    Given lbl1 and lbl2 are valid reasons, why is lbl3 also a reason for not taking vaccines?
    Given lbl1 is reason, why is lbl2 also a reason for not taking vaccines?
    '''

    if dataset == 'caves':
        ALL_LABELS = np.array(['none'] + sorted(['unnecessary', 'side-effect', 'ineffective', 'mandatory', 'pharma', 'political', 'conspiracy', 'rushed', 'country', 'ingredients', 'religious']))
    elif dataset =='hatexplain':
        ALL_LABELS = np.array(['normal', 'hatespeech', 'offensive'])
    else:
        ALL_LABELS = np.array(['positive', 'neutral', 'negative'])

    tokenized_tweet_with_question, tweet_text, tweet_labels, tokenized_keywords, start_token, end_token = [], [], [], [], [], []
    total_highlights, keyword_spans = [], []
    
    for idx, dat in enumerate(data):
        
        tweet = ' '
        tweet += dat['tweet_text']
        
        tokenized_tweet = [preprocess_and_tokenize(tweet, tokenizer)]
        tokenized_tweet = [tokenized_tweet[0][:-1]]
        
        keys = [dat['keyword']]
        keys = keyword_position(keys, tokenizer)
        
        labels = dat['labels']

        if dataset == '14lap':
            aspect = dat['aspect']
        else:
            aspect = None

        highlights = get_label_highlight(tokenized_tweet, keys, [labels], ALL_LABELS)

        keyword_spans = span_finder(highlights, use_only_max_length=max_span_only)

        if not is_bert:
            tweet += ' <unk>'
            tweet += '</s>'
            tokenized_tweet[0].extend([1437,3,2]) # 1437=' ', <unk>=3, </s>=2
            null_token_id = len(tokenized_tweet[0]) - 2
        else:
            tweet += ' [UNK] ' # token_id = 100
            tweet += '[SEP]' # token_id = 102
            tokenized_tweet[0].extend([100, 102])
            null_token_id = len(tokenized_tweet[0]) - 2

        if len(labels)==3: #[political, country, mandatory]
            
            original_tokenized_tweet = tokenized_tweet[0].copy()
            tokenized_tweet = tokenized_tweet[0].copy()

            if joint_qna:
                label_combos = [comb for comb in itertools.combinations(labels, 2)] # [(political, mandatory), (country, mandatory), (political, country)]
                
                for tuples in label_combos: 
                    
                    given_labels = list(tuples) # [political, mandatory]
                    
                    label_for_prediction = [i for i in labels if i not in given_labels] # country
                    
                    for high, spans in zip(highlights, keyword_spans):
                        
                        for tup, span in zip(high, spans):
                            
                            if tup[0] == label_for_prediction[0]:
                                
                                question = f'Given \"{given_labels[0]}\" and \"{given_labels[1]}\" are valid reasons, why is \"{label_for_prediction[0]}\" also a reason for not taking vaccines?'
                                question = tokenizer(question).input_ids[1:-1]
                                
                                tokenized_tweet.extend(question)
                                
                                tweet_text.append(tokenizer.decode(tokenized_tweet))
                                
                                tokenized_tweet_with_question.append(tokenized_tweet)
                                
                                tokenized_keywords.append(tokenizer.decode(tokenized_tweet[span[0][0]:span[0][1]]))
                                
                                tweet_labels.append(label_for_prediction[0])
                                
                                start_token.append(span[0][0])
                                end_token.append(span[0][1])
                                
                                tokenized_tweet = original_tokenized_tweet.copy()
            else:
                for lab in labels:
                    for high, spans in zip(highlights, keyword_spans):
                        for tup, span in zip(high, spans):
                            if tup[0] == lab:
                                question = f'Why is \"{lab}\" a reason for not taking vaccines?'
                                question = tokenizer(question).input_ids[1:-1]
                                
                                tokenized_tweet.extend(question)
                                            
                                tweet_text.append(tokenizer.decode(tokenized_tweet))
                                
                                tokenized_tweet_with_question.append(tokenized_tweet)
                                
                                tokenized_keywords.append(tokenizer.decode(tokenized_tweet[span[0][0]:span[0][1]]))
                                
                                tweet_labels.append(lab)
                                
                                start_token.append(span[0][0])
                                end_token.append(span[0][1])
                                
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
                
                question = f'Why is \"{neg}\" a reason for not taking vaccines?'
                question = tokenizer(question).input_ids[1:-1]
                
                tokenized_tweet.extend(question)
                
                tweet_text.append(tokenizer.decode(tokenized_tweet))
                
                tokenized_tweet_with_question.append(tokenized_tweet)
                
                tokenized_keywords.append('')
                
                tweet_labels.append(neg)
                
                start_token.append(null_token_id)
                end_token.append(null_token_id)
                
                tokenized_tweet = original_tokenized_tweet.copy()
        
        elif len(labels)==2:
            
            original_tokenized_tweet = tokenized_tweet[0].copy()
            tokenized_tweet = tokenized_tweet[0].copy()
            
            if joint_qna:
                for lab in labels:
                    
                    given_label = [i for i in labels if i!=lab]
                    
                    for high, spans in zip(highlights, keyword_spans):
                        
                        for tup, span in zip(high, spans):
                            
                            if tup[0] == lab:
                                
                                question = f'Given \"{given_label[0]}\" is a valid reason, why is \"{lab}\" also a reason for not taking vaccines?'
                                question = tokenizer(question).input_ids[1:-1]
                                
                                tokenized_tweet.extend(question)
                                
                                tweet_text.append(tokenizer.decode(tokenized_tweet))
                                
                                tokenized_tweet_with_question.append(tokenized_tweet)
                                
                                tokenized_keywords.append(tokenizer.decode(tokenized_tweet[span[0][0]:span[0][1]]))
                                
                                tweet_labels.append(lab)
                                
                                start_token.append(span[0][0])
                                end_token.append(span[0][1])
                                
                                tokenized_tweet = original_tokenized_tweet.copy()
            else:
                for lab in labels:
                    for high, spans in zip(highlights, keyword_spans):
                        for tup, span in zip(high, spans):
                            if tup[0] == lab:
                                question = f'Why is \"{lab}\" a reason for not taking vaccines?'
                                question = tokenizer(question).input_ids[1:-1]
                                
                                tokenized_tweet.extend(question)
                                            
                                tweet_text.append(tokenizer.decode(tokenized_tweet))
                                
                                tokenized_tweet_with_question.append(tokenized_tweet)
                                
                                tokenized_keywords.append(tokenizer.decode(tokenized_tweet[span[0][0]:span[0][1]]))
                                
                                tweet_labels.append(lab)
                                
                                start_token.append(span[0][0])
                                end_token.append(span[0][1])
                                
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
                
                question = f'Why is \"{neg}\" a reason for not taking vaccines?'
                question = tokenizer(question).input_ids[1:-1]
                
                tokenized_tweet.extend(question)
                
                tweet_text.append(tokenizer.decode(tokenized_tweet))
                tokenized_tweet_with_question.append(tokenized_tweet)
                
                tokenized_keywords.append('')
                
                tweet_labels.append(neg)
                
                start_token.append(null_token_id)
                end_token.append(null_token_id)
                
                tokenized_tweet = original_tokenized_tweet.copy()
        
        elif labels[0] != 'none' and labels[0] != 'normal':           
            
            original_tokenized_tweet = tokenized_tweet[0].copy()
            tokenized_tweet = tokenized_tweet[0].copy()
            
            for lab in labels:
                
                for high, spans in zip(highlights, keyword_spans):
                    
                    for tup, span in zip(high, spans):
                        
                        if tup[0] == lab:

                            if dataset == 'caves':
                                question = f"Why is \"{lab}\" a reason for not taking vaccines?"
                            elif dataset =='14lap' or ('res' in dataset.split()):
                                question = f'Why is the sentiment \"{lab}\"?'
                            else:
                                question = f'Why is the sentence \"{lab}\"?'
                            
                            question = tokenizer(question).input_ids[1:-1]
                            tokenized_tweet.extend(question)
                            
                            tweet_text.append(tokenizer.decode(tokenized_tweet))
                            
                            tokenized_tweet_with_question.append(tokenized_tweet)
                            
                            tokenized_keywords.append(tokenizer.decode(tokenized_tweet[span[0][0]:span[0][1]]))
                            
                            tweet_labels.append(lab)
                            
                            start_token.append(span[0][0])
                            end_token.append(span[0][1])
                            
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

                if dataset == 'caves':
                    question = f'Why is \"{neg}\" a reason for not taking vaccines?'
                elif dataset == '14lap' or ('res' in dataset.split()):
                    question = f'Why is the sentiment \"{neg}\"?'
                else:
                    question = f'Why is the sentence \"{neg}\"?'
                
                question = tokenizer(question).input_ids[1:-1]
                
                tokenized_tweet.extend(question)
                
                tweet_text.append(tokenizer.decode(tokenized_tweet))
                
                tokenized_tweet_with_question.append(tokenized_tweet)
                
                tokenized_keywords.append('')
                
                tweet_labels.append(neg)
                
                start_token.append(null_token_id)
                end_token.append(null_token_id)
                
                tokenized_tweet = original_tokenized_tweet.copy()
        else:
            original_tokenized_tweet = tokenized_tweet[0].copy()
            tokenized_tweet = tokenized_tweet[0].copy()
            
            if dataset == 'caves':
                question = f"Why is \"none\" a reason for not taking vaccines?"
            else:
                question = f'Why is the sentence \"normal\"?'
            
            question = tokenizer(question).input_ids[1:-1]
            
            tokenized_tweet.extend(question)
            
            tweet_text.append(tokenizer.decode(tokenized_tweet))
            
            tokenized_tweet_with_question.append(tokenized_tweet)
            
            tokenized_keywords.append('')
            
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

                if dataset == 'caves':
                    question = f'Why is \"{neg}\" a reason for not taking vaccines?'
                else:
                    question = f'Why is the sentence \"{neg}\"?'

                question = tokenizer(question).input_ids[1:-1]
                
                tokenized_tweet.extend(question)
                
                tweet_text.append(tokenizer.decode(tokenized_tweet))
                
                tokenized_tweet_with_question.append(tokenized_tweet)
                
                tokenized_keywords.append('')
                
                tweet_labels.append(neg)
                
                start_token.append(null_token_id)
                end_token.append(null_token_id)
                
                tokenized_tweet = original_tokenized_tweet.copy()

    return tokenized_tweet_with_question, tweet_text, tweet_labels, tokenized_keywords, start_token, end_token


# %% LOADING DATA

def fix_end_token(end_tokens, start_tokens):
    end_token = []
    for start, end in zip(start_tokens, end_tokens):
        start = int(start)
        end = int(end)
        if start == end: end_token.append(end)
        #elif end-start ==1: end_token.append(end)
        else: end_token.append(end-1)
    return end_token

def load_data_from_file(filename):
    if filename.split('/')[-2] == 'train':
        f= open(filename+'train_labels.json')
        labels = json.load(f)
        f= open(filename+'train_keywords.json')
        keywords = json.load(f)
        f= open(filename+'train_start_tokens.json')
        start_token = json.load(f)
        f= open(filename+'train_end_tokens.json')
        end_token = json.load(f)
        end_token = fix_end_token(end_token, start_token)
        f= open(filename+'train_tokenized_twt.json')
        tweet = json.load(f)
        return tweet, start_token, end_token, keywords, labels
    elif filename.split('/')[-2] == 'val':
        f= open(filename+'val_labels.json')
        labels = json.load(f)
        f= open(filename+'val_keywords.json')
        keywords = json.load(f)
        f= open(filename+'val_start_tokens.json')
        start_token = json.load(f)
        f= open(filename+'val_end_tokens.json')
        end_token = json.load(f)
        end_token = fix_end_token(end_token, start_token)
        f= open(filename+'val_tokenized_twt.json')
        tweet = json.load(f)
        return tweet, start_token, end_token, keywords, labels
    else:
        f= open(filename+'test_labels.json')
        labels = json.load(f)
        f= open(filename+'test_keywords.json')
        keywords = json.load(f)
        f= open(filename+'test_tokenized_twt.json')
        tweet = json.load(f)
        return tweet, keywords, labels



def get_test_data(testfile=None, 
                       tokenizer=None, 
                       is_testing=True, 
                       is_bert=False, 
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
        
        tweet = ' '
        tweet += dat['tweet_text']
        tokenized_tweet = [preprocess_and_tokenize(tweet, tokenizer)[:-1]]

        keys = [dat['keyword']]
        keys = keyword_position(keys, tokenizer)

        labels = dat['labels']

        highlights = get_label_highlight(tokenized_tweet, keys, [labels], ALL_LABELS)

        keyword_spans = span_finder(highlights, use_only_max_length=False)
        
        if not is_bert:
            tweet += ' <unk>'
            tweet += '</s>'
            tokenized_tweet[0].extend([1437,3,2])
            null_token_id = len(tokenized_tweet[0]) - 2 ## <unk> token id
        else:
            tweet += ' [UNK] ' # token_id = 100
            tweet += '[SEP]' # token_id = 102
            tokenized_tweet[0].extend([100, 102])
            null_token_id = len(tokenized_tweet[0]) - 2

        original_tokenized_tweet = tokenized_tweet[0].copy()
        tokenized_tweet = tokenized_tweet[0].copy()
        
        questions_per_sample = {}
        questions_per_sample['tokenized_question'] = []
        questions_per_sample['question_text'] = []
        questions_per_sample['gt_labels'] = []
        questions_per_sample['gt_keywords'] = []
        questions_per_sample['labels'] = []
        questions_per_sample['start_token'] = None
        questions_per_sample['end_token'] = None

        for lab in ALL_LABELS:
            if dataset == 'caves':
                question = f'Why is  \"{lab}\" a reason for not taking vaccines?'
            elif dataset =='14lap' or ('res' in dataset.split()):
                question = f'Why is the sentiment \"{lab}\"?'
            else:
                question = f'Why is the sentence \"{lab}\"?'
            
            question = tokenizer(question).input_ids[1:-1]
            
            ## if lab is actually a ground truth label
            if lab in labels and (lab != 'none' and lab != 'normal'):
                for high, spans in zip(highlights, keyword_spans):
                    for tup, span in zip(high, spans):
                        if tup[0] == lab:
                            #if is_testing:
                            union_key_binary = tup[1]
                            union_key = [tokenized_tweet[i] for i in range(len(union_key_binary)) if union_key_binary[i] !=0]
                            questions_per_sample['gt_keywords'].append(tokenizer.decode(union_key))
                            
                            if not is_testing:
                                questions_per_sample['start_token'] = span[0][0]
                                questions_per_sample['end_token'] = span[0][1]
                            
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
                
                if not is_testing:
                    questions_per_sample['start_token'] = 0 # when the ground truth label itself is 'none'  
                    questions_per_sample['end_token'] = 0
                tokenized_tweet = original_tokenized_tweet.copy()
            
            else: # when framing contrasting questions
                tokenized_tweet.extend(question)
                questions_per_sample['question_text'].append(tokenizer.decode(tokenized_tweet))
                questions_per_sample['tokenized_question'].append(tokenized_tweet)
                questions_per_sample['labels'].append(lab)
                if not is_testing:
                    questions_per_sample['start_token'] = null_token_id   
                    questions_per_sample['end_token'] = null_token_id
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
    
    # tweets = pad_sequences(tweets, padding='post', value=tokenizer.pad_token_id)
    tweets = pad_sequence(tweets, batch_first = True, padding_value = tokenizer.pad_token_id)
    
    keywords_final = []
    
    if keywords is not None:
        
        for k in keywords:
            
            if len(k) == 0 or k =='':
                keywords_final.append([-1])
            else:
                keywords_final.append(k)
        
        # keywords_final = pad_sequences(keywords_final, padding='post', value=tokenizer.pad_token_id)
        keywords_final = pad_sequence(keywords_final, batch_first = True, padding_value = tokenizer.pad_token_id)
        
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
