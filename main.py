import os
import json
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AdamW, get_linear_schedule_with_warmup
from modules.data_utils import dataloader
from modules.traineval_utils import seed

from modules.train import train
from modules.eval import evaluate

# CHOOSE THE PARAMS
DATASET_NAME = 'caves' # 'hatexplain'


DATADIR = './data_' + DATASET_NAME

#########################################################
#########################################################
# %% Define Parameters, Tokenizer, Model and Dataloaders:
#########################################################
#########################################################


params = {}
params['device'] = 'cuda'
params['dataset'] = DATASET_NAME # THE DATASET BEING USED.
params['batch_size'] = 32 # DO NOT CHANGE THIS, MAKES CHANGES TO THE GRADIENT ACCUMULATION STEPS FOR INCREASING THE BATCH SIZE
params['learning_rate'] = 1e-05 #2e-5 # DEFAULT IS 3e-05
params['epsilon'] = 1e-8
params['weight_decay'] = 0 #0.01
params['num_epochs'] = 10 
params['n_best'] = 20 # NUMBER OF TOP CANDIDATES TO CONSIDER FOR START + END INDEX LOGIT CALCULATION, DEFAULT = 10 (preferably do not lower)
params['max_answer_length'] = 17 # THE MAXIMUM DIFFERENCE BETWEEN THE START AND END INDEX (CHANGES BASED ON THE DATASET BEING USED)
params['num_grad_acc_step'] = 2
params['num_neg_samples'] = 3 # NUMBER OF NEGATIVE SAMPLES BEING USED PER SAMPLE FOR TRAINING (SHOULD BE EQUAL TO THE ONE USED DURING INPUT FILE CREATION)
params['modelname'] = 'ct_bert' # NAME USED TO SAVE THE MODEL


 
model_save_path = './data_%s/models/%s_neg_%d_BS_%d/'%(params['dataset'], params['modelname'], params['num_neg_samples'], params['batch_size'] * params['num_grad_acc_step'])
                               

if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)
    print(f'Directory created: {model_save_path}')
params['model_save_path'] = model_save_path


#########################################################
#########################################################
# %% Initialize models, load data:
#########################################################
#########################################################

#tokenizer = AutoTokenizer.from_pretrained('roberta-large')
tokenizer = AutoTokenizer.from_pretrained('digitalepidemiologylab/covid-twitter-bert-v2')

seed()

model = AutoModelForQuestionAnswering.from_pretrained('digitalepidemiologylab/covid-twitter-bert-v2')
model = model.to(params['device'])


with open(os.path.join(DATADIR, 'train_data.json')) as f:
    data = json.load(f)
    train_tweet, train_start_token, train_end_token, train_keyword, train_labels = [data[x] for x in ['tokenized_text', 'start_tokens', 'end_tokens', 'tokenized_keywords', 'labels']]


with open(os.path.join(DATADIR, 'val_data.json')) as f:
    data = json.load(f)
    val_tweet, val_start_token, val_end_token, val_keyword, val_labels = [data[x] for x in ['tokenized_text', 'start_tokens', 'end_tokens', 'tokenized_keywords', 'labels']]


train_dataloader = dataloader(tweets=train_tweet, 
                               start_tokens=train_start_token,
                               end_tokens=train_end_token,
                               tokenizer=tokenizer,
                               params=params,
                               is_train=True)

val_dataloader = dataloader(tweets=val_tweet, 
                            start_tokens=val_start_token, 
                            end_tokens=val_end_token, 
                            tokenizer=tokenizer, 
                            params=params, 
                            is_train=False)


with open(os.path.join(DATADIR, 'test_data.json')) as f:
    test_samples = json.load(f)



no_decay = ["bias", "LayerNorm.weight"]

optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": params['weight_decay']
    },
    {
        "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
]

optimizer = AdamW(optimizer_grouped_parameters, lr=params['learning_rate'], eps=params['epsilon'])


total_steps =  params['num_epochs'] * len(train_dataloader) // params['num_grad_acc_step'] # // params['batch_size']
scheduler = get_linear_schedule_with_warmup(optimizer, num_training_steps=total_steps, num_warmup_steps = int(total_steps/10))
print(total_steps)



#########################################################
#########################################################
# %% Train and test
#########################################################
#########################################################


train(model, train_dataloader, val_dataloader, optimizer, scheduler, params)

evaluate(model, test_samples, tokenizer, params)
