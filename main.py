import os
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AdamW, get_linear_schedule_with_warmup
from modules.data_utils import load_data_from_file, get_test_data, dataloader
from modules.traineval_utils import seed

from modules.train import train
from modules.eval import evaluate

# CHOOSE THE PARAMS
DATASET_NAME = 'caves' # 'hatexplain'


test = '../CAVES_data/test.csv'

TRAINDIR = './data_' + DATASET_NAME + '/train/'
VALDIR = './data_' + DATASET_NAME + '/val/'




################################################################################
################################################################################
# %% Define Parameters, Tokenizer, Model and Dataloaders:
################################################################################
################################################################################


params = {}
params['device'] = 'cuda'
params['dataset'] = DATASET_NAME # THE DATASET BEING USED.
params['batch_size'] = 16 # DO NOT CHANGE THIS, MAKES CHANGES TO THE GRADIENT ACCUMULATION STEPS FOR INCREASING THE BATCH SIZE
params['learning_rate'] = 5e-06 #2e-5 # DEFAULT IS 3e-05
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



#tokenizer = AutoTokenizer.from_pretrained('roberta-large')
tokenizer = AutoTokenizer.from_pretrained('digitalepidemiologylab/covid-twitter-bert-v2')
#tokenizer = AutoTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

seed()

# model = AutoModelForQuestionAnswering.from_pretrained('digitalepidemiologylab/covid-twitter-bert-v2')
# model = model.to(params['device'])



test_samples = get_test_data(test, # TEST CSV FILE LOCATION
                            tokenizer, 
                            is_testing=True, # ALWAYS SET TO TRUE
                            is_bert=True, # TRUE FOR BERT AND FALSE FOR ROBERTA AND OTHER MODELS.
                            dataset=params['dataset']) # 'caves' for covax, 'hatexplain' for hatexplain dataset and similarly for lap & res



train_tweet, train_start_token, train_end_token, train_keyword, train_labels = load_data_from_file(TRAINDIR)
val_tweet, val_start_token, val_end_token, val_keyword, val_labels =  load_data_from_file(VALDIR)


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



#%% TRAIN and TEST

train(model, train_dataloader, val_dataloader, optimizer, scheduler, params)

evaluate(model, test_samples, tokenizer, params)
