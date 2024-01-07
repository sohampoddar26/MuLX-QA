import torch
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, jaccard_score
from .traineval_utils import load_model, normalize_answer, jaccard, convert_labels_to_id, best_logit_score



def test_model(data:list, model, tokenizer, params):

    if params['dataset'] == 'caves':
        ALL_LABELS = np.array(['none'] + sorted(['unnecessary', 'side-effect', 'ineffective', 'mandatory', 'pharma', 'political', 'conspiracy', 'rushed', 'country', 'ingredients', 'religious']))
    elif params['dataset'] =='hatexplain':
        ALL_LABELS = np.array(['normal', 'hatespeech', 'offensive'])
    else:
        ALL_LABELS = np.array(['positive', 'neutral', 'negative'])

    labels_to_id = convert_labels_to_id(ALL_LABELS)
    inputs, predicted_keywords, predicted_labels, pred_labels, true_labels = [], [], [], [], []
    raw_predicted_keywords = []
    tuple_predicted_labels = []
    tuple_true_labels = []
    # While true and pred counts are only concerned with labels.
    # correct count is for both label and keyword matching (if jacc > 0.5)
    true_count, pred_count, correct_count = 0, 0, 0
    
    for idx, sample in enumerate(data):
        
        pred_label = [1 for i in range(len(ALL_LABELS))]
        true_label = [0 for i in range(len(ALL_LABELS))]
        tuple_pred_label = [0 for i in range(len(ALL_LABELS))]
        
        gt_label = sample['gt_labels']
        gt_keyword = sample['gt_keywords']
        
        for lab in gt_label:
            true_label[labels_to_id[lab]] = int(1)
        
        predicted_label = []
        predicted_keyword = []
        raw_keyword = []
        inp = []
        
        for lab, input in zip(sample['labels'], sample['tokenized_question']):

            attn = torch.tensor([1 for i in range(len(input))]).to(params['device']).unsqueeze(0)
            input = torch.tensor(input).to(params['device']).unsqueeze(0)

            output = model(input_ids=input, attention_mask=attn)
            
            input = input.squeeze(0).detach().cpu().tolist()
            inp.append(input)
            
            start_logit = output.start_logits.squeeze(0).detach().cpu().numpy()
            end_logit = output.end_logits.squeeze(0).detach().cpu().numpy()

            best_answer = best_logit_score(start_logit, end_logit, params['n_best'], params['max_answer_length'])
            out = tokenizer.decode(input[best_answer['start']:best_answer['end']+1])

            #pdb.set_trace()
            if (out =='<unk>' or out == '[UNK]') or ('<unk>' in out or '[UNK]' in out):
                pred_label[labels_to_id[lab]] = 0 #updating the one-hot vector
                #predicted_label.append(None)
                raw_keyword.append(out)
                continue
            elif out == '<s>' or out == '[CLS]':
                predicted_label.append('none') # updating the label list
                predicted_keyword.append('') # updating the keyword list
                raw_keyword.append(out)
            else:
                predicted_label.append(lab)
                predicted_keyword.append(out)
                raw_keyword.append(out)
        
        # Calculating the pred, true and correct count
        pred_count += len(predicted_label)
        true_count += len(gt_label)

        for p_l, p_k in zip(predicted_label, predicted_keyword):
            for i in range(len(gt_label)): #['mandatory', 'conspiracy'], ['none']
                if gt_label[i] == p_l:
                    if p_l != 'none' and p_l != 'normal':
                        # calculate the jaccard score between the predicted keyword and the true keyword:
                        true_key = gt_keyword[i]
                        true_key = normalize_answer(true_key)
                        p_k = normalize_answer(p_k)
                        if jaccard(true_key.split(), p_k.split()) >= 0.5:
                            correct_count += 1
                            tuple_pred_label[labels_to_id[gt_label[i]]] = 1
                    else:
                        true_key = gt_keyword[i]
                        '''assert true_key == '', 'True Label is none but keyword still defined!'
                        try:
                            assert p_k == '', 'Predicted Label is none but keyword still predicted!'
                        except:
                            pdb.set_trace()'''
                        correct_count += 1
                        tuple_pred_label[labels_to_id[gt_label[i]]] = 1
                else:
                    continue

        raw_predicted_keywords.append(raw_keyword)
        predicted_keywords.append(predicted_keyword)
        
        predicted_labels.append(predicted_label)
        
        inputs.append(inp)
        
        true_labels.append(true_label)
        pred_labels.append(pred_label)
        
        tuple_true_labels.append(true_label)
        tuple_predicted_labels.append(tuple_pred_label)

    out = {}
    out['predicted_keyword'] = predicted_keywords
    out['raw_predicted_keywords'] = raw_predicted_keywords
    out['inputs'] = inputs
    out['true_labels'] = true_labels
    out['pred_labels'] = pred_labels
    out['pred_count'] = pred_count
    out['true_count'] = true_count
    out['correct_count'] = correct_count
    out['tuple_pred_labels'] = tuple_predicted_labels
    out['tuple_true_labels'] = tuple_true_labels

    return out



def evaluate(model, test_samples, tokenizer, params):
        model = load_model(model, params['model_save_path']+'/lowest_loss_model.pt')
        model.eval()
        with torch.no_grad():
            out = test_model(test_samples, model, tokenizer, params)

        
        precision = out['correct_count']  / out['pred_count']
        recall = out['correct_count'] / out['true_count']
        f1 = (2 * precision * recall) / (precision + recall + 1e-8)
        
        print("Pre, Rec, F1:")
        print(precision)
        print(recall)
        print(f1)
        
