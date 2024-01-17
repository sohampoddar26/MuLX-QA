import torch
import random
import numpy as np
import re
import string


def seed(seed=10):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)



def save_model(epoch, model, loss, model_save_path):
    checkpoint = {'epoch': epoch,
                  'state_dict': model.state_dict(),
                  'loss': loss
                  }
    torch.save(checkpoint, model_save_path)
    print("\nSaving model at iteration {} with validation Loss {}".format(epoch, loss))

def load_model(model, model_save_path):
    checkpoint = torch.load(model_save_path)
    model.load_state_dict(checkpoint['state_dict'])
    print(f"\nLoaded Model from epoch {checkpoint['epoch']}")
    
    return model


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def convert_labels_to_id(ALL_LABELS):
    labels_to_id = {}
    for i in range(len(ALL_LABELS)):
        labels_to_id[ALL_LABELS[i]] = i
    return labels_to_id



def jaccard(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union



def best_logit_score(start_logit, end_logit, n_best, max_answer_length):
    start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
    end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
    answers = []
    for start_index in start_indexes:
        for end_index in end_indexes:
            # Skip answers with a length that is either < 0 or > max_answer_length.
            if (end_index < start_index or end_index - start_index + 1 > max_answer_length):
                continue
            
            answers.append({"logit_score": start_logit[start_index] + end_logit[end_index],
                            "start": start_index,
                            "end": end_index})

    best_answer = max(answers, key=lambda x: x["logit_score"])
    return best_answer


