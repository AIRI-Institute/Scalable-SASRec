import numpy as np
import torch

from utils import topn_recommendations, downvote_seen_items
from data import data_to_sequences


def get_test_scores(model, data_description, testset_, holdout_, device):
    sasrec_scores = sasrec_model_scoring(model, testset_, data_description, device)
    downvote_seen_items(sasrec_scores, testset_, data_description)

    sasrec_recs = topn_recommendations(sasrec_scores, topn=50)
    test_scores = model_evaluate(sasrec_recs, holdout_, data_description)
    return test_scores


def sasrec_model_scoring(params, data, data_description, device):
    model = params
    model.eval()
    test_sequences = data_to_sequences(data, data_description)
    # perform scoring on a user-batch level
    scores = []
    for _, seq in test_sequences.items():
        with torch.no_grad():
            predictions = model.score(torch.tensor(seq, device=device, dtype=torch.long))
        scores.append(predictions.detach().cpu().numpy())
    return np.concatenate(scores, axis=0)


def calculate_topn_metrics(recommended_items, holdout_items, n_items, n_test_users, topn):
    hits_mask = recommended_items[:, :topn] == holdout_items.reshape(-1, 1)

    # HR calculation
    hr = np.mean(hits_mask.any(axis=1))

    # MRR calculation
    hit_rank = np.where(hits_mask)[1] + 1.0
    mrr = np.sum(1 / hit_rank) / n_test_users
   
    #NDCG calculation
    ndcg = np.sum(1 / np.log2(hit_rank + 1.)) / n_test_users

    #COV calculation
    cov = np.unique(recommended_items[:, :topn]).size / n_items

    return {'hr': hr, 'mrr': mrr, 'ndcg': ndcg, 'cov': cov}


def model_evaluate(recommended_items, holdout, holdout_description, topn_list=(1, 5, 10, 20, 50)):
    n_items = holdout_description['n_items']
    itemid = holdout_description['items']
    holdout_items = holdout[itemid].values
    n_test_users = recommended_items.shape[0]
    assert recommended_items.shape[0] == len(holdout_items)

    metrics = {}
    for topn in topn_list:
        metrics = metrics | {f'{key}@{topn}': value for key, value in calculate_topn_metrics(recommended_items, holdout_items, n_items, n_test_users, topn).items()}

    return metrics
