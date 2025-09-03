#!/usr/bin/env python
import argparse
from os.path import basename, splitext

import mmengine
import numpy as np
import pandas as pd
import torch
from numpy.linalg import norm, pinv
from scipy.special import logsumexp, softmax
from sklearn import metrics
from sklearn.covariance import EmpiricalCovariance
from sklearn.metrics import pairwise_distances_argmin_min
from tqdm import tqdm
from scipy.spatial.distance import cdist
import pickle
from copy import deepcopy
import faiss
from mpmath import exp
import time

EPSILON = 1e-8
#mapping = 'datalists/mapping.pkl'

def parse_args():
    parser = argparse.ArgumentParser(description='Say hello')
    parser.add_argument('fc', help='Path to config')
    parser.add_argument('id_train_feature', help='Path to data')
    parser.add_argument('id_val_feature', help='Path to output file')
    parser.add_argument('ood_features', nargs='+', help='Path to ood features')
    parser.add_argument(
        '--train_label',
        default='datalists/imagenet2012_train_random_200k.txt',
        help='Path to train labels')
    parser.add_argument(
        '--clip_quantile', default=0.99, help='Clip quantile to react')

    return parser.parse_args()


# region Helper
def num_fp_at_recall(ind_conf, ood_conf, tpr):
    num_ind = len(ind_conf)

    if num_ind == 0 and len(ood_conf) == 0:
        return 0, 0.
    if num_ind == 0:
        return 0, np.max(ood_conf) + 1

    recall_num = int(np.floor(tpr * num_ind))
    thresh = np.sort(ind_conf)[-recall_num]
    num_fp = np.sum(ood_conf >= thresh)
    return num_fp, thresh


def fpr_recall(ind_conf, ood_conf, tpr):
    num_fp, thresh = num_fp_at_recall(ind_conf, ood_conf, tpr)
    num_ood = len(ood_conf)
    fpr = num_fp / max(1, num_ood)
    return fpr, thresh

def near_far_mean(result, list_near_ood, list_far_ood):
    far_ood_result = [entry for entry in result if any(far in entry["oodset"] for far in list_far_ood)]
    far_ood_auroc = 0
    far_ood_fpr = 0
    for entry in far_ood_result:
        far_ood_auroc += entry['auroc']
        far_ood_fpr += entry['fpr']

    near_ood_result = [entry for entry in result if any(near in entry["oodset"] for near in list_near_ood)]
    near_ood_auroc = 0
    near_ood_fpr = 0
    for entry in near_ood_result:
        near_ood_auroc += entry['auroc']
        near_ood_fpr += entry['fpr']

    return near_ood_auroc / len(list_near_ood), near_ood_fpr / len(list_near_ood), far_ood_auroc / len(list_far_ood), far_ood_fpr / len(list_far_ood)


def auc(ind_conf, ood_conf):
    conf = np.concatenate((ind_conf, ood_conf))
    ind_indicator = np.concatenate(
        (np.ones_like(ind_conf), np.zeros_like(ood_conf)))

    fpr, tpr, _ = metrics.roc_curve(ind_indicator, conf)
    precision_in, recall_in, _ = metrics.precision_recall_curve(
        ind_indicator, conf)
    precision_out, recall_out, _ = metrics.precision_recall_curve(
        1 - ind_indicator, 1 - conf)

    auroc = metrics.auc(fpr, tpr)
    aupr_in = metrics.auc(recall_in, precision_in)
    aupr_out = metrics.auc(recall_out, precision_out)

    return auroc, aupr_in, aupr_out


def kl(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))


def gradnorm(x, w, b):
    fc = torch.nn.Linear(*w.shape[::-1])
    fc.weight.data[...] = torch.from_numpy(w)
    fc.bias.data[...] = torch.from_numpy(b)
    fc.cuda()

    x = torch.from_numpy(x).float().cuda()
    logsoftmax = torch.nn.LogSoftmax(dim=-1).cuda()

    confs = []

    for i in tqdm(x):
        targets = torch.ones((1, 1000)).cuda()
        fc.zero_grad()
        loss = torch.mean(
            torch.sum(-targets * logsoftmax(fc(i[None])), dim=-1))
        loss.backward()
        layer_grad_norm = torch.sum(torch.abs(
            fc.weight.grad.data)).cpu().numpy()
        confs.append(layer_grad_norm)

    return np.array(confs)

def ash_b(x, percentile=65):
    assert x.dim() == 4
    assert 0 <= percentile <= 100
    b, c, h, w = x.shape

    # calculate the sum of the input per sample
    s1 = x.sum(dim=[1, 2, 3])

    n = x.shape[1:].numel()
    k = n - int(np.round(n * percentile / 100.0))
    t = x.view((b, c * h * w))
    v, i = torch.topk(t, k, dim=1)
    fill = s1 / k
    fill = fill.unsqueeze(dim=1).expand(v.shape)
    t.zero_().scatter_(dim=1, index=i, src=fill)
    return x


def ash_p(x, percentile=65):
    assert x.dim() == 4
    assert 0 <= percentile <= 100

    b, c, h, w = x.shape

    n = x.shape[1:].numel()
    k = n - int(np.round(n * percentile / 100.0))
    t = x.view((b, c * h * w))
    v, i = torch.topk(t, k, dim=1)
    t.zero_().scatter_(dim=1, index=i, src=v)

    return x


def ash_s(x, percentile=65):
    assert x.dim() == 4
    assert 0 <= percentile <= 100
    b, c, h, w = x.shape

    # calculate the sum of the input per sample
    s1 = x.sum(dim=[1, 2, 3])
    n = x.shape[1:].numel()
    k = n - int(np.round(n * percentile / 100.0))
    t = x.view((b, c * h * w))
    v, i = torch.topk(t, k, dim=1)
    t.zero_().scatter_(dim=1, index=i, src=v)

    # calculate new sum of the input per sample after pruning
    s2 = x.sum(dim=[1, 2, 3])

    # apply sharpening
    scale = s1 / s2
    x = x * torch.exp(scale[:, None, None, None])

    return x

def scale(x, percentile=65):
    input = x.clone()
    assert x.dim() == 4
    assert 0 <= percentile <= 100
    b, c, h, w = x.shape

    # calculate the sum of the input per sample
    s1 = x.sum(dim=[1, 2, 3])
    n = x.shape[1:].numel()
    k = n - int(np.round(n * percentile / 100.0))
    t = x.view((b, c * h * w))
    v, i = torch.topk(t, k, dim=1)
    t.zero_().scatter_(dim=1, index=i, src=v)

    # calculate new sum of the input per sample after pruning
    s2 = x.sum(dim=[1, 2, 3])

    # apply sharpening
    scale = s1 / s2

    return input * torch.exp(scale[:, None, None, None])

def she_distance(penultimate, target, metric='inner_product'):
    if metric == 'inner_product':
        return np.sum(np.multiply(penultimate, target), axis=1)
    elif metric == 'euclidean':
        return -np.sqrt(np.sum((penultimate - target) ** 2, axis=1))
    elif metric == 'cosine':
        dot_product = np.sum(penultimate * target, axis=1)
        norm_penultimate = np.linalg.norm(penultimate, axis=1)
        norm_target = np.linalg.norm(target, axis=1)
        return dot_product / (norm_penultimate * norm_target)
    else:
        raise ValueError(f'Unknown metric: {metric}')


def generalized_entropy(softmax_id_val, gamma=0.1, M=100):
    probs = softmax_id_val
    probs_sorted = np.sort(probs, axis=1)[:, -M:]
    scores = np.sum(probs_sorted**gamma * (1 - probs_sorted)**gamma, axis=1)
    
    return -scores

def knn_score(bankfeas, queryfeas, k=100, min=False):

    bankfeas = deepcopy(np.array(bankfeas))
    queryfeas = deepcopy(np.array(queryfeas))

    index = faiss.IndexFlatIP(bankfeas.shape[-1])
    index.add(bankfeas)
    D, _ = index.search(queryfeas, k)
    if min:
        scores = np.array(D.min(axis=1))
    else:
        scores = np.array(D.mean(axis=1))
    return scores

def softmax_temperature(x, axis=None, temperature=1.0):
    x = x / temperature  # Apply temperature scaling
    x_max = np.amax(x, axis=axis, keepdims=True)
    exp_x_shifted = np.exp(x - x_max)
    return exp_x_shifted / np.sum(exp_x_shifted, axis=axis, keepdims=True)


def mme(logits, distance, r, vlogits, fdbd, mapping = None, plus = True):
    logits_ = logsumexp(logits, axis=-1)
    max_logits_indices = np.argmax(logits, axis=1)
    temperature = 0.1
    if mapping is not None:
        map_func = np.vectorize(lambda x: mapping.get(x, None))
        max_logits_indices = map_func(max_logits_indices)
        temperature = 0.5

    distance = softmax_temperature(distance, axis=1, temperature=temperature)
    distance = 1 / distance
    distance_ = np.max(distance, axis=1)
    max_distance_indices = np.argmax(distance, axis=1)

    index_matches_bool = (max_logits_indices == max_distance_indices)
    if mapping is not None:
        index_matches_float = np.where(index_matches_bool, 0.5, 1.0)
    else:
        index_matches_float = np.where(index_matches_bool, 2.0, 1.0)

    if plus:
        confidence_score = np.exp(logits_ - vlogits) * distance_ * index_matches_float * (1 - r) * fdbd
    else:
        confidence_score = np.exp(logits_ - vlogits) * distance_ * index_matches_float * fdbd
    
    max_float = np.finfo(confidence_score.dtype).max #just for numerical stabilities
    confidence_score[confidence_score == np.inf] = max_float
    return confidence_score


def main():
    args = parse_args()
    print(args)

    ood_names = [splitext(basename(ood))[0] for ood in args.ood_features]
    print(f'ood datasets: {ood_names}')

    if 'imagenet' in args.id_val_feature:
        with open('datalists/mapping.pkl', 'rb') as handle:
            mapping = pickle.load(handle)
    else:
        mapping = None

    list_far_ood = ['inaturalist', 'texture', 'openimage']
    list_near_ood = ['ssb', 'ninco']

    w, b = mmengine.load(args.fc)
    print(f'{w.shape=}, {b.shape=}')

    train_labels = np.array([
        int(line.rsplit(' ', 1)[-1])
        for line in mmengine.list_from_file(args.train_label)
    ],
                            dtype=int)

    recall = 0.95

    print('load features')

    feature_id_train = mmengine.load(args.id_train_feature)
    feature_id_val = mmengine.load(args.id_val_feature)

    feature_oods = {
        name: mmengine.load(feat)
        for name, feat in zip(ood_names, args.ood_features)
    }
    #args.clip_quantile = 1 #only for wideresnet10
    clip_high = np.quantile(feature_id_train, args.clip_quantile)
    print(f'clip quantile high {args.clip_quantile}, clip {clip_high:.4f}')

    clip_low = np.quantile(feature_id_train, 1 - args.clip_quantile)
    print(f'clip quantile low {1 - args.clip_quantile}, clip {clip_low:.4f}')

    # ---------------------------------------
    # Introduced by Energy + React, used in PCA
    feature_id_val_clip = np.clip(mmengine.load(args.id_val_feature), a_min=None, a_max=clip_high)
    feature_oods_clip = {
        name: np.clip(mmengine.load(feat), a_min=None, a_max=clip_high)
        for name, feat in zip(ood_names, args.ood_features)
    }
    # ---------------------------------------

    # ---------------------------------------
    # Introduced by Energy + VRA
    feature_id_train_clip_VRA = np.clip(mmengine.load(args.id_train_feature), a_min=clip_low, a_max=clip_high)
    feature_id_val_clip_VRA = np.clip(mmengine.load(args.id_val_feature), a_min=clip_low, a_max=clip_high)
    feature_oods_clip_VRA = {
        name: np.clip(mmengine.load(feat), a_min=clip_low, a_max=clip_high)
        for name, feat in zip(ood_names, args.ood_features)
    }
    # ---------------------------------------

    print(f'{feature_id_train.shape=}, {feature_id_val.shape=}')
    for name, ood in feature_oods.items():
        print(f'{name} {ood.shape}')
    print('computing logits...')
    logit_id_train = feature_id_train @ w.T + b
    logit_id_val = feature_id_val @ w.T + b
    logit_oods = {name: feat @ w.T + b for name, feat in feature_oods.items()}

    logit_id_val_clip = feature_id_val_clip @ w.T + b
    logit_oods_clip = {name: feat @ w.T + b for name, feat in feature_oods_clip.items()}

    logit_id_val_clip_VRA = feature_id_val_clip_VRA @ w.T + b
    logit_oods_clip_VRA = {name: feat @ w.T + b for name, feat in feature_oods_clip_VRA.items()}


    max_logits_indices = np.argmax(logit_id_val, axis=1)

    print('computing softmax...')
    softmax_id_train = softmax(logit_id_train, axis=-1)
    softmax_id_val = softmax(logit_id_val, axis=-1)
    softmax_oods = {
        name: softmax(logit, axis=-1)
        for name, logit in logit_oods.items()
    }

    # ---------------------------------------
    # Introduced by Residual, used in ViM and Ours
    if feature_id_val.shape[-1] >= 2048:
        DIM = 1000
    elif feature_id_val.shape[-1] >= 768:
        DIM = 512
    else:
        DIM = feature_id_val.shape[-1] // 2
    print(f'{DIM=}')
    
    u = -np.matmul(pinv(w), b)
    print('computing principal space...')
    ec = EmpiricalCovariance(assume_centered=True)
    ec.fit(feature_id_train - u)
    eig_vals, eigen_vectors = np.linalg.eig(ec.covariance_)
    NS = np.ascontiguousarray(
        (eigen_vectors.T[np.argsort(eig_vals * -1)[DIM:]]).T)
    # ---------------------------------------

    # ---------------------------------------
    # Introduced by ViM, used in ours
    print('computing alpha...')
    vlogit_id_train = norm(np.matmul(feature_id_train - u, NS), axis=-1)
    alpha = logit_id_train.max(axis=-1).mean() / vlogit_id_train.mean()
    print(f'{alpha=:.4f}')
    # ---------------------------------------

    # ---------------------------------------
    # Introduced by Ours, similar to Mahalanobis?
    class_means = np.zeros((train_labels.max() + 1, feature_id_train.shape[1]))
    for i in tqdm(range(train_labels.max() + 1)):
        vectors = feature_id_train[train_labels == i]
        vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
        mean = np.mean(vectors, axis=0)
        mean = mean / np.linalg.norm(mean)
        class_means[i, :] = mean
    # ---------------------------------------

    # ---------------------------------------
    # Introduced by PCA
    feature_mean = np.mean(feature_id_train, axis=0)

    cov = np.cov(feature_id_train.T)
    u_, s, v = np.linalg.svd(cov)

    k = 256

    M = u_[:, :k] @ u_[:, :k].T
    dim = M.shape[0]
    # ---------------------------------------

    df = pd.DataFrame(columns=['method', 'oodset', 'auroc', 'fpr'])

    dfs = []

    # ---------------------------------------
    method = 'MSP'
    print(f'\n{method}')
    result = []
    score_id = softmax_id_val.max(axis=-1)

    for name, softmax_ood in softmax_oods.items():
        score_ood = softmax_ood.max(axis=-1)
        auc_ood = auc(score_id, score_ood)[0]
        fpr_ood, _ = fpr_recall(score_id, score_ood, recall)
        result.append(
            dict(method=method, oodset=name, auroc=auc_ood, fpr=fpr_ood))
        print(f'{method}: {name} auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}')
    df = pd.DataFrame(result)
    dfs.append(df)

    near_auroc, near_fpr, far_auroc, far_fpr = near_far_mean(result, list_near_ood, list_far_ood)

    print(f"far mean auroc: {far_auroc:.2%}, far mean fpr: {far_fpr:.2%}")
    print(f"near mean auroc: {near_auroc:.2%}, near mean fpr: {near_fpr:.2%}")
    print(f'mean auroc {df.auroc.mean():.2%}, {df.fpr.mean():.2%}')

    # ---------------------------------------
    method = 'MaxLogit'
    print(f'\n{method}')
    result = []
    score_id = logit_id_val.max(axis=-1)
    for name, logit_ood in logit_oods.items():
        score_ood = logit_ood.max(axis=-1)
        auc_ood = auc(score_id, score_ood)[0]
        fpr_ood, _ = fpr_recall(score_id, score_ood, recall)
        result.append(
            dict(method=method, oodset=name, auroc=auc_ood, fpr=fpr_ood))
        print(f'{method}: {name} auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}')
    df = pd.DataFrame(result)
    dfs.append(df)

    near_auroc, near_fpr, far_auroc, far_fpr = near_far_mean(result, list_near_ood, list_far_ood)

    print(f"far mean auroc: {far_auroc:.2%}, far mean fpr: {far_fpr:.2%}")
    print(f"near mean auroc: {near_auroc:.2%}, near mean fpr: {near_fpr:.2%}")
    print(f'mean auroc {df.auroc.mean():.2%}, {df.fpr.mean():.2%}')

    # ---------------------------------------
    method = 'Energy'
    print(f'\n{method}')
    result = []
    score_id = logsumexp(logit_id_val, axis=-1)
    for name, logit_ood in logit_oods.items():
        score_ood = logsumexp(logit_ood, axis=-1)
        auc_ood = auc(score_id, score_ood)[0]
        fpr_ood, _ = fpr_recall(score_id, score_ood, recall)
        result.append(
            dict(method=method, oodset=name, auroc=auc_ood, fpr=fpr_ood))
        print(f'{method}: {name} auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}')
    df = pd.DataFrame(result)
    dfs.append(df)

    near_auroc, near_fpr, far_auroc, far_fpr = near_far_mean(result, list_near_ood, list_far_ood)

    print(f"far mean auroc: {far_auroc:.2%}, far mean fpr: {far_fpr:.2%}")
    print(f"near mean auroc: {near_auroc:.2%}, near mean fpr: {near_fpr:.2%}")
    print(f'mean auroc {df.auroc.mean():.2%}, {df.fpr.mean():.2%}')

    # ---------------------------------------
    method = 'Energy+React'
    print(f'\n{method}')
    result = []

    score_id = logsumexp(logit_id_val_clip, axis=-1)
    for name, logit_ood_clip in logit_oods_clip.items():
        score_ood = logsumexp(logit_ood_clip, axis=-1)
        auc_ood = auc(score_id, score_ood)[0]
        fpr_ood, _ = fpr_recall(score_id, score_ood, recall)
        result.append(
            dict(method=method, oodset=name, auroc=auc_ood, fpr=fpr_ood))
        print(f'{method}: {name} auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}')
    df = pd.DataFrame(result)
    dfs.append(df)

    near_auroc, near_fpr, far_auroc, far_fpr = near_far_mean(result, list_near_ood, list_far_ood)

    print(f"far mean auroc: {far_auroc:.2%}, far mean fpr: {far_fpr:.2%}")
    print(f"near mean auroc: {near_auroc:.2%}, near mean fpr: {near_fpr:.2%}")
    print(f'mean auroc {df.auroc.mean():.2%}, {df.fpr.mean():.2%}')

    # ---------------------------------------
    method = 'Energy+VRA'
    print(f'\n{method}')
    result = []

    score_id = logsumexp(logit_id_val_clip_VRA, axis=-1)
    for name, logit_ood_clip_VRA in logit_oods_clip_VRA.items():
        score_ood = logsumexp(logit_ood_clip_VRA, axis=-1)
        auc_ood = auc(score_id, score_ood)[0]
        fpr_ood, _ = fpr_recall(score_id, score_ood, recall)
        result.append(
            dict(method=method, oodset=name, auroc=auc_ood, fpr=fpr_ood))
        print(f'{method}: {name} auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}')
    df = pd.DataFrame(result)
    dfs.append(df)

    near_auroc, near_fpr, far_auroc, far_fpr = near_far_mean(result, list_near_ood, list_far_ood)

    print(f"far mean auroc: {far_auroc:.2%}, far mean fpr: {far_fpr:.2%}")
    print(f"near mean auroc: {near_auroc:.2%}, near mean fpr: {near_fpr:.2%}")
    print(f'mean auroc {df.auroc.mean():.2%}, {df.fpr.mean():.2%}')

    # --------------------------------------
    method = 'ViM'
    print(f'\n{method}')
    result = []

    vlogit_id_val = norm(np.matmul(feature_id_val - u, NS), axis=-1) * alpha
    energy_id_val = logsumexp(logit_id_val, axis=-1)
    score_id = -vlogit_id_val + energy_id_val

    for name, logit_ood, feature_ood in zip(ood_names, logit_oods.values(),
                                            feature_oods.values()):
        energy_ood = logsumexp(logit_ood, axis=-1)
        vlogit_ood = norm(np.matmul(feature_ood - u, NS), axis=-1) * alpha
        score_ood = -vlogit_ood + energy_ood
        auc_ood = auc(score_id, score_ood)[0]
        fpr_ood, _ = fpr_recall(score_id, score_ood, recall)
        result.append(
            dict(method=method, oodset=name, auroc=auc_ood, fpr=fpr_ood))
        print(f'{method}: {name} auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}')
    df = pd.DataFrame(result)
    dfs.append(df)

    near_auroc, near_fpr, far_auroc, far_fpr = near_far_mean(result, list_near_ood, list_far_ood)

    print(f"far mean auroc: {far_auroc:.2%}, far mean fpr: {far_fpr:.2%}")
    print(f"near mean auroc: {near_auroc:.2%}, near mean fpr: {near_fpr:.2%}")
    print(f'mean auroc {df.auroc.mean():.2%}, {df.fpr.mean():.2%}')

    # ---------------------------------------
    method = 'DICE'
    print(f'\n{method}')
    result = []
    
    contrib = feature_mean[None, :] * w
    thresh = np.percentile(contrib, 70)
    mask = (contrib > thresh)
    masked_w = w * mask

    output_id_val = feature_id_val @ masked_w.T + b
    score_id = logsumexp(output_id_val, axis=-1)

    for name, logit_ood, feature_ood in zip(ood_names, logit_oods.values(),
                                            feature_oods.values()):
        output_ood = feature_ood @ masked_w.T + b
        score_ood = logsumexp(output_ood, axis=-1)

        auc_ood = auc(score_id, score_ood)[0]
        fpr_ood, _ = fpr_recall(score_id, score_ood, recall)
        result.append(
            dict(method=method, oodset=name, auroc=auc_ood, fpr=fpr_ood))
        print(f'{method}: {name} auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}')
    df = pd.DataFrame(result)
    dfs.append(df)

    near_auroc, near_fpr, far_auroc, far_fpr = near_far_mean(result, list_near_ood, list_far_ood)

    print(f"far mean auroc: {far_auroc:.2%}, far mean fpr: {far_fpr:.2%}")
    print(f"near mean auroc: {near_auroc:.2%}, near mean fpr: {near_fpr:.2%}")
    print(f'mean auroc {df.auroc.mean():.2%}, {df.fpr.mean():.2%}')

    # ---------------------------------------
    method = 'ASH'
    print(f'\n{method}')
    result = []
    percentile_ash = 90 

    feature_id_val_torch = torch.from_numpy(np.copy(feature_id_val))
    feature_id_val_torch = ash_s(feature_id_val_torch.view(feature_id_val_torch.size(0), -1, 1, 1), percentile_ash)
    feature_id_val_torch = feature_id_val_torch.view(feature_id_val_torch.size(0), -1)
    feature_id_val_ash = feature_id_val_torch.cpu().detach().numpy()
    logit_ash_id_val = feature_id_val_ash @ w.T + b
    score_id = logsumexp(logit_ash_id_val, axis=-1)


    for name, logit_ood, feature_ood in zip(ood_names, logit_oods.values(),
                                            feature_oods.values()):
        feature_ood_torch = torch.from_numpy(np.copy(feature_ood))
        feature_ood_torch = ash_s(feature_ood_torch.view(feature_ood_torch.size(0), -1, 1, 1), percentile_ash)
        feature_ood_torch = feature_ood_torch.view(feature_ood_torch.size(0), -1)
        feature_ood_ash = feature_ood_torch.cpu().detach().numpy()
        logit_ash_ood = feature_ood_ash @ w.T + b
        score_ood = logsumexp(logit_ash_ood, axis=-1)

        auc_ood = auc(score_id, score_ood)[0]
        fpr_ood, _ = fpr_recall(score_id, score_ood, recall)
        result.append(
            dict(method=method, oodset=name, auroc=auc_ood, fpr=fpr_ood))
        print(f'{method}: {name} auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}')
    df = pd.DataFrame(result)
    dfs.append(df)

    near_auroc, near_fpr, far_auroc, far_fpr = near_far_mean(result, list_near_ood, list_far_ood)

    print(f"far mean auroc: {far_auroc:.2%}, far mean fpr: {far_fpr:.2%}")
    print(f"near mean auroc: {near_auroc:.2%}, near mean fpr: {near_fpr:.2%}")
    print(f'mean auroc {df.auroc.mean():.2%}, {df.fpr.mean():.2%}')

    # ---------------------------------------
    method = 'SHE'
    print(f'\n{method}')
    result = []
    
    metric = 'inner_product'
    pred_labels_train = np.argmax(softmax_id_train, axis=-1)

    if mapping is not None:
        map_func = np.vectorize(lambda x: mapping.get(x, None))
        pred_labels_train = map_func(pred_labels_train)


    activation_log = []
    for i in range(train_labels.max() + 1):
        mask = (train_labels == i) & (pred_labels_train == i)
        class_correct_activations = feature_id_train[mask]
        activation_log.append(np.mean(class_correct_activations, axis=0)) #pred and train are discordant

    activation_log = np.stack(activation_log, axis=0)
    if mapping is not None:
        score_id = she_distance(feature_id_val, activation_log[map_func(np.argmax(softmax_id_val, axis=-1))], metric)
    else:
        score_id = she_distance(feature_id_val, activation_log[np.argmax(softmax_id_val, axis=-1)], metric)


    for name, softmax_ood, feature_ood in zip(ood_names, softmax_oods.values(),
                                            feature_oods.values()):
        if mapping is not None:
            score_ood = she_distance(feature_ood, activation_log[map_func(np.argmax(softmax_ood, axis=-1))], metric)
        else:
            score_ood = she_distance(feature_ood, activation_log[np.argmax(softmax_ood, axis=-1)], metric)
        auc_ood = auc(score_id, score_ood)[0]
        fpr_ood, _ = fpr_recall(score_id, score_ood, recall)
        result.append(
            dict(method=method, oodset=name, auroc=auc_ood, fpr=fpr_ood))
        print(f'{method}: {name} auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}')
    df = pd.DataFrame(result)
    dfs.append(df)

    near_auroc, near_fpr, far_auroc, far_fpr = near_far_mean(result, list_near_ood, list_far_ood)

    print(f"far mean auroc: {far_auroc:.2%}, far mean fpr: {far_fpr:.2%}")
    print(f"near mean auroc: {near_auroc:.2%}, near mean fpr: {near_fpr:.2%}")
    print(f'mean auroc {df.auroc.mean():.2%}, {df.fpr.mean():.2%}')

    # ---------------------------------------
    method = 'GEN'
    print(f'\n{method}')
    result = []

    gamma_gen = 0.1
    M_gen = 100
    score_id = generalized_entropy(softmax_id_val, gamma_gen, M_gen)

    for name, softmax_ood, feature_ood in zip(ood_names, softmax_oods.values(),
                                            feature_oods.values()):
        score_ood = generalized_entropy(softmax_ood, gamma_gen, M_gen)
        auc_ood = auc(score_id, score_ood)[0]
        fpr_ood, _ = fpr_recall(score_id, score_ood, recall)
        result.append(
            dict(method=method, oodset=name, auroc=auc_ood, fpr=fpr_ood))
        print(f'{method}: {name} auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}')
    df = pd.DataFrame(result)
    dfs.append(df)

    near_auroc, near_fpr, far_auroc, far_fpr = near_far_mean(result, list_near_ood, list_far_ood)

    print(f"far mean auroc: {far_auroc:.2%}, far mean fpr: {far_fpr:.2%}")
    print(f"near mean auroc: {near_auroc:.2%}, near mean fpr: {near_fpr:.2%}")
    print(f'mean auroc {df.auroc.mean():.2%}, {df.fpr.mean():.2%}')

    # ---------------------------------------
    method = 'NNGuide'
    print(f'\n{method}')
    result = []

    mask = np.zeros(feature_id_train.shape[0], dtype=bool)
    mask[:600] = True
    np.random.shuffle(mask)

    normalizer = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True) + 1e-10
    bank_feas = normalizer(feature_id_train[mask])
    bank_logits = logit_id_train[mask]
    bank_confs = logsumexp(bank_logits, axis=-1)
    bank_guide = bank_feas * bank_confs[:, None]

    feature_id_val_norm = normalizer(feature_id_val)
    energy_id_val = logsumexp(logit_id_val, axis=-1)
    conf_id_val = knn_score(bank_guide, feature_id_val_norm, k=100)
    score_id = conf_id_val * energy_id_val

    for name, logit_ood, feature_ood in zip(ood_names, logit_oods.values(),
                                            feature_oods.values()):
        feature_ood_norm = normalizer(feature_ood)
        energy_ood = logsumexp(logit_ood, axis=-1)
        conf_ood = knn_score(bank_guide, feature_ood_norm, k=100)
        score_ood = conf_ood * energy_ood
        auc_ood = auc(score_id, score_ood)[0]
        fpr_ood, _ = fpr_recall(score_id, score_ood, recall)
        result.append(
            dict(method=method, oodset=name, auroc=auc_ood, fpr=fpr_ood))
        print(f'{method}: {name} auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}')
    df = pd.DataFrame(result)
    dfs.append(df)

    near_auroc, near_fpr, far_auroc, far_fpr = near_far_mean(result, list_near_ood, list_far_ood)

    print(f"far mean auroc: {far_auroc:.2%}, far mean fpr: {far_fpr:.2%}")
    print(f"near mean auroc: {near_auroc:.2%}, near mean fpr: {near_fpr:.2%}")
    print(f'mean auroc {df.auroc.mean():.2%}, {df.fpr.mean():.2%}')

    # ---------------------------------------
    method = 'SCALE'
    print(f'\n{method}')
    result = []
    
    percentile_scale = 85 #Imagenet

    feature_id_val_torch = torch.from_numpy(np.copy(feature_id_val))
    feature_id_val_torch = scale(feature_id_val_torch.view(feature_id_val_torch.size(0), -1, 1, 1), percentile_scale)
    feature_id_val_torch = feature_id_val_torch.view(feature_id_val_torch.size(0), -1)
    feature_id_val_scale = feature_id_val_torch.cpu().detach().numpy()
    logit_scale_id_val = feature_id_val_scale @ w.T + b
    score_id = logsumexp(logit_scale_id_val, axis=-1)


    for name, logit_ood, feature_ood in zip(ood_names, logit_oods.values(),
                                            feature_oods.values()):
        feature_ood_torch = torch.from_numpy(np.copy(feature_ood))
        feature_ood_torch = scale(feature_ood_torch.view(feature_ood_torch.size(0), -1, 1, 1), percentile_scale)
        feature_ood_torch = feature_ood_torch.view(feature_ood_torch.size(0), -1)
        feature_ood_scale = feature_ood_torch.cpu().detach().numpy()
        logit_scale_ood = feature_ood_scale @ w.T + b
        score_ood = logsumexp(logit_scale_ood, axis=-1)

        auc_ood = auc(score_id, score_ood)[0]
        fpr_ood, _ = fpr_recall(score_id, score_ood, recall)
        result.append(
            dict(method=method, oodset=name, auroc=auc_ood, fpr=fpr_ood))
        print(f'{method}: {name} auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}')
    df = pd.DataFrame(result)
    dfs.append(df)

    near_auroc, near_fpr, far_auroc, far_fpr = near_far_mean(result, list_near_ood, list_far_ood)

    print(f"far mean auroc: {far_auroc:.2%}, far mean fpr: {far_fpr:.2%}")
    print(f"near mean auroc: {near_auroc:.2%}, near mean fpr: {near_fpr:.2%}")
    print(f'mean auroc {df.auroc.mean():.2%}, {df.fpr.mean():.2%}')

    # ---------------------------------------
    method = 'fDBD'
    print(f'\n{method}')
    result = []

    num_classes = train_labels.max() + 1
    denominator_matrix = np.zeros((num_classes, num_classes))
    for p in range(num_classes):
        w_p = w - w[p, :]
        denominator = np.linalg.norm(w_p, axis=1)
        denominator[p] = 1
        denominator_matrix[p, :] = denominator
    
    nn_idx_id_val = logit_id_val.argmax(axis=-1)
    logits_id_val_sub = np.abs(logit_id_val - np.max(logit_id_val, axis=-1, keepdims=True))
    score_id = np.sum(logits_id_val_sub / denominator_matrix[nn_idx_id_val], axis=1) / np.linalg.norm(feature_id_val - feature_mean, axis=1)

    for name, logit_ood, feature_ood in zip(ood_names, logit_oods.values(),
                                            feature_oods.values()):
        start = time.perf_counter()
        nn_idx_ood = logit_ood.argmax(axis=-1)
        logits_ood_sub = np.abs(logit_ood - np.max(logit_ood, axis=-1, keepdims=True))
        score_ood = np.sum(logits_ood_sub / denominator_matrix[nn_idx_ood], axis=1) / np.linalg.norm(feature_ood - feature_mean, axis=1)
        mid = time.perf_counter()
        print(f"Latency of fDBD on {name}: {mid - start:.6f} seconds")

        auc_ood = auc(score_id, score_ood)[0]
        fpr_ood, _ = fpr_recall(score_id, score_ood, recall)
        result.append(
            dict(method=method, oodset=name, auroc=auc_ood, fpr=fpr_ood))
        print(f'{method}: {name} auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}')
    df = pd.DataFrame(result)
    dfs.append(df)

    near_auroc, near_fpr, far_auroc, far_fpr = near_far_mean(result, list_near_ood, list_far_ood)

    print(f"far mean auroc: {far_auroc:.2%}, far mean fpr: {far_fpr:.2%}")
    print(f"near mean auroc: {near_auroc:.2%}, near mean fpr: {near_fpr:.2%}")
    print(f'mean auroc {df.auroc.mean():.2%}, {df.fpr.mean():.2%}')

    # ---------------------------------------
    method = 'Residual'
    print(f'\n{method}')
    result = []

    score_id = -norm(np.matmul(feature_id_val - u, NS), axis=-1)
    for name, logit_ood, feature_ood in zip(ood_names, logit_oods.values(),
                                            feature_oods.values()):
        score_ood = -norm(np.matmul(feature_ood - u, NS), axis=-1)
        auc_ood = auc(score_id, score_ood)[0]
        fpr_ood, _ = fpr_recall(score_id, score_ood, recall)
        result.append(
            dict(method=method, oodset=name, auroc=auc_ood, fpr=fpr_ood))
        print(f'{method}: {name} auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}')
    df = pd.DataFrame(result)
    dfs.append(df)

    near_auroc, near_fpr, far_auroc, far_fpr = near_far_mean(result, list_near_ood, list_far_ood)

    print(f"far mean auroc: {far_auroc:.2%}, far mean fpr: {far_fpr:.2%}")
    print(f"near mean auroc: {near_auroc:.2%}, near mean fpr: {near_fpr:.2%}")
    print(f'mean auroc {df.auroc.mean():.2%}, {df.fpr.mean():.2%}')

    # ---------------------------------------
    method = 'GradNorm'
    print(f'\n{method}')
    result = []
    score_id = gradnorm(feature_id_val, w, b)
    for name, feature_ood in feature_oods.items():
        score_ood = gradnorm(feature_ood, w, b)
        auc_ood = auc(score_id, score_ood)[0]
        fpr_ood, _ = fpr_recall(score_id, score_ood, recall)
        result.append(
            dict(method=method, oodset=name, auroc=auc_ood, fpr=fpr_ood))
        print(f'{method}: {name} auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}')
    df = pd.DataFrame(result)
    dfs.append(df)

    near_auroc, near_fpr, far_auroc, far_fpr = near_far_mean(result, list_near_ood, list_far_ood)

    print(f"far mean auroc: {far_auroc:.2%}, far mean fpr: {far_fpr:.2%}")
    print(f"near mean auroc: {near_auroc:.2%}, near mean fpr: {near_fpr:.2%}")
    print(f'mean auroc {df.auroc.mean():.2%}, {df.fpr.mean():.2%}')

    # ---------------------------------------
    method = 'Mahalanobis' #to factorise if necessary
    print(f'\n{method}')
    result = []

    print('computing classwise mean feature...')
    train_means = []
    train_feat_centered = []
    for i in tqdm(range(train_labels.max() + 1)):
        fs = feature_id_train[train_labels == i]
        _m = fs.mean(axis=0)
        train_means.append(_m)
        train_feat_centered.extend(fs - _m)

    print('computing precision matrix...')
    ec = EmpiricalCovariance(assume_centered=True)
    ec.fit(np.array(train_feat_centered).astype(np.float64))

    print('go to gpu...')
    mean = torch.from_numpy(np.array(train_means)).cuda().float()
    prec = torch.from_numpy(ec.precision_).cuda().float()

    score_id = -np.array(
        [(((f - mean) @ prec) * (f - mean)).sum(axis=-1).min().cpu().item()
         for f in tqdm(torch.from_numpy(feature_id_val).cuda().float())])
    for name, feature_ood in feature_oods.items():
        score_ood = -np.array([
            (((f - mean) @ prec) * (f - mean)).sum(axis=-1).min().cpu().item()
            for f in tqdm(torch.from_numpy(feature_ood).cuda().float())
        ])
        auc_ood = auc(score_id, score_ood)[0]
        fpr_ood, _ = fpr_recall(score_id, score_ood, recall)
        result.append(
            dict(method=method, oodset=name, auroc=auc_ood, fpr=fpr_ood))
        print(f'{method}: {name} auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}')
    df = pd.DataFrame(result)
    dfs.append(df)

    near_auroc, near_fpr, far_auroc, far_fpr = near_far_mean(result, list_near_ood, list_far_ood)

    print(f"far mean auroc: {far_auroc:.2%}, far mean fpr: {far_fpr:.2%}")
    print(f"near mean auroc: {near_auroc:.2%}, near mean fpr: {near_fpr:.2%}")
    print(f'mean auroc {df.auroc.mean():.2%}, {df.fpr.mean():.2%}')

    # ---------------------------------------
    method = 'KL-Matching' 
    print(f'\n{method}')
    result = []

    print('computing classwise mean softmax...')
    pred_labels_train = np.argmax(softmax_id_train, axis=-1)
    mean_softmax_train = [
        softmax_id_train[pred_labels_train == i].mean(axis=0)
        for i in tqdm(range(1000))
    ]

    score_id = -pairwise_distances_argmin_min(
        softmax_id_val, np.array(mean_softmax_train), metric=kl)[1]

    for name, softmax_ood in softmax_oods.items():
        score_ood = -pairwise_distances_argmin_min(
            softmax_ood, np.array(mean_softmax_train), metric=kl)[1]
        auc_ood = auc(score_id, score_ood)[0]
        fpr_ood, _ = fpr_recall(score_id, score_ood, recall)
        result.append(
            dict(method=method, oodset=name, auroc=auc_ood, fpr=fpr_ood))
        print(f'{method}: {name} auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}')
    df = pd.DataFrame(result)
    dfs.append(df)

    near_auroc, near_fpr, far_auroc, far_fpr = near_far_mean(result, list_near_ood, list_far_ood)

    print(f"far mean auroc: {far_auroc:.2%}, far mean fpr: {far_fpr:.2%}")
    print(f"near mean auroc: {near_auroc:.2%}, near mean fpr: {near_fpr:.2%}")
    print(f'mean auroc {df.auroc.mean():.2%}, {df.fpr.mean():.2%}')

    # ---------------------------------------
    method = 'PCA' #to factorise if necessary
    print(f'\n{method}')
    result = []

    feature_mean = np.mean(feature_id_train, axis=0)

    threshold_pca = np.quantile(feature_id_train, 0.92)

    feature_id_val_clip_pca = np.clip(feature_id_val, a_min=None, a_max=threshold_pca)

    logit_id_val = feature_id_val.clip(min=None, max=threshold_pca) @ w.T + b
    rec_norm = np.linalg.norm((feature_id_val - feature_mean) @ (np.identity(dim) - M), axis=-1)
    r_id = rec_norm / np.linalg.norm(feature_id_val_clip_pca, axis=-1)

    score_id = logsumexp(logit_id_val, axis=-1) * (1.0 - r_id)


    for name, logit_ood, feature_ood in zip(ood_names, logit_oods.values(),
                                            feature_oods.values()):
        feature_ood_clip = np.clip(feature_ood, a_min=None, a_max=threshold_pca)
        logit_ood_clip = feature_ood_clip @ w.T + b
        rec_ood = np.linalg.norm((feature_ood_clip - feature_mean) @ (np.identity(dim) - M), axis=-1)
        r_ood = rec_ood / np.linalg.norm(feature_ood_clip, axis=-1)
        score_ood = logsumexp(logit_ood_clip, axis=-1) * (1.0 - r_ood)
        auc_ood = auc(score_id, score_ood)[0]
        fpr_ood, _ = fpr_recall(score_id, score_ood, recall)
        result.append(
            dict(method=method, oodset=name, auroc=auc_ood, fpr=fpr_ood))
        print(f'{method}: {name} auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}')
    df = pd.DataFrame(result)
    dfs.append(df)

    near_auroc, near_fpr, far_auroc, far_fpr = near_far_mean(result, list_near_ood, list_far_ood)

    print(f"far mean auroc: {far_auroc:.2%}, far mean fpr: {far_fpr:.2%}")
    print(f"near mean auroc: {near_auroc:.2%}, near mean fpr: {near_fpr:.2%}")
    print(f'mean auroc {df.auroc.mean():.2%}, {df.fpr.mean():.2%}')

    # ---------------------------------------
    method = 'MME' #Combination of VRA + SCALE + PCA + NME (perso) + fDBD
    print(f'\n{method}')
    result = []

    percentile_scale = 90

    feature_id_val_torch = torch.from_numpy(np.copy(feature_id_val))
    feature_id_val_torch = scale(feature_id_val_torch.view(feature_id_val_torch.size(0), -1, 1, 1), percentile_scale)
    feature_id_val_torch = feature_id_val_torch.view(feature_id_val_torch.size(0), -1)
    feature_id_val_scale = feature_id_val_torch.cpu().detach().numpy()
    logit_scale_id_val = feature_id_val_scale @ w.T + b

    vlogit_id_val = norm(np.matmul(feature_id_val_clip_VRA - u, NS), axis=-1) * alpha

    rec_norm = np.linalg.norm((feature_id_val_clip_VRA - feature_mean) @ (np.identity(dim) - M), axis=-1)
    r_id = rec_norm / np.linalg.norm(feature_id_val_clip_VRA, axis=-1)
    
    vectors_id = (feature_id_val.T / (np.linalg.norm(feature_id_val.T, axis=0) + EPSILON)).T

    distances_id = cdist(class_means, vectors_id, "sqeuclidean")  # [nb_classes, N]
    distances_id = distances_id.T

    logit_id_val_clip = np.clip(feature_id_val, a_min=clip_low, a_max=clip_high) @ w.T + b

    nn_idx_id_val = logit_id_val.argmax(axis=-1)
    logits_id_val_sub = np.abs(logit_id_val_clip_VRA - np.max(logit_id_val_clip_VRA, axis=-1, keepdims=True))
    fdbd_id = np.sum(logits_id_val_sub / denominator_matrix[nn_idx_id_val], axis=1) / np.linalg.norm(feature_id_val_clip_VRA - feature_mean, axis=1)

    score_id = mme(logit_scale_id_val, distances_id, r_id, vlogit_id_val, fdbd_id, mapping=mapping)


    for name, logit_ood, feature_ood_clip_VRA, feature_ood in zip(ood_names, logit_oods.values(),
                                            feature_oods_clip_VRA.values(), feature_oods.values()):

        feature_ood_torch = torch.from_numpy(np.copy(feature_ood))
        feature_ood_torch = scale(feature_ood_torch.view(feature_ood_torch.size(0), -1, 1, 1), percentile_scale)
        feature_ood_torch = feature_ood_torch.view(feature_ood_torch.size(0), -1)
        feature_ood_scale = feature_ood_torch.cpu().detach().numpy()
        logit_scale_ood = feature_ood_scale @ w.T + b
        
        vlogit_ood = norm(np.matmul(feature_ood_clip_VRA - u, NS), axis=-1) * alpha

        rec_norm = np.linalg.norm((feature_ood_clip_VRA - feature_mean) @ (np.identity(dim) - M), axis=-1)
        r_ood = rec_norm / np.linalg.norm(feature_ood_clip_VRA, axis=-1)

        vectors_ood = (feature_ood.T / (np.linalg.norm(feature_ood.T, axis=0) + EPSILON)).T

        distances_ood = cdist(class_means, vectors_ood, "sqeuclidean")  # [nb_classes, N]
        distances_ood = distances_ood.T

        #logit_ood_clip = np.clip(feature_ood, a_min=clip_low, a_max=clip_high) @ w.T + b
        logit_ood_clip = feature_ood_clip_VRA @ w.T + b

        nn_idx_ood = logit_ood.argmax(axis=-1)
        logits_ood_sub = np.abs(logit_ood_clip - np.max(logit_ood_clip, axis=-1, keepdims=True))
        fdbd_ood = np.sum(logits_ood_sub / denominator_matrix[nn_idx_ood], axis=1) / np.linalg.norm(feature_ood_clip_VRA - feature_mean, axis=1)

        score_ood = mme(logit_scale_ood, distances_ood, r_ood, vlogit_ood, fdbd_ood, mapping=mapping)

        auc_ood = auc(score_id, score_ood)[0]
        fpr_ood, _ = fpr_recall(score_id, score_ood, recall)
        result.append(
            dict(method=method, oodset=name, auroc=auc_ood, fpr=fpr_ood))
        print(f'{method}: {name} auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}')
    df = pd.DataFrame(result)
    dfs.append(df)

    near_auroc, near_fpr, far_auroc, far_fpr = near_far_mean(result, list_near_ood, list_far_ood)

    print(f"far mean auroc: {far_auroc:.2%}, far mean fpr: {far_fpr:.2%}")
    print(f"near mean auroc: {near_auroc:.2%}, near mean fpr: {near_fpr:.2%}")
    print(f'mean auroc {df.auroc.mean():.2%}, {df.fpr.mean():.2%}')


if __name__ == '__main__':
    main()
