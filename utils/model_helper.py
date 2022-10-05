import torch
import os
import numpy as np

def save_model(args, models):
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    for idx, model in enumerate(models):
        torch.save(model.state_dict(), os.path.join(args.model_dir, 'best_{}.pkl'.format(idx)))

def load_model(args, models):
    for idx, model in enumerate(models):
        model.load_state_dict(torch.load(os.path.join(args.model_dir, 'best_{}.pkl'.format(idx))))
        model.eval()
    return models

def pdist2sq(A, B):
    # return pairwise euclidean difference matrix
    D = torch.sum((torch.unsqueeze(A, 1) - torch.unsqueeze(B, 0))**2, 2) 
    return D

def rbf_kernel(A, B, rbf_sigma=1):
    rbf_sigma = torch.tensor(rbf_sigma)
    return torch.exp(-pdist2sq(A, B) / torch.square(rbf_sigma) *.5)

def calculate_mmd(A, B, rbf_sigma=1):
    Kaa = rbf_kernel(A, A, rbf_sigma)
    Kab = rbf_kernel(A, B, rbf_sigma)
    Kbb = rbf_kernel(B, B, rbf_sigma)
    mmd = Kaa.mean() - 2 * Kab.mean() + Kbb.mean()
    return mmd

def IPM_loss(x, t, w, k=5, rbf_sigma=1):
    _, idx = torch.sort(t)
    xw = x * w
    sorted_x = x[idx]
    sorted_xw = xw[idx]
    split_x = torch.tensor_split(sorted_x, k)
    split_xw = torch.tensor_split(sorted_xw, k)
    loss = torch.zeros(k)
    for i in range(k):
        A = split_xw[i]
        tmp_loss = torch.zeros(k - 1)
        idx = 0
        for j in range(k):
            if i == j:
                continue
            B = split_x[j]
            partial_loss = calculate_mmd(A, B, rbf_sigma)
            tmp_loss[idx] = partial_loss
            idx += 1
        loss[i] = tmp_loss.max()

    return loss.mean()
    