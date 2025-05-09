import argparse
import yaml
import pathlib
from typing import List, Optional
import yaml
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from scipy import linalg
from tqdm import tqdm
from datetime import datetime
from dataloader import *
from models import *
from utils.builder import build_pipeline , PIPELINES
from utils.builder import build_metrics
from utils.builder import build_models
from utils.mathfunc import calculate_statistics, calculate_frechet_distance
from dataloader.RandomDataset import RandomDegradationDataset, TargetDataset
import pickle
import pdb
import os
datestring = datetime.now().strftime('%y%m%d-%H%M%S')
def get_features(dataset: Dataset,
                 model: torch.nn.Module,
                 batch_size=50,
                 dims=2048,
                 num_workers=1):
    device = next(model.parameters()).device
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             drop_last=False,
                                             num_workers=num_workers)
    size = len(dataset)
    pred_arr = torch.zeros((size, dims), device=device)

    start_idx = 0
    for data in dataloader:
        batch = data['img'].to(device) / 255.0

        with torch.no_grad():
            pred = model(batch)
        pred_arr[start_idx:start_idx + pred.shape[0]] = pred
        start_idx = start_idx + pred.shape[0]

    return pred_arr


def custom_softmax_formula(x: torch.Tensor, alpha: float=25, eps: float=1e-8):
    x_exp = torch.exp((1 - x) ** alpha) - 1
    denominator = torch.sum(x_exp, dim=0)
    custom_softmax_values = x_exp / (denominator + eps)
    return custom_softmax_values

if __name__ == "__main__":

    device = 'cuda'
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="yaml file.")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    model = build_models(config['model']).cuda()

    target_dataset = TargetDataset(config['dataset']['tgt_path'], [])
 
    target_features = get_features(target_dataset,
                                   model,
                                   dims=model.feature_dim)
    target_features = target_features.to(device)
    t_mean, t_var = calculate_statistics(target_features)

    N = RandomDegradationDataset(config['dataset']['src_path'], config['degradations']).num_possible_bins
  
    dists = torch.zeros((N,), device=device)
    
    for n in tqdm(range(N), total=N, ncols=0):
        synthetic_dataset = RandomDegradationDataset(config['dataset']['src_path'],
                                                     config['degradations'],
                                                     global_bin_index=n)
        # pdb.set_trace()
        synthetic_features = get_features(synthetic_dataset,
                                          model,
                                          dims=model.feature_dim)
        synthetic_features = synthetic_features.to(device)
        s_mean, s_var = calculate_statistics(synthetic_features)
        dists[n] = calculate_frechet_distance(t_mean, t_var, s_mean, s_var)

    dists = (dists - dists.min()) / (dists.max() - dists.min())
    w = custom_softmax_formula(dists)
    w = w.cpu()
    dataset = RandomDegradationDataset(config['dataset']['src_path'], config['degradations'])
    dshape = [len(d.p) for d in dataset.degradations]
    print(dshape)    
    print(w) 
    print(torch.topk(w,5))

    top_prob, top_idx = torch.topk(w, 5)

    for p, i in zip(top_prob.numpy(), top_idx.numpy()):
        bin_indices = []
        gind = i
        for d in dshape:
            lidx = gind % d
            gind = gind // d
            bin_indices.append(lidx)
        print(i, p, bin_indices)

    mat = np.zeros(dshape)
    for n in range(N):
        gind = n
        bin_indices = []
        for d in dshape:
            lidx = gind % d
            gind = gind // d
            bin_indices.append([lidx])
        mat[tuple(bin_indices)] = w[n]

    directory_path = "./weights_sar"
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    setting_name=os.path.splitext(args.config)[0].split("/")[-1]
    pickle_name = f"{directory_path}/{setting_name}_{datestring}.pkl"

    with open(pickle_name, 'wb') as f:
        pickle.dump(w, f)


