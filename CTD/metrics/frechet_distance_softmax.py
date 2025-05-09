from copy import deepcopy
from typing import Tuple, List

import numpy as np
import torch

from utils.builder import METRICS
from utils.builder import build_models
import pdb
from frechet import euclidean, DiscreteFrechet, FastDiscreteFrechetSparse, FastDiscreteFrechetMatrix 
from frechetdist import frdist

def custom_softmax_formula(x, i:int, alpha=10, j=1, n=75, eps=1e-8):
    numerator = torch.exp((1 - x[i]) ** alpha) - 1
    denominator = torch.exp((1 - x[j:n+1])**alpha - 1)
    sum_result = torch.sum(denominator)
    custom_softmax_values = numerator / (sum_result + eps)
    return custom_softmax_values

@METRICS.register_module()
class FrechetDistanceSoftmax(torch.nn.Module):

    def __init__(self, model_cfg: dict):
        super().__init__()
        self.model = build_models(model_cfg)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        # Feature Extraction with AlexNet 
        f1 = self.model(x)
        f2 = self.model(y)

        f1_d = f1.detach()
        f2_d = f2.detach()
        # Calculate Distance between f1, f2 (You can choose which functions to calculate e.g (DiscreteFrechet, Fast..))
        dist = frdist(f1_d,f2_d)

        # Normalization in interval of [0,1]
        normalized_dist = (dist - dist.min()) / (dist.max() - dist.min())
        softmax_dist = torch.softmax(normalized_dist, dim=0)
        
        return normalized_dist

if __name__ == '__main__':
    from models import *
    cfg = dict(type='AlexNetImageNet')
    metric = FrechetDistanceSoftmax(cfg)
    # 나중에 72 -> 224로 resize해서 바꿔줄 것
    x = torch.zeros(1, 3, 224, 224)
    y = torch.zeros(1, 3, 224, 224)
    dist = metric(x, y)
    print(dist)
