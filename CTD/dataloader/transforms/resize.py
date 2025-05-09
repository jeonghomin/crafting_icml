from copy import deepcopy
from typing import Tuple

import cv2
import numpy as np

from utils.builder import PIPELINES


@PIPELINES.register_module()
class RandomResize:

    def __init__(self,
                 bins: int or list,
                 scale_range: Tuple[int, int],
                 arb_size : None,
                 interpolation: str):
        self.bins = bins
        self.scale_range = scale_range
        self.interpolation = interpolation
        self.arb_size = arb_size
        num_bins = bins if type(bins) is int else len(bins)
        
        quality_length = scale_range[1] - scale_range[0]
        if type(bins) is int:
            interval = quality_length / self.bins
            init = scale_range[0]
            self.bins = [(init + interval*n, init + interval*(n+1))
                                for n in range(bins)]
        else:
            self.bins = bins
        self.p = [1.0 / num_bins for _ in range(num_bins)]
        
        self.resize_method = {
            'nearest': cv2.INTER_NEAREST,
            'bilinear': cv2.INTER_LINEAR,
            'bicubic': cv2.INTER_CUBIC}[self.interpolation]

    def __call__(self, x: np.ndarray, idx: int=None):
        if idx is None:
            idx = np.random.choice(len(self.bins), size=1, p=self.p)[0]
        resize_range = self.bins[idx]
        scale_factor = np.random.uniform(resize_range[0], resize_range[1])
        
        H, W = x.shape[0:2]
        if self.arb_size is not None : 
            x = cv2.resize(x,
                       (int(self.arb_size), int(self.arb_size)),
                       interpolation=self.resize_method)
        else  :
            x = cv2.resize(x,
                       (int(W*scale_factor), int(H*scale_factor)),
                       interpolation=self.resize_method)
        return x, idx

    def update_weight(self, new_p: list):
        assert len(self.p) == len(new_p)
        self.p = deepcopy(new_p)

if __name__ == '__main__':
    random_jpeg = RandomResize(bins=1, scale_range=[0.25, 0.25], interpolation='bicubic')
    x = np.random.rand(128, 128, 3)
    print('sample weights:', random_jpeg.p)
    print('bins:', random_jpeg.bins)
    x_blurred, idx = random_jpeg(x)
    print(x.shape, x_blurred.shape, idx)