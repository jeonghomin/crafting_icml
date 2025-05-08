from copy import deepcopy
from typing import Tuple

import numpy as np

from utils.builder import PIPELINES


@PIPELINES.register_module()
class RandomNoise:

    def __init__(self,
                 bins: int or list,
                 noise_range: Tuple[float, float],
                 is_gray_noise: bool=True):
        self.bins = bins
        self.noise_range = noise_range
        self.is_gray_noise = is_gray_noise
        num_bins = bins if type(bins) is int else len(bins)
        sigma_length = noise_range[1] - noise_range[0]
        if type(bins) is int:
            interval = sigma_length / self.bins
            init = noise_range[0]
            self.bins = [(init + interval*n, init + interval*(n+1))
                                for n in range(bins)]
        else:
            self.bins = bins
        self.p = [1.0 / num_bins for _ in range(num_bins)]

    def __call__(self, x: np.ndarray, idx: int=None):
        if idx is None:
            idx = np.random.choice(len(self.bins), size=1, p=self.p)[0]
        sigma_range = self.bins[idx]
        sigma = np.random.uniform(sigma_range[0], sigma_range[1])
        noise = np.float32(np.random.randn(*(x.shape))) * sigma
        if self.is_gray_noise:
            noise = noise[:, :, :1]
        x_noised = x + noise
        return x_noised, idx

    def update_weight(self, new_p: list):
        assert len(self.p) == len(new_p)
        self.p = deepcopy(new_p)

if __name__ == '__main__':
    random_noise = RandomNoise(bins=5, noise_range=[0, 50])
    x = np.random.rand(128, 128, 3)
    print('sample weights:', random_noise.p)
    print('bins:', random_noise.bins)
    x_noised, idx = random_noise(x)
    print(x.shape, x_noised.shape, idx)