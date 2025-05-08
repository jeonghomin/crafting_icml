from copy import deepcopy
from typing import Tuple

import cv2
import numpy as np

from utils.builder import PIPELINES


@PIPELINES.register_module()
class RandomJPEGCompression:

    def __init__(self,
                 bins: int or list,
                 quality_range: Tuple[int, int],
                 color_type: str='color'):
        self.bins = bins
        self.quality_range = quality_range
        self.color_type = color_type
        num_bins = bins if type(bins) is int else len(bins)
        quality_length = quality_range[1] - quality_range[0]
        if type(bins) is int:
            interval = quality_length / self.bins
            init = quality_range[0]
            self.bins = [(init + interval*n, init + interval*(n+1))
                                for n in range(bins)]
        else:
            self.bins = bins
        self.p = [1.0 / num_bins for _ in range(num_bins)]

    def __call__(self, x: np.ndarray, idx: int=None):
        if idx is None:
            idx = np.random.choice(len(self.bins), size=1, p=self.p)[0]
        quality_range = self.bins[idx]
        jpeg_param = round(np.random.uniform(quality_range[0], quality_range[1]))
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_param]
        _, x_encoded = cv2.imencode('.jpg', x, encode_param)
        color_code = 1
        if self.color_type != 'color':
            color_code = 0
        x_decoded = cv2.imdecode(x_encoded, color_code)
        return x_decoded, idx

    def update_weight(self, new_p: list):
        assert len(self.p) == len(new_p)
        self.p = deepcopy(new_p)

if __name__ == '__main__':
    random_jpeg = RandomJPEGCompression(bins=5, quality_range=[40, 90])
    x = np.random.rand(128, 128, 3)
    print('sample weights:', random_jpeg.p)
    print('bins:', random_jpeg.bins)
    x_blurred, idx = random_jpeg(x)
    print(x.shape, x_blurred.shape, idx)