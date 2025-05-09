import argparse
import yaml
import pathlib
from typing import List, Optional
import yaml
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from utils.builder import build_pipeline
from utils.builder import build_metrics
from utils.builder import build_models
from torchvision import transforms
import tifffile
import os
IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
                    'tif', 'tiff', 'webp'}
from PIL import Image
class RandomDegradationDataset(Dataset):

    def __init__(self,
                 path: str,
                 pipelines: list,
                 global_bin_index: Optional[int]=None):
        self.path = path
        self.pipelines = pipelines
        self.global_bin_index = global_bin_index
        self.num_possible_bins = 1
        self.degradations = []
        for param in pipelines:
            d = build_pipeline(param)
            self.num_possible_bins *= len(d.p)
            self.degradations.append(d)
        self.img_paths = self._load_img_paths()
        self.transforms = transforms.Compose([
            transforms.RandomCrop(size = (228,228))
            # transforms.RandomHorizontalFlip(p=0.5),
            # transforms.RandomRotation()
        ])

    def _load_img_paths(self) -> List[str]:
        path = pathlib.Path(self.path)
        files = sorted([file for ext in IMAGE_EXTENSIONS
                        for file in path.glob('*.{}'.format(ext))])
        return files
    

    def __len__(self):
        return len(self.img_paths)
    
    def random_crop(self, image, crop_size):
    
            height, width, _ = image.shape

            if height >= crop_size and width >= crop_size:
                
                x_start = np.random.randint(0, width - crop_size + 1)
                y_start = np.random.randint(0, height - crop_size + 1)

                
                cropped_image = image[y_start:y_start + crop_size, x_start:x_start + crop_size]

                return cropped_image

    def __getitem__(self, n):

        results = dict()
        img_path = str(self.img_paths[n])
        
        x = cv2.imread(img_path)
        # x = self.random_crop(x,606)
        results['bin_indices'] = []
        results['path'] = []
        results['path'].append(img_path)
        if self.global_bin_index is not None:
            bin_indices = []
            gind = self.global_bin_index
            for d in self.degradations:
                lidx = gind % len(d.p)
                gind = gind // len(d.p)
                bin_indices.append(lidx)
        
        else:
            bin_indices = [None] * len(self.degradations)
        
        for d, b_idx in zip(self.degradations, bin_indices):
            x, idx = d(x, b_idx)
            
            
            if b_idx is not None:
                assert idx == b_idx
            results['bin_indices'].append(idx)
        x = np.transpose(x, (2, 0, 1))
        x = np.clip(x, 0, 255).astype(np.uint8)
        results['img'] = torch.from_numpy(x)

        return results
    
class TargetDataset(Dataset):

    def __init__(self,
                 path: str,
                 pipelines: list,
                 global_bin_index: Optional[int]=None):
        self.path = path
        self.pipelines = pipelines
        self.global_bin_index = global_bin_index
        self.num_possible_bins = 1
        self.degradations = []
        for param in pipelines:
            d = build_pipeline(param)
            self.num_possible_bins *= len(d.p)
            self.degradations.append(d)
        self.img_paths = self._load_img_paths()
        
    

    def _load_img_paths(self) -> List[str]:
        path = pathlib.Path(self.path)
        files = sorted([file for ext in IMAGE_EXTENSIONS
                        for file in path.glob('*.{}'.format(ext))])
        return files

    def __len__(self):
        return len(self.img_paths)
        
    def random_crop(self, image, crop_size):
    
            height, width, _ = image.shape

            if height >= crop_size and width >= crop_size:
                
                x_start = np.random.randint(0, width - crop_size + 1)
                y_start = np.random.randint(0, height - crop_size + 1)

                
                cropped_image = image[y_start:y_start + crop_size, x_start:x_start + crop_size]

                return cropped_image
            
    def __getitem__(self, n):
        results = dict()
        img_path = str(self.img_paths[n])
        x = cv2.imread(img_path)
        # x = self.random_crop(x,303)
        results['bin_indices'] = []
        results['path'] = []
        results['path'].append(img_path)

        if self.global_bin_index is not None:
            bin_indices = []
            gind = self.global_bin_index
            for d in self.degradations:
                lidx = gind % len(d.p)
                gind = gind // len(d.p)
                bin_indices.append(lidx)
        
        else:
            bin_indices = [None] * len(self.degradations)
        
        for d, b_idx in zip(self.degradations, bin_indices):
            x, idx = d(x, b_idx)
            
            
            if b_idx is not None:
                assert idx == b_idx
            results['bin_indices'].append(idx)
        if len(x.shape) == 2 :
            x = np.expand_dims(x, 2)

        x = np.transpose(x, (2, 0, 1))
        x = np.clip(x, 0, 255).astype(np.uint8)
        results['img'] = torch.from_numpy(x)

        return results   
    
class SynDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, 
                 pipelines: list,
                 weight,
                 global_bin_index: Optional[int]=None,
                 augmentation=False,
                 transform=None, task=None, 
                 opts=None, ):
        self.data_dir = data_dir
        self.transform = transform
        self.task = task
        self.opts = opts
        self.pipelines = pipelines
        self.global_bin_index = global_bin_index
        self.num_possible_bins = 1
        self.degradations = []
        self.weight = weight
        self.bin_indices = []    

        for param in pipelines:
            d = build_pipeline(param)
            self.num_possible_bins *= len(d.p)
            self.degradations.append(d)
        dshape = [len(d.p) for d in self.degradations]
        self.augmentation = augmentation
        # weight normalization to numpy
        self.weight = np.array(self.weight)

        for n in range(len(self.weight)):
            gind = n
            bin_dice = []
            for i in dshape:
                lidx = gind % i
                gind = gind // i
                bin_dice.append(lidx)
            self.bin_indices.append(str(bin_dice)) 
        

        lst_data = os.listdir(self.data_dir)
        lst_data = [f for f in lst_data if f.endswith('bmp') | f.endswith('jpg') | f.endswith('png') | f.endswith('tif')]
        lst_data.sort()
        self.lst_data = lst_data

    def __len__(self):
        return len(self.lst_data)

    def __getitem__(self, index):

        data = dict()
        HR = cv2.imread(os.path.join(self.data_dir, self.lst_data[index]))
        key = np.random.choice(np.array(self.bin_indices),1, p=self.weight)
        key = [int(item) for item in key[0].strip('[]').split(',')]
        # key = [0,0,0,2]
        # Updated at Apr 5 2020
        data = {'label': HR
                }    
        data['input'] = data['label']

            
        for d, b_idx in zip(self.degradations, key):

            data['input'], idx = d(data['input'], b_idx)
            if b_idx is not None:
                assert idx == b_idx

        for key in data.keys():
            data[key] = np.transpose(data[key], (2, 0, 1))
            data[key] = np.clip(data[key], 0, 255).astype(np.uint8)
            data[key] = torch.from_numpy(data[key])
        
        data['path'] = []
        data['path'].append(os.path.join(self.data_dir, self.lst_data[index]))

        return data