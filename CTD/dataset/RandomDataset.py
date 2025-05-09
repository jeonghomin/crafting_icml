import argparse
import yaml
import pathlib
from typing import List, Optional
import yaml
import cv2
import numpy as np
import torch
import os, glob
from torch.utils.data import Dataset
from PIL import Image
import random
import math
from torchvision import transforms
from utils.builder import build_pipeline
from utils.builder import build_metrics
from utils.builder import build_models
import pdb
IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
                    'tif', 'tiff', 'webp'}

class TrainSet(Dataset):

    def __init__(self,
                 path: str,
                 pipelines: list,
                 weight,
                 global_bin_index: Optional[int]=None,
                 augmentation=False,
                 ):
        self.path = path
        self.pipelines = pipelines
        self.global_bin_index = global_bin_index
        self.num_possible_bins = 1
        self.degradations = []
        self.HR_list = sorted(glob.glob(os.path.join(path, '*.bmp')))
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
        
    def __len__(self):
        return len(self.HR_list)

    def __getitem__(self, idx):
        results = dict()
        
        HR = cv2.imread(self.HR_list[idx])

        if self.augmentation:
            alpha = random.uniform(0.7, 1.3)
            beta = random.uniform(-20, 20)
            if random.random() < 0.5:
                HR = cv2.convertScaleAbs(HR, alpha=alpha, beta=beta)
                HR = np.clip(HR, 0, 255)
      
        key = np.random.choice(np.array(self.bin_indices),1, p=self.weight)
        key = [int(item) for item in key[0].strip('[]').split(',')]
        
        LR = HR
        # degradation part
        for d, b_idx in zip(self.degradations, key):
            # degradation 값 가져오기 1. uniform 2. ctd sampling
            LR, idx = d(LR, b_idx)
            
            if b_idx is not None:
                assert idx == b_idx
        # h, w, _ = HR.shape
        # LR= np.array(Image.fromarray(HR).resize((w // 4, h // 4), Image.BICUBIC))

        results = {'HR': HR,
                  'LR': LR
                  }

        for key in results.keys():
            results[key] = results[key].astype(np.float32) / 255.
            results[key] = torch.from_numpy(results[key]).permute(2, 0, 1).float()
        # print(results['LR'].shape, results['HR'].shape)
        # exit()
        return results
    

class DIV2kTrainSet(Dataset):

    def __init__(self,
                 path: str,
                 pipelines: list,
                 weight,
                 global_bin_index: Optional[int]=None,
                 augmentation=False,
                 ):
        self.path = path
        self.pipelines = pipelines
        self.global_bin_index = global_bin_index
        self.num_possible_bins = 1
        self.degradations = []
        self.HR_list = sorted(glob.glob(os.path.join(path, '*.png')))
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
        
    def __len__(self):
        return len(self.HR_list)

    def __getitem__(self, idx):

        def random_crop(image, crop_size):

            height, width, _ = image.shape

            if height >= crop_size and width >= crop_size:
                
                x_start = np.random.randint(0, width - crop_size + 1)
                y_start = np.random.randint(0, height - crop_size + 1)

                
                cropped_image = image[y_start:y_start + crop_size, x_start:x_start + crop_size]

                return cropped_image
            
        results = dict()
        
        HR = cv2.imread(self.HR_list[idx])

        HR = random_crop(HR, 96)

        if self.augmentation:
            alpha = random.uniform(0.7, 1.3)
            beta = random.uniform(-20, 20)
            if random.random() < 0.5:
                HR = cv2.convertScaleAbs(HR, alpha=alpha, beta=beta)
                HR = np.clip(HR, 0, 255)
        
        # reading weights
        key = np.random.choice(np.array(self.bin_indices),1, p=self.weight)
        key = [int(item) for item in key[0].strip('[]').split(',')]
        
        LR = HR
        # degradation part
        for d, b_idx in zip(self.degradations, key):
            # degradation 값 가져오기 1. uniform 2. ctd sampling
            LR, idx = d(LR, b_idx)
            
            if b_idx is not None:
                assert idx == b_idx
        # h, w, _ = HR.shape
        # LR= np.array(Image.fromarray(HR).resize((w // 4, h // 4), Image.BICUBIC))

        results = {'HR': HR,
                  'LR': LR,
                  }

        for key in results.keys():
            results[key] = results[key].astype(np.float32) / 255.
            results[key] = torch.from_numpy(results[key]).permute(2, 0, 1).float()
        # print(results['LR'].shape, results['HR'].shape)
        # exit()
        return results
class Set5(Dataset):
    def __init__(self, path):
        self.path = path
        
        self.HR_list = sorted(glob.glob(os.path.join(self.path,"image_SRF_4", '*_HR.png')))
        self.LR_list = sorted(glob.glob(os.path.join(self.path,"image_SRF_4", '*_LR.png')))
        
        # print(os.path.join(self.path, 'HR/*.bmp'))
        # exit()
        # self.scale = args.sr_scale
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.HR_list)

    def __getitem__(self, idx):
        HR_name = os.path.basename(self.HR_list[idx])
        
        HR = cv2.imread(self.HR_list[idx])
        LR = cv2.imread(self.LR_list[idx])

        

        sample = {'HR': HR,
                  'LR': LR
                  }

        for key in sample.keys():
            sample[key] = sample[key].astype(np.float32) / 255.
            sample[key] = torch.from_numpy(sample[key]).permute(2, 0, 1).float()

        sample['HR_name'] = HR_name

        return sample
class Testset(Dataset):
    def __init__(self, path):
        self.path = path
        
        self.HR_list = sorted(glob.glob(os.path.join(self.path, 'HR/*.bmp')))
        self.LR_list = sorted(glob.glob(os.path.join(self.path, 'LR/*.bmp')))

        # print(os.path.join(self.path, 'HR/*.bmp'))
        # exit()
        # self.scale = args.sr_scale
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.HR_list)

    def __getitem__(self, idx):
        HR_name = os.path.basename(self.HR_list[idx])
        
        HR = cv2.imread(self.HR_list[idx])
        LR = cv2.imread(self.LR_list[idx])


        sample = {'HR': HR,
                  'LR': LR
                  }

        for key in sample.keys():
            sample[key] = sample[key].astype(np.float32) / 255.
            sample[key] = torch.from_numpy(sample[key]).permute(2, 0, 1).float()

        sample['HR_name'] = HR_name

        return sample