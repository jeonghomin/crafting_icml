import os
import numpy as np

import torch
import torch.nn as nn
from typing import List, Optional
import matplotlib.pyplot as plt
from util import *
from utils.builder import build_pipeline
from utils.builder import build_metrics
from utils.builder import build_models
import matplotlib.pyplot as plt
import glob
import cv2
import pdb
class Dataset(torch.utils.data.Dataset):
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
        # Updated at Apr 5 2020
        self.to_tensor = ToTensor()
        

        lst_data = os.listdir(self.data_dir)
        lst_data = [f for f in lst_data if f.endswith('bmp') | f.endswith('jpeg') | f.endswith('png')]
        lst_data.sort()
        self.lst_data = lst_data

    def __len__(self):
        return len(self.lst_data)

    def __getitem__(self, index):
        data = dict()
        HR = cv2.imread(os.path.join(self.data_dir, self.lst_data[index]))
        
        sz = HR.shape
        key = np.random.choice(np.array(self.bin_indices),1, p=self.weight)
        key = [int(item) for item in key[0].strip('[]').split(',')]
        # key = [0,0,0,2]
        # Updated at Apr 5 2020
        data = {'label': HR
                }    
        
        if self.task == "inpainting":
            data['input'] = add_sampling(data['label'], type=self.opts[0], opts=self.opts[1])
        elif self.task == "denoising":
            data['input'] = add_noise(data['label'], type=self.opts[0], opts=self.opts[1])
        if self.transform:
            data = self.transform(data)
        
        data['input'] = data['label']
        if self.task == "super_resolution":
            # data['input'] = add_blur(data['label'], type=self.opts[0], opts=self.opts[1])
       
            # data['input'] = data['input'] * 255
            # degradation part
            for d, b_idx in zip(self.degradations, key):
                # degradation 값 가져오기 1. uniform 2. ctd sampling
                
                data['input'], idx = d(data['input'], b_idx)
             
                if b_idx is not None:
                    assert idx == b_idx
        # print(key)
        for key in data.keys():
            data[key] = data[key].astype(np.float32) / 255.
        
        data = self.to_tensor(data)

        return data

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform):
        self.data_dir = data_dir
        
        # Updated at Apr 5 2020
        self.to_tensor = ToTensor()
        self.transform = transform

        # lst_data = os.listdir(self.data_dir)
        # lst_data = [f for f in lst_data if f.endswith('bmp') | f.endswith('jpeg') | f.endswith('png')]
        # lst_data.sort()
        # self.lst_data = lst_data
        self.HR_list = sorted(glob.glob(os.path.join(self.data_dir,"HR", '*.bmp')))
        self.LR_list = sorted(glob.glob(os.path.join(self.data_dir,"LR", '*.bmp')))

    def __len__(self):
        return len(self.HR_list)

    def __getitem__(self, index):
        HR = plt.imread(self.HR_list[index])
        LR = plt.imread(self.LR_list[index])
        sz = HR.shape

        
        if sz[0] > sz[1]:
            HR = HR.transpose((1, 0, 2))
            LR = LR.transpose((1, 0, 2))

        if HR.ndim == 2:
            HR = HR[:, :, np.newaxis]

        if HR.dtype == np.uint8:
            HR = HR / 255.0
            LR = LR / 255.0

        # Updated at Apr 5 2020
        data = {'label': HR,
                'input': LR
                }
        
        data = self.to_tensor(data)

        return data
class BicubicDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None, task=None, opts=None):
        self.data_dir = data_dir
        self.transform = transform
        self.task = task
        self.opts = opts

        # Updated at Apr 5 2020
        self.to_tensor = ToTensor()

        lst_data = os.listdir(self.data_dir)
        lst_data = [f for f in lst_data if f.endswith('jpg') | f.endswith('jpeg') | f.endswith('png')]

        # lst_label = [f for f in lst_data if f.startswith('label')]
        # lst_input = [f for f in lst_data if f.startswith('input')]
        #
        # lst_label.sort()
        # lst_input.sort()
        #
        # self.lst_label = lst_label
        # self.lst_input = lst_input

        lst_data.sort()
        self.lst_data = lst_data

    def __len__(self):
        return len(self.lst_data)

    def __getitem__(self, index):
        # label = np.load(os.path.join(self.data_dir, self.lst_label[index]))
        # input = np.load(os.path.join(self.data_dir, self.lst_input[index]))

        img = plt.imread(os.path.join(self.data_dir, self.lst_data[index]))
        sz = img.shape

        if sz[0] > sz[1]:
            img = img.transpose((1, 0, 2))

        if img.ndim == 2:
            img = img[:, :, np.newaxis]

        if img.dtype == np.uint8:
            img = img / 255.0

        # label = img
        #
        # if self.task == "inpainting":
        #     input = add_sampling(img, type=self.opts[0], opts=self.opts[1])
        # elif self.task == "denoising":
        #     input = add_noise(img, type=self.opts[0], opts=self.opts[1])
        # elif self.task == "super_resolution":
        #     input = add_blur(img, type=self.opts[0], opts=self.opts[1])
        #
        # data = {'input': input, 'label': label}
        #
        # if self.transform:
        #     data = self.transform(data)

        # Updated at Apr 5 2020
        data = {'label': img}

        if self.task == "inpainting":
            data['input'] = add_sampling(data['label'], type=self.opts[0], opts=self.opts[1])
        elif self.task == "denoising":
            data['input'] = add_noise(data['label'], type=self.opts[0], opts=self.opts[1])

        if self.transform:
            data = self.transform(data)

        if self.task == "super_resolution":
            data['input'] = add_blur(data['label'], type=self.opts[0], opts=self.opts[1])

        data = self.to_tensor(data)

        return data
## 트렌스폼 구현하기
class ToTensor(object):
    def __call__(self, data):
        # label, input = data['label'], data['input']
        #
        # label = label.transpose((2, 0, 1)).astype(np.float32)
        # input = input.transpose((2, 0, 1)).astype(np.float32)
        #
        # data = {'label': torch.from_numpy(label), 'input': torch.from_numpy(input)}

        # Updated at Apr 5 2020
        for key, value in data.items():
            value = value.transpose((2, 0, 1)).astype(np.float32)
            data[key] = torch.from_numpy(value)

        return data

class Normalization(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        # label, input = data['label'], data['input']
        #
        # input = (input - self.mean) / self.std
        # label = (label - self.mean) / self.std
        #
        # data = {'label': label, 'input': input}

        # Updated at Apr 5 2020
        for key, value in data.items():
            data[key] = (value - self.mean) / self.std

        return data


class RandomFlip(object):
    def __call__(self, data):
        # label, input = data['label'], data['input']

        if np.random.rand() > 0.5:
            # label = np.fliplr(label)
            # input = np.fliplr(input)

            # Updated at Apr 5 2020
            for key, value in data.items():
                data[key] = np.flip(value, axis=0)

        if np.random.rand() > 0.5:
            # label = np.flipud(label)
            # input = np.flipud(input)

            # Updated at Apr 5 2020
            for key, value in data.items():
                data[key] = np.flip(value, axis=1)

        # data = {'label': label, 'input': input}

        return data


class RandomCrop(object):
  def __init__(self, shape):
      self.shape = shape

  def __call__(self, data):
    # input, label = data['input'], data['label']
    # h, w = input.shape[:2]

    h, w = data['label'].shape[:2]
    new_h, new_w = self.shape

    top = np.random.randint(0, h - new_h)
    left = np.random.randint(0, w - new_w)

    id_y = np.arange(top, top + new_h, 1)[:, np.newaxis]
    id_x = np.arange(left, left + new_w, 1)

    # input = input[id_y, id_x]
    # label = label[id_y, id_x]
    # data = {'label': label, 'input': input}

    # Updated at Apr 5 2020
    for key, value in data.items():
        data[key] = value[id_y, id_x]

    return data
