from datetime import date
import os
import numpy as np
import glob
import h5py

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import SimpleITK as sitk
from scipy import ndimage

import random

from sklearn.model_selection import KFold, train_test_split

def normalize_intensity(img_np, normalization="max_min", norm_values=(0, 1, 1, 0)): 
    """
    Accepts an image tensor and normalizes it
    :param normalization: choices = "max", "mean" , type=str
    :param norm_values: (MEAN, STD, MAX, MIN)
    """
    if normalization == "mean":
        mask = img_np[img_np != 0.0]
        desired = img_np[mask]
        mean_val, std_val = desired.mean(), desired.std()
        img_np = (img_np - mean_val) / std_val
    
    elif normalization == "max":
        max_val = norm_values[2]
        img_np = img_np / max_val
    
    elif normalization == "max_min":
        img_np = (img_np - norm_values[3]) / (norm_values[2] - norm_values[3])
    
    elif normalization == "full_volume_mean":
        img_np = (img_np - norm_values[0]) / norm_values[1]
    
    elif normalization == 'candi':
        normalized_np = (img_np- norm_values[0]) / norm_values[1]
        final_np = np.where(img_np == 0., img_np, normalized_np)

        final_np = (final_np - np.min(final_np)) / (np.max(final_np) - np.min(final_np))
        x = np.where(img_np == 0., img_np, final_np)
        return x
    
    else:
        img_np = img_np
    
    return img_np

def random_rot_flip(image1, image2, image3):
    k = np.random.randint(0, 4)
    image1 = np.rot90(image1, k)
    image2 = np.rot90(image2, k)
    image3 = np.rot90(image3, k)
    
    axis = np.random.randint(0, 2)
    image1 = np.flip(image1, axis=axis).copy()
    image2 = np.flip(image2, axis=axis).copy()
    image3 = np.flip(image3, axis=axis).copy()
    return image1, image2, image3


def random_rotate(image1, image2, image3):
    angle = np.random.randint(-20, 20)
    image1 = ndimage.rotate(image1, angle, order=0, reshape=False)
    image2 = ndimage.rotate(image2, angle, order=0, reshape=False)
    image3 = ndimage.rotate(image3, angle, order=0, reshape=False)
    return image1, image2, image3

class RandomGenerator(object):
    def __init__(self):
        # self.output_size = output_size
        pass

    def __call__(self, sample):
        image1, image2, image3 = sample
        if random.random() > 0.5:
            image1, image2, image3 = random_rot_flip(image1, image2, image3)
        elif random.random() > 0.5:
            image1, image2, image3 = random_rotate(image1, image2, image3)

        image1 = torch.from_numpy(image1.astype(np.float32)).unsqueeze(0)
        image2 = torch.from_numpy(image2.astype(np.float32)).unsqueeze(0)
        image3 = torch.from_numpy(image3.astype(np.float32)).unsqueeze(0)

        return image1, image2, image3

class DatasetIXI(Dataset):
    def __init__(self, scale=2, mode='train', sigma=1, transform=None, kfold=0, fold_num=0) -> None:
        super().__init__()
        self.scale = int(scale)
        self.transform = transform
        self.mode = mode
        self.sigma = sigma

        self.data_path = "F:\data\IXI_dataset/"

        if self.mode == 'train':
            self.samples = glob.glob('F:\data\IXI_dataset/IXI-T2-PD/train/*.h5')
        elif self.mode == 'val':
            self.samples = glob.glob('F:\data\IXI_dataset/IXI-T2-PD/val/*.h5')
        else:
            self.samples = glob.glob('F:\data\IXI_dataset/IXI-T2-PD/test/*.h5')

    def __getitem__(self, index):
        data = h5py.File(self.samples[index])
        t2_img = data['T2'][:]
        pd_img = data['PD'][:]

        t2_blur = ndimage.gaussian_filter(t2_img, sigma=self.sigma)
        t2_downsampled = ndimage.zoom(t2_blur, zoom=1.0/self.scale)

        if self.transform:
            D_T2, T2, PD = self.transform((t2_downsampled, t2_img, pd_img))
        else:
            D_T2, T2, PD = t2_downsampled, t2_img, pd_img

        return D_T2, T2, PD
    
    def __len__(self):
        return len(self.samples)