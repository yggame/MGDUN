# Copyright 2021 Zhongyang Zhang
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import numpy as np
from pathlib2 import Path

from torch.utils.data import DataLoader, DistributedSampler
import torchvision.transforms as transforms

from data.util_fastMRI.volume_sampler import VolumeSampler
from math_fastMRI import *
from data.util_fastMRI import *

def load_model_path(root=None, version=None, v_num=None, best=False):
    """ When best = True, return the best model's path in a directory 
        by selecting the best model with largest epoch. If not, return
        the last model saved. You must provide at least one of the 
        first three args.
    Args: 
        root: The root directory of checkpoints. It can also be a
            model ckpt file. Then the function will return it.
        version: The name of the version you are going to load.
        v_num: The version's number that you are going to load.
        best: Whether return the best model.
    """
    def sort_by_epoch(path):
        name = path.stem
        epoch=int(name.split('-')[1].split('=')[1])
        return epoch
    
    def generate_root():
        if root is not None:
            return root
        elif version is not None:
            return str(Path('lightning_logs', version, 'checkpoints'))
        else:
            return str(Path('lightning_logs', f'version_{v_num}', 'checkpoints'))

    if root==version==v_num==None:
        return None

    root = generate_root()
    if Path(root).is_file():
        return root
    if best:
        files=[i for i in list(Path(root).iterdir()) if i.stem.startswith('best')]
        files.sort(key=sort_by_epoch, reverse=True)
        res = str(files[0])
    else:
        res = str(Path(root) / 'last.ckpt')
    return res

def load_model_path_by_args(args):
    return load_model_path(root=args.load_dir, version=args.load_ver, v_num=args.load_v_num)

def get_number_of_learnable_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])

def print_options(opt, logging):
    """Print and save options

    It will print both current options and default values(if different).
    It will save options into a text file / [checkpoints_dir] / opt.txt
    """
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        # comment = ''
        # default = self.parser.get_default(k)
        # if v != default:
        #     comment = '\t[default: %s]' % str(default)
        # message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '{:>25}: {:<30}\n'.format(str(k), str(v))
    message += '------------------- End -----------------\n'
    logging.info(message)


def create_data_loader(args, dataset_Model, data_transform, data_partition, sample_rate=None):
        sample_rate = sample_rate or args.sample_rate
        dataset = dataset_Model(
            root=args.data_path / f"{args.challenge}_{data_partition}",
            transform=data_transform,
            sample_rate=sample_rate,
            challenge=args.challenge,
            mode=data_partition
        )

        is_train = data_partition == "train"

        # ensure that entire volumes go to the same GPU in the ddp setting
        sampler = None
        if args.use_ddp:
            if is_train:
                sampler = DistributedSampler(dataset)
            else:
                sampler = VolumeSampler(dataset)

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=False,
            drop_last=is_train,
            sampler=sampler,
        )

        return dataloader


class DataTransform(object):
    """
    Data Transformer for training U-Net models.
    """

    def __init__(self, which_challenge, mask_func=None, use_seed=True):
        """
        Args:
            which_challenge (str): Either "singlecoil" or "multicoil" denoting
                the dataset.
            mask_func (fastmri.data.subsample.MaskFunc): A function that can
                create a mask of appropriate shape.
            use_seed (bool): If true, this class computes a pseudo random
                number generator seed from the filename. This ensures that the
                same mask is used for all the slices of a given volume every
                time.
        """
        if which_challenge not in ("singlecoil", "multicoil"):
            raise ValueError(f'Challenge should either be "singlecoil" or "multicoil"')

        self.mask_func = mask_func
        self.which_challenge = which_challenge
        self.use_seed = use_seed

    def __call__(self, kspace, mask, target, attrs, fname, slice_num):
        """
        Args:
            kspace (numpy.array): Input k-space of shape (num_coils, rows,
                cols, 2) for multi-coil data or (rows, cols, 2) for single coil
                data.
            mask (numpy.array): Mask from the test dataset.
            target (numpy.array): Target image.
            attrs (dict): Acquisition related information stored in the HDF5
                object.
            fname (str): File name.
            slice_num (int): Serial number of the slice.

        Returns:
            (tuple): tuple containing:
                image (torch.Tensor): Zero-filled input image.
                target (torch.Tensor): Target image converted to a torch
                    Tensor.
                mean (float): Mean value used for normalization.
                std (float): Standard deviation value used for normalization.
                fname (str): File name.
                slice_num (int): Serial number of the slice.
        """
        kspace = transforms.to_tensor(kspace)


        image = ifft2c(kspace)

        # crop input to correct size
        if target is not None:
            crop_size = (target.shape[-2], target.shape[-1])
        else:
            crop_size = (attrs["recon_size"][0], attrs["recon_size"][1])

        # check for sFLAIR 203
        if image.shape[-2] < crop_size[1]:
            crop_size = (image.shape[-2], image.shape[-2])
   
        image = transforms.complex_center_crop(image, crop_size)

        #getLR
        imgfft = fft2c(image)
        imgfft = transforms.complex_center_crop(imgfft,(160,160))
        LR_image = ifft2c(imgfft)

        # absolute value
        LR_image = complex_abs(LR_image)

        # normalize input
        LR_image, mean, std = transforms.normalize_instance(torch.tensor(LR_image, dtype=torch.float), eps=1e-11)
        LR_image = LR_image.clamp(-6, 6)

        # normalize target
        if target is not None:
            target = transforms.to_tensor(target)
            target = transforms.center_crop(target, crop_size)
            target = transforms.normalize(target, mean, std, eps=1e-11)
            target = target.clamp(-6, 6)
        else:
            target = torch.Tensor([0])

        return LR_image, target, mean, std, fname, slice_num


def train_data_transform(args):
        mask = create_mask_for_mask_type(
            args.mask_type, args.center_fractions, args.accelerations,
        )

        return DataTransform(args.challenge, mask, use_seed=False)

def val_data_transform(args):
    mask = create_mask_for_mask_type(
        args.mask_type, args.center_fractions, args.accelerations,
    )
    return DataTransform(args.challenge, mask)

def test_data_transform(args):
    return DataTransform(args.challenge)