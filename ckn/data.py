# -*- coding: utf-8 -*-
import os
import scipy.io as sio

import numpy as np

import torch
import torch.utils.data as data
import torchvision.transforms as transforms


class Rescale(object):
    def __init__(self):
        self.xmax = None
        self.xmin = None

    def __call__(self, pic):
        if self.xmax is None:
            self.xmax = pic.max()
            self.xmin = pic.min()
            pic = 255 * (pic - self.xmin) / (self.xmax - self.xmin)
            return pic.astype('uint8')
        return self.xmin + pic * (self.xmax - self.xmin)

def create_dataset(root, train=True, dataugmentation=False):
    # load dataset
    if not '.mat' in root:
        mean_pix = [x/255.0 for x in [125.3, 123.0, 113.9]]
        std_pix = [x/255.0 for x in [63.0, 62.1, 66.7]]
        tr = [transforms.ToTensor(), transforms.Normalize(mean=mean_pix, std=std_pix)]
        if dataugmentation:
            dt = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
            tr = dt + tr
        dataset = torchvision.datasets.CIFAR10(
            root,
            train=train,
            transform=transforms.Compose(tr),
            download=True,
        )
        return dataset
    else:
        tr = [transforms.ToTensor()]
        if dataugmentation:
            dt = [transforms.ToPILImage(), transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
            tr = dt + tr
        dataset = CIFARmatlab(
            root,
            train=train,
            transform=transforms.Compose(tr),
            augment=dataugmentation
        )
        return dataset 


class CIFARmatlab(data.Dataset):
    def __init__(self, root, train=True, transform=None, augment=False, dtype='float32'):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.train = train # training set or test set
        if self.train:
            split = 'tr'
        else:
            split = 'te'
        matdata = sio.loadmat(root)
        R = matdata['X' + split][:, :32, :].transpose(2, 1, 0)
        G = matdata['X' + split][:, 32: 64, :].transpose(2, 1, 0)
        B = matdata['X' + split][:, 64:, :].transpose(2, 1, 0)
        data = np.stack([R, G, B], axis=3)
        labels = [e[0] for e in matdata['Y' + split]]
        data = data.astype(dtype)
        labels = labels
        if self.train:
            self.train_data = data
            self.train_labels = labels
        else:
            self.test_data = data
            self.test_labels = labels
        self.augment = augment
 
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        if self.transform is not None:
            if self.augment:
                rs = Rescale()
                img = rs(img)
            img = self.transform(img)
            if self.augment:
                img = rs(img)
                del rs
        target = torch.tensor(target, dtype=torch.long)
        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)
