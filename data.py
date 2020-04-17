# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
from torch.utils.data import Dataset, DataLoader
import os
import torch
import numpy as np
import h5py
import random
import os.path as osp
import sys
from six.moves import xrange
import math
import scipy.misc
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class NTUDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = np.array(y, dtype='int')

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return [self.x[index], int(self.y[index])]

class NTUDataLoaders(object):
    def __init__(self, dataset ='NTU', case = 0, aug = 1, seg = 30):
        self.dataset = dataset
        self.case = case
        self.aug = aug
        self.seg = seg
        self.create_datasets()
        self.train_set = NTUDataset(self.train_X, self.train_Y)
        self.val_set = NTUDataset(self.val_X, self.val_Y)
        self.test_set = NTUDataset(self.test_X, self.test_Y)

    def get_train_loader(self, batch_size, num_workers):
        if self.aug == 0:
            return DataLoader(self.train_set, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers,
                              collate_fn=self.collate_fn_fix_val, pin_memory=False, drop_last=True)
        elif self.aug ==1:
            return DataLoader(self.train_set, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers,
                              collate_fn=self.collate_fn_fix_train, pin_memory=True, drop_last=True)

    def get_val_loader(self, batch_size, num_workers):
        if self.dataset == 'NTU' or self.dataset == 'kinetics' or self.dataset == 'NTU120':
            return DataLoader(self.val_set, batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              collate_fn=self.collate_fn_fix_val, pin_memory=True, drop_last=True)
        else:
            return DataLoader(self.val_set, batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              collate_fn=self.collate_fn_fix_val, pin_memory=True, drop_last=True)


    def get_test_loader(self, batch_size, num_workers):
        return DataLoader(self.test_set, batch_size=batch_size,
                          shuffle=False, num_workers=num_workers,
                          collate_fn=self.collate_fn_fix_test, pin_memory=True, drop_last=True)

    def get_train_size(self):
        return len(self.train_Y)

    def get_val_size(self):
        return len(self.val_Y)

    def get_test_size(self):
        return len(self.test_Y)

    def create_datasets(self):
        if self.dataset == 'NTU':
            if self.case ==0:
                self.metric = 'CS'
            elif self.case == 1:
                self.metric = 'CV'
            path = osp.join('./data/ntu', 'NTU_' + self.metric + '.h5')

        f = h5py.File(path , 'r')
        self.train_X = f['x'][:]
        self.train_Y = np.argmax(f['y'][:],-1)
        self.val_X = f['valid_x'][:]
        self.val_Y = np.argmax(f['valid_y'][:], -1)
        self.test_X = f['test_x'][:]
        self.test_Y = np.argmax(f['test_y'][:], -1)
        f.close()

        ## Combine the training data and validation data togehter as ST-GCN
        self.train_X = np.concatenate([self.train_X, self.val_X], axis=0)
        self.train_Y = np.concatenate([self.train_Y, self.val_Y], axis=0)
        self.val_X = self.test_X
        self.val_Y = self.test_Y

    def collate_fn_fix_train(self, batch):
        """Puts each data field into a tensor with outer dimension batch size
        """
        x, y = zip(*batch)

        if self.dataset == 'kinetics' and self.machine == 'philly':
            x = np.array(x)
            x = x.reshape(x.shape[0], x.shape[1],-1)
            x = x.reshape(-1, x.shape[1] * x.shape[2], x.shape[3]*x.shape[4])
            x = list(x)

        x, y = self.Tolist_fix(x, y, train=1)
        lens = np.array([x_.shape[0] for x_ in x], dtype=np.int)
        idx = lens.argsort()[::-1]  # sort sequence by valid length in descending order
        y = np.array(y)[idx]
        x = torch.stack([torch.from_numpy(x[i]) for i in idx], 0)

        if self.dataset == 'NTU':
            if self.case == 0:
                theta = 0.3
            elif self.case == 1:
                theta = 0.5
        elif self.dataset == 'NTU120':
            theta = 0.3

        #### data augmentation
        x = _transform(x, theta)
        #### data augmentation
        y = torch.LongTensor(y)

        return [x, y]

    def collate_fn_fix_val(self, batch):
        """Puts each data field into a tensor with outer dimension batch size
        """
        x, y = zip(*batch)
        x, y = self.Tolist_fix(x, y, train=1)
        idx = range(len(x))
        y = np.array(y)

        x = torch.stack([torch.from_numpy(x[i]) for i in idx], 0)
        y = torch.LongTensor(y)

        return [x, y]

    def collate_fn_fix_test(self, batch):
        """Puts each data field into a tensor with outer dimension batch size
        """
        x, y = zip(*batch)
        x, labels = self.Tolist_fix(x, y ,train=2)
        idx = range(len(x))
        y = np.array(y)


        x = torch.stack([torch.from_numpy(x[i]) for i in idx], 0)
        y = torch.LongTensor(y)

        return [x, y]

    def Tolist_fix(self, joints, y, train = 1):
        seqs = []

        for idx, seq in enumerate(joints):
            zero_row = []
            for i in range(len(seq)):
                if (seq[i, :] == np.zeros((1, 150))).all():
                        zero_row.append(i)

            seq = np.delete(seq, zero_row, axis = 0)

            seq = turn_two_to_one(seq)
            seqs = self.sub_seq(seqs, seq, train=train)

        return seqs, y

    def sub_seq(self, seqs, seq , train = 1):
        group = self.seg

        if self.dataset == 'SYSU' or self.dataset == 'SYSU_same':
            seq = seq[::2, :]

        if seq.shape[0] < self.seg:
            pad = np.zeros((self.seg - seq.shape[0], seq.shape[1])).astype(np.float32)
            seq = np.concatenate([seq, pad], axis=0)

        ave_duration = seq.shape[0] // group

        if train == 1:
            offsets = np.multiply(list(range(group)), ave_duration) + np.random.randint(ave_duration, size=group)
            seq = seq[offsets]
            seqs.append(seq)

        elif train == 2:
            offsets1 = np.multiply(list(range(group)), ave_duration) + np.random.randint(ave_duration, size=group)
            offsets2 = np.multiply(list(range(group)), ave_duration) + np.random.randint(ave_duration, size=group)
            offsets3 = np.multiply(list(range(group)), ave_duration) + np.random.randint(ave_duration, size=group)
            offsets4 = np.multiply(list(range(group)), ave_duration) + np.random.randint(ave_duration, size=group)
            offsets5 = np.multiply(list(range(group)), ave_duration) + np.random.randint(ave_duration, size=group)

            seqs.append(seq[offsets1])
            seqs.append(seq[offsets2])
            seqs.append(seq[offsets3])
            seqs.append(seq[offsets4])
            seqs.append(seq[offsets5])

        return seqs

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def turn_two_to_one(seq):
    new_seq = list()
    for idx, ske in enumerate(seq):
        if (ske[0:75] == np.zeros((1, 75))).all():
            new_seq.append(ske[75:])
        elif (ske[75:] == np.zeros((1, 75))).all():
            new_seq.append(ske[0:75])
        else:
            new_seq.append(ske[0:75])
            new_seq.append(ske[75:])
    return np.array(new_seq)

def _rot(rot):
    cos_r, sin_r = rot.cos(), rot.sin()
    zeros = rot.new(rot.size()[:2] + (1,)).zero_()
    ones = rot.new(rot.size()[:2] + (1,)).fill_(1)

    r1 = torch.stack((ones, zeros, zeros),dim=-1)
    rx2 = torch.stack((zeros, cos_r[:,:,0:1], sin_r[:,:,0:1]), dim = -1)
    rx3 = torch.stack((zeros, -sin_r[:,:,0:1], cos_r[:,:,0:1]), dim = -1)
    rx = torch.cat((r1, rx2, rx3), dim = 2)

    ry1 = torch.stack((cos_r[:,:,1:2], zeros, -sin_r[:,:,1:2]), dim =-1)
    r2 = torch.stack((zeros, ones, zeros),dim=-1)
    ry3 = torch.stack((sin_r[:,:,1:2], zeros, cos_r[:,:,1:2]), dim =-1)
    ry = torch.cat((ry1, r2, ry3), dim = 2)

    rz1 = torch.stack((cos_r[:,:,2:3], sin_r[:,:,2:3], zeros), dim =-1)
    r3 = torch.stack((zeros, zeros, ones),dim=-1)
    rz2 = torch.stack((-sin_r[:,:,2:3], cos_r[:,:,2:3],zeros), dim =-1)
    rz = torch.cat((rz1, rz2, r3), dim = 2)

    rot = rz.matmul(ry).matmul(rx)
    return rot

def _transform(x, theta):
    x = x.contiguous().view(x.size()[:2] + (-1, 3))
    rot = x.new(x.size()[0],3).uniform_(-theta, theta)
    rot = rot.repeat(1, x.size()[1])
    rot = rot.contiguous().view((-1, x.size()[1], 3))
    rot = _rot(rot)
    x = torch.transpose(x, 2, 3)
    x = torch.matmul(rot, x)
    x = torch.transpose(x, 2, 3)

    x = x.contiguous().view(x.size()[:2] + (-1,))
    return x
