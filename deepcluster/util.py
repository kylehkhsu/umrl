# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import pickle

import numpy as np
import torch
from torch.utils.data.sampler import Sampler

# import models

import argparse

# def load_model(path):
#     """Loads model and return it without DataParallel table."""
#     if os.path.isfile(path):
#         print("=> loading checkpoint '{}'".format(path))
#         checkpoint = torch.load(path)

#         # size of the top layer
#         N = checkpoint['state_dict']['top_layer.bias'].size()

#         # build skeleton of the model
#         sob = 'sobel.0.weight' in checkpoint['state_dict'].keys()
#         model = models.__dict__[checkpoint['arch']](sobel=sob, out=int(N[0]))

#         # deal with a dataparallel table
#         def rename_key(key):
#             if not 'module' in key:
#                 return key
#             return ''.join(key.split('.module'))

#         checkpoint['state_dict'] = {rename_key(key): val
#                                     for key, val
#                                     in checkpoint['state_dict'].items()}

#         # load weights
#         model.load_state_dict(checkpoint['state_dict'])
#         print("Loaded")
#     else:
#         model = None
#         print("=> no checkpoint found at '{}'".format(path))
#     return model

def resume_model(resume, model):
    if os.path.isfile(resume):
        print("=> loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume)

        # remove top_layer parameters from checkpoint
        for key in list(checkpoint['state_dict'].keys()):
            if 'num_batches' in key:
                del checkpoint['state_dict'][key]

        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
                .format(resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(resume))


def get_argparse():
    parser = argparse.ArgumentParser(description='PyTorch Implementation of DeepCluster')

    parser.add_argument('data', metavar='DIR', help='path to dataset')
    parser.add_argument('--arch', '-a', type=str, metavar='ARCH',
                        choices=['alexnet', 'vgg16', 'resnet18'], default='resnet18',
                        help='CNN architecture (default: alexnet)')
    parser.add_argument('--sobel', action='store_true', default=False, help='Sobel filtering')
    parser.add_argument('--clustering', type=str, choices=['GMM', 'BGMM', 'Kmeans', 'PIC'],
                        default='GMM', help='clustering algorithm (default: Kmeans)')
    parser.add_argument('--nmb_cluster', '--k', type=int, default=1000,
                        help='number of cluster for k-means (default: 10000)')
    parser.add_argument('--lr', default=0.05, type=float,
                        help='learning rate (default: 0.05)')
    parser.add_argument('--wd', default=-5, type=float,
                        help='weight decay pow (default: -5)')
    parser.add_argument('--reassign', type=float, default=1.,
                        help="""how many epochs of training between two consecutive
                        reassignments of clusters (default: 1)""")
    parser.add_argument('--workers', default=10, type=int,
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of total epochs to run (default: 200)')
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts) (default: 0)')
    parser.add_argument('--batch', default=256, type=int,
                        help='mini-batch size (default: 256)')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum (default: 0.9)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to checkpoint (default: None)')
    parser.add_argument('--expname', default='', type=str, 
                        help='experiment name')
    parser.add_argument('--checkpoints', type=int, default=25000,
                        help='how many iterations between two checkpoints (default: 25000)')
    parser.add_argument('--seed', type=int, default=31, help='random seed (default: 31)')
    parser.add_argument('--exp', type=str, default='', help='path to exp folder')
    parser.add_argument('--verbose', action='store_true', help='chatty')

    parser.add_argument('--traj_enc', type=str, default='bow', help='encoder for trajectory')
    parser.add_argument('--traj_length', type=int, default=5, help='random seed (default: 31)')
    parser.add_argument('--ep_length', type=int, default=50, help='random seed (default: 31)')
    parser.add_argument('--group', type=int, default=1, help='random seed (default: 31)')

    parser.add_argument('--export', type=int, default=0, help='random seed (default: 31)')
    parser.add_argument('--export-path', type=str, default='/home/ajabri/clones/deepcluster/html/')
    parser.add_argument('--dump-html', type=int, default=0, help='dump html visualization')

    return parser


class UnifLabelSampler(Sampler):
    """Samples elements uniformely accross pseudolabels.
        Args:
            N (int): size of returned iterator.
            images_lists: dict of key (target), value (list of data with this target)
    """

    def __init__(self, N, images_lists):
        self.N = N
        self.images_lists = images_lists
        self.indexes = self.generate_indexes_epoch()

    def generate_indexes_epoch(self):
        nonzero_clusters = [ll for ll in self.images_lists if len(ll) > 0]
        nonzero_idxs = [i for i,_ in enumerate(self.images_lists) if len(_) > 0]

        size_per_pseudolabel = int(self.N / len(nonzero_clusters)) + 1
        res = np.zeros(size_per_pseudolabel * len(nonzero_clusters))

        for i,idx in enumerate(nonzero_idxs):
            indexes = np.random.choice(
                self.images_lists[idx],
                size_per_pseudolabel,
                replace=(len(self.images_lists[idx]) <= size_per_pseudolabel)
            )
            res[i * size_per_pseudolabel: (i + 1) * size_per_pseudolabel] = indexes

        np.random.shuffle(res)
        return res[:self.N].astype('int')

    def __iter__(self):
        return iter(self.indexes)

    def __len__(self):
        return self.N


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


def learning_rate_decay(optimizer, t, lr_0):
    for param_group in optimizer.param_groups:
        lr = lr_0 / np.sqrt(1 + lr_0 * param_group['weight_decay'] * t)
        param_group['lr'] = lr


class Logger():
    """ Class to update every epoch to keep trace of the results
    Methods:
        - log() log and save
    """

    def __init__(self, path):
        self.path = path
        self.data = []

    def log(self, train_point):
        self.data.append(train_point)
        with open(os.path.join(self.path), 'wb') as fp:
            pickle.dump(self.data, fp, -1)
