# TODO
# Try PIC and look at smallest clusters!


# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import os
import pickle
import time
import deepcluster.debug as debug

import faiss
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import TensorDataset
import deepcluster.folder as folder
import deepcluster.clustering as clsutering
import deepcluster.models as models

import deepcluster.util as util
from deepcluster.util import AverageMeter, Logger, UnifLabelSampler
import deepcluster.vis_utils
from itertools import chain



# import deepcluster.export_clusters as export

def main(args):

    # fix random seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # CNN
    if args.verbose:
        print('Architecture: {}'.format(args.arch))
    
    model = models.__dict__[args.arch](sobel=args.sobel, traj_enc=args.traj_enc)
    fd = int(model.top_layer.weight.size()[1])
    
    model.top_layer = None
    model.features = torch.nn.DataParallel(model.features)
    model.cuda()
    cudnn.benchmark = True
    
    # create optimizer
    optimizer = torch.optim.SGD(
        filter(lambda x: x.requires_grad, model.parameters()),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=10**args.wd,
    )

    # define loss function
    criterion = nn.CrossEntropyLoss().cuda()

    # optionally resume from a checkpoint
    if args.resume:
        util.resume_model(resume, model)
    
    # creating checkpoint repo
    exp_check = os.path.join(args.exp, 'checkpoints')
    if not os.path.isdir(exp_check):
        os.makedirs(exp_check)

    # creating cluster assignments log
    cluster_log = Logger(os.path.join(args.exp, 'clusters'))

    smoother = models.mini_models.GaussianSmoothing(3, 5, 1)

    # load the data
    end = time.time()

    tra, (mean, std), (m1, std1), (norm, unnorm) = vis_utils.make_transform(args.data)

    if hasattr(args, 'pretransform'):
        tra = args.pretransform + tra

    dataset = folder.ImageFolder(args.data, transform=transforms.Compose(tra),
        args=[args.ep_length, args.traj_length], samples=None if not hasattr(args, 'samples') else args.samples)

    if args.verbose: print('Load dataset: {0:.2f} s'.format(time.time() - end))

    # import bow_dataset
    dataloader = torch.utils.data.DataLoader(dataset,
                                            #  batch_sampler=sampler,
                                             shuffle=False,
                                             batch_size=args.batch,
                                             num_workers=args.workers,
                                             pin_memory=True)

    # clustering algorithm to use
    if args.group > 1:
        args.group = args.ep_length - args.traj_length + 1
    
    deepcluster = clustering.__dict__[args.clustering](args.nmb_cluster, group=args.group)

    # training convnet with DeepCluster
    for epoch in range(args.start_epoch, args.epochs):
        end = time.time()

        # remove head
        model.top_layer = None
        model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])

        # get the features for the whole dataset
        print('computing features')
        features, idxs, poses = compute_features(dataloader, model, len(dataset), args)

        # idxs = idxs[np.argsort(idxs)]
        assert(all(np.argsort(np.argsort(idxs, kind='stable')) == np.argsort(idxs, kind='stable'))) 
        # features = features[np.argsort(idxs)]

        # cluster the features
        # print('clustering')
        print('clustering')
        clustering_loss = deepcluster.cluster(features, verbose=args.verbose)
        print('cluster loss', clustering_loss)

        # assign pseudo-labels
        print('assign clustering')
        train_dataset = clustering.cluster_assign(
            deepcluster.images_lists,
            dataset.imgs, transforms.Compose(tra))

        # uniformely sample per target
        sampler = UnifLabelSampler(int(args.reassign * len(train_dataset)), deepcluster.images_lists)
                                #    [ll for ll in deepcluster.images_lists if len(ll) > 0])

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=int(args.batch / 4),
            num_workers=args.workers,
            sampler=sampler,
            pin_memory=True,
        )

        # set last fully connected layer
        mlp = list(model.classifier.children())
        mlp.append(nn.ReLU(inplace=True).cuda())
        model.classifier = nn.Sequential(*mlp)
        model.top_layer = nn.Linear(fd, len(deepcluster.images_lists))
        model.top_layer.weight.data.normal_(0, 0.01)
        model.top_layer.bias.data.zero_()
        model.top_layer.cuda()

        # train network with clusters as pseudo-labels
        end = time.time()
        loss = train(train_dataloader, model, criterion, optimizer, epoch, args)

        # print log
        if args.verbose:
            print('###### Epoch [{0}] ###### \n'
                  'Time: {1:.3f} s\n'
                  'Clustering loss: {2:.3f} \n'
                  'ConvNet loss: {3:.3f}'
                  .format(epoch, time.time() - end, clustering_loss, loss))
            try:
                nmi = normalized_mutual_info_score(
                    clustering.arrange_clustering(deepcluster.images_lists),
                    clustering.arrange_clustering(cluster_log.data[-1])
                )
                print('NMI against previous assignment: {0:.3f}'.format(nmi))
            except IndexError:
                pass
            print('####################### \n')
        # save running checkpoint
        torch.save({'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict()},
                   os.path.join(args.exp, 'checkpoint.pth.tar'))

        # save cluster assignments
        cluster_log.log(deepcluster.images_lists)

    # if args.export > 0:
    #     clus = export.export(args, model, dataloader, dataset)
    #     return model, args, clus
    
    return model, args, None

def train(loader, model, crit, opt, epoch, args):
    """Training of the CNN.
        Args:
            loader (torch.utils.data.DataLoader): Data loader
            model (nn.Module): CNN
            crit (torch.nn): loss
            opt (torch.optim.SGD): optimizer for every parameters with True
                                   requires_grad in model except top layer
            epoch (int)
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    data_time = AverageMeter()
    forward_time = AverageMeter()
    backward_time = AverageMeter()

    # switch to train mode
    model.train()

    # create an optimizer for the last fc layer
    optimizer_tl = torch.optim.SGD(
        model.top_layer.parameters(),
        lr=args.lr,
        weight_decay=10**args.wd,
    )


    end = time.time()

    for _ in range(1):
        for i, (input_tensor, target) in enumerate(loader):
            data_time.update(time.time() - end)

            # import pdb; pdb.set_trace() 
            # save checkpoint
            n = len(loader) * epoch + i
            if n % args.checkpoints == 0:
                path = os.path.join(
                    args.exp,
                    'checkpoints',
                    'checkpoint_' + str(n / args.checkpoints) + '.pth.tar',
                )
                if args.verbose:
                    print('Save checkpoint at: {0}'.format(path))
                torch.save({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'optimizer' : opt.state_dict()
                }, path)

            target = target.cuda(async=True)
            input_var = torch.autograd.Variable(input_tensor.cuda())
            target_var = torch.autograd.Variable(target)

            output = model(input_var)
            loss = crit(output, target_var)

            # record loss
            losses.update(loss.data[0], input_tensor.size(0))

            # compute gradient and do SGD step
            opt.zero_grad()
            optimizer_tl.zero_grad()
            loss.backward()
            opt.step()
            optimizer_tl.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if args.verbose and (i % 50) == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data: {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss: {loss.val:.4f} ({loss.avg:.4f})'
                    .format(epoch, i, len(loader), batch_time=batch_time,
                            data_time=data_time, loss=losses))

    return losses.avg

def compute_features(dataloader, model, N, args):
    if args.verbose:
        print('Compute features')
    batch_time = AverageMeter()
    end = time.time()
    model.eval()
    # discard the label information in the dataloader
    for i, (input_tensor, idx, pose) in enumerate(dataloader):
        # print(i)
        with torch.no_grad():
            input_var = torch.autograd.Variable(input_tensor.cuda())

        aux = model(input_var).data.cpu().numpy()
        idx = idx.data.cpu().numpy()
        pose = pose.data.cpu().numpy()

        if i == 0:
            features = np.zeros((N, aux.shape[1])).astype('float32')
            poses = np.zeros((N, pose.shape[1])).astype('float32')
            idxs = np.zeros((N)).astype(np.int)

        if i < len(dataloader) - 1:
            features[i * args.batch: (i + 1) * args.batch] = aux.astype('float32')
            poses[i * args.batch: (i + 1) * args.batch] = pose.astype(np.int)            
            idxs[i * args.batch: (i + 1) * args.batch] = idx.astype(np.int)
        else:
            # special treatment for final batch
            features[i * args.batch:] = aux.astype('float32')
            poses[i * args.batch:] = pose.astype(np.int)            
            idxs[i * args.batch:] = idx.astype(np.int)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.verbose and (i % 50) == 0:
            print('{0} / {1}\t'
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})'
                  .format(i, len(dataloader), batch_time=batch_time))
    return features, idxs, poses


if __name__ == '__main__':
    parser = util.get_argparse()
    args = parser.parse_args()
    main(args)
