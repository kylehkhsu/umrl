import pickle
from deepcluster.util import AverageMeter

import deepcluster.models.mini_models as dc_models
import faiss
import ipdb
import torch
import torch.nn as nn
from itertools import chain
import numpy as np
import os
import torch.utils.data
import torch.utils.data.sampler
import mixture.models as mix_models

import matplotlib.pyplot as plt


class IdentityLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(IdentityLinear, self).__init__()
        self.linear_positive = nn.Linear(in_features, out_features, bias=bias)
        self.linear_negative = nn.Linear(in_features, out_features, bias=bias)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        positive = self.relu(self.linear_positive(x))
        negative = -self.relu(self.linear_negative(-x))
        return positive + negative


class MLPNet(nn.Module):
    r"""
    A variant of AlexNet.
    The changes with respect to the original AlexNet are:
        - LRN (local response normalization) layers are not included
        - The Fully Connected (FC) layers (fc6 and fc7) have smaller dimensions
          due to the lower resolution of mini-places images (128x128) compared
          with ImageNet images (usually resized to 256x256)
    """
    def __init__(self, num_classes, **kwargs):
        super(MLPNet, self).__init__()
        self.hidden_size = 256
        self.feature_size = 256
        self.num_classes = num_classes

        self.encoder = nn.Sequential(

            # linears
            nn.Linear(in_features=2, out_features=self.hidden_size, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=self.hidden_size, out_features=self.feature_size, bias=True),
            nn.ReLU(inplace=True),

            # for identity init
            # IdentityLinear(2, self.hidden_size, True),
            # IdentityLinear(self.hidden_size, self.hidden_size, True),
            # IdentityLinear(self.hidden_size, self.hidden_size, True),
            # IdentityLinear(self.hidden_size, self.hidden_size, True),
            # IdentityLinear(self.hidden_size, self.feature_size, True),
            # nn.ReLU(inplace=True)
        )

        self.top_layer = nn.Sequential(
            nn.Linear(self.feature_size, num_classes),
        )

        self.init_model()

    def init_model(self):
        def weights_init(m):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                nn.init.kaiming_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif classname == 'Linear':
                # init normal
                # nn.init.normal_(m.weight.data, std=0.01)

                # init orthgonal
                # nn.init.orthogonal_(m.weight.data, gain=1)

                # init

                # init identity + normal
                nn.init.eye_(m.weight.data)
                normal_init = torch.zeros_like(m.weight.data)
                nn.init.normal_(normal_init, std=0.01)
                m.weight.data.add_(normal_init)
                if m.bias is not None:
                    m.bias.data.zero_()

        self.apply(weights_init)
        return self

    def forward(self, input):
        x = self.encoder(input)
        if self.top_layer:
            x = self.top_layer(x)
        return x

    def prepare_for_inference(self):
        self.top_layer = None
        assert type(list(self.encoder.children())[-1]) is nn.ReLU
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-1])
        self.eval()

    def prepare_for_training(self):
        assert self.top_layer is None
        encoder = list(self.encoder.children())
        encoder.append(nn.ReLU(inplace=True).cuda())
        self.encoder = nn.Sequential(*encoder)
        self.top_layer = nn.Linear(in_features=self.feature_size, out_features=self.num_classes)
        self.top_layer.weight.data.normal_(0, 0.01)
        self.top_layer.bias.data.zero_()
        self.top_layer.cuda()
        self.train()


class DeepClusterer:
    def __init__(self, args, clusterer):
        self.args = args
        self.clusterer = clusterer
        self.model = MLPNet(args.num_components)

        self.pca = None

        # optimization
        self.lr = 0.01
        self.momentum = 0.9
        self.weight_decay = 1e-5
        self.num_epochs = 10
        self.batch_size_trajectory = 64
        self.num_workers = 1

        self.model.top_layer = None     # this gets its own optimizer later
        self.model_optimizer = torch.optim.SGD(
            filter(lambda x: x.requires_grad, self.model.parameters()),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay
        )

        # data
        self.num_trajectories = -1
        self.episode_length = args.episode_length
        self.feature_size = self.model.feature_size
        self.visualize_features = False

    def preprocess_trajectories(self, trajectories):
        if isinstance(trajectories, list) and isinstance(trajectories[0], list):
            trajectories = list(chain(*trajectories))
            trajectories = torch.stack(trajectories, dim=0)
        elif isinstance(trajectories, np.ndarray):
            trajectories = torch.FloatTensor(trajectories)
        else:
            raise ValueError

        self.num_trajectories = trajectories.shape[0]
        assert self.episode_length == trajectories.shape[1]

        return trajectories

    def fit(self, trajectories):
        trajectories = self.preprocess_trajectories(trajectories)
        states = trajectories.reshape([-1, trajectories.shape[-1]])

        dataset = torch.utils.data.TensorDataset(trajectories)
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 shuffle=False,
                                                 batch_size=self.batch_size_trajectory,
                                                 num_workers=self.num_workers,
                                                 pin_memory=False)

        for i_epoch in range(self.num_epochs):
            print('epoch {}'.format(i_epoch))
            # infer features, preprocess features
            # if i_epoch != 0:
            features = self.compute_features(dataloader)
            assert features.ndim == 3
            features = np.reshape(features, [-1, features.shape[-1]])
            # else:
            #     features = states
            # features = self.preprocess_features(features)

            # visualize features
            if self.visualize_features:
                self.visualize(features, i_epoch)

            # cluster features
            labels = self.clusterer.fit_predict(features, group=self.episode_length)
            # print('EM lower bound: {}'.format(self.clusterer.lower_bound_))
            print('k-means inertia: {}'.format(self.clusterer.inertia_))

            if i_epoch != self.num_epochs - 1:
                # assign pseudo-labels
                labels = torch.LongTensor(labels)
                train_dataset = torch.utils.data.TensorDataset(states, labels)
                train_dataloader = torch.utils.data.DataLoader(
                    train_dataset,
                    batch_size=self.batch_size_trajectory*self.episode_length,
                    num_workers=self.num_workers,
                    shuffle=True
                )

                # weigh loss according to inverse-frequency of cluster
                weight = torch.zeros(self.args.num_components, dtype=torch.float32)
                unique_labels, counts = np.unique(labels.numpy(), return_counts=True)
                inverse_counts = 1 / counts
                p_label = inverse_counts / np.sum(inverse_counts)
                for i, label in enumerate(unique_labels):
                    weight[label] = p_label[i]

                loss_function = nn.CrossEntropyLoss(weight=weight).cuda()

                # train model
                self.train(train_dataloader, loss_function)

    def compute_features(self, dataloader):
        self.model.cuda()
        self.model.prepare_for_inference()
        features = np.zeros((self.num_trajectories, self.episode_length, self.feature_size), dtype=np.float32)
        with torch.no_grad():
            for i, (input_tensor,) in enumerate(dataloader):
                feature_batch = self.model(input_tensor.cuda()).cpu().numpy()
                if i < len(dataloader) - 1:
                    features[i * self.batch_size_trajectory: (i + 1) * self.batch_size_trajectory] = feature_batch
                else:
                    features[i * self.batch_size_trajectory:] = feature_batch
        self.model.cpu()
        return features

    def train(self, loader, loss_function):
        losses = AverageMeter()

        self.model.prepare_for_training()

        # create optimizer for top layer
        optimizer_tl = torch.optim.SGD(
            self.model.top_layer.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        self.model.cuda()

        for _ in range(10):
            for i, (input_tensor, label) in enumerate(loader):
                output = self.model(input_tensor.cuda())
                loss = loss_function(output, label.cuda())
                losses.update(loss.item(), label.shape[0])

                self.model_optimizer.zero_grad()
                optimizer_tl.zero_grad()
                loss.backward()
                self.model_optimizer.step()
                optimizer_tl.step()

        self.model.cpu()

        print('classification loss: {}'.format(losses.avg))

    def visualize(self, features, iteration, trajectories=None):
        # trajectories = self.preprocess_trajectories(trajectories)
        # states = trajectories.reshape([-1, trajectories.shape[-1]])
        #
        # dataset = torch.utils.data.TensorDataset(trajectories)
        # dataloader = torch.utils.data.DataLoader(dataset,
        #                                          shuffle=False,
        #                                          batch_size=self.batch_size,
        #                                          num_workers=self.num_workers,
        #                                          pin_memory=False)
        #
        # features = self.compute_features(dataloader)
        features = features.reshape([-1, features.shape[-1]])

        #
        #
        #
        # fig, axes = plt.subplots(1, 1, sharex='all', sharey='all', figsize=[5, 5])
        #
        # xs = np.arange(start=-10, stop=10, step=1, dtype=np.float32)
        # ys = np.arange(start=-10, stop=10, step=1, dtype=np.float32)
        # x, y = np.meshgrid(xs, ys)
        #
        # c = np.linspace(0, 1, num=len(xs)*len(ys)).reshape([len(xs), len(ys)])
        # ipdb.set_trace()

        # plt.scatter(states[:, 0], states[:, 1], s=2**2)
        # plt.savefig('./vis/deepcluster_states.png')
        #
        # plt.clf()
        os.makedirs(self.args.log_dir, exist_ok=True)
        plt.scatter(features[:, 0], features[:, 1], s=2**2)
        plt.savefig(os.path.join(self.args.log_dir, 'features_{}.png'.format(iteration)))
        plt.close('all')


if __name__ == '__main__':
    # filename = '/home/kylehsu/experiments/umrl/output/point2d/20190108/context-all_mog_K50_T50_lambda0.5_ent0.1_N1000/history.pkl'
    # history = pickle.load(open(filename, 'rb'))
    # trajectories = history['trajectories']
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.num_components = 20
    args.episode_length = 100
    args.seed = 1
    # args.log_dir = './output/deepcluster/cluster-kmeans_init-normal_layers5_h4_f2'
    args.log_dir = './output/deepcluster/debug'

    # set seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    from mixture.models import BayesianGaussianMixture, GaussianMixture, KMeans
    # clusterer = GaussianMixture(n_components=args.num_components,
    #                             covariance_type='full',
    #                             verbose=1,
    #                             verbose_interval=100,
    #                             max_iter=1000,
    #                             n_init=1)
    clusterer = KMeans(n_clusters=args.num_components,
                       n_init=1,
                       max_iter=300,
                       verbose=0,
                       algorithm='full')

    dc = DeepClusterer(args, clusterer)

    trajectories = np.load('./mixture/data_itr20.pkl')
    trajectories = trajectories / 10
    dc.fit(trajectories)

    # dc.visualize(trajectories)
