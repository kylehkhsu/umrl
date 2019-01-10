import pickle
# import deepcluster
# import deepcluster.vis_utils
# import deepcluster.util
# import deepcluster.dc_main
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
        self.hidden_size = 64
        self.num_classes = num_classes

        self.encoder = nn.Sequential(
            nn.Linear(in_features=2, out_features=self.hidden_size, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size, bias=True),
            nn.ReLU(inplace=True),
        )

        self.top_layer = nn.Sequential(
            nn.Linear(self.hidden_size, num_classes),
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
                nn.init.normal_(m.weight.data, std=0.01)
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
        assert self.top_layer is not None
        self.top_layer = None
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-1])
        self.eval()

    def prepare_for_training(self):
        assert self.top_layer is None
        encoder = list(self.encoder.children())
        encoder.append(nn.ReLU(inplace=True).cuda())
        self.encoder = nn.Sequential(encoder)
        self.top_layer = nn.Linear(in_features=self.hidden_size, out_features=self.num_classes)
        self.top_layer.weight.data.normal_(0, 0.01)
        self.top_layer.bias.data.zero_()
        self.top_layer.cuda()
        self.train()


class DeepClusterer:
    def __init__(self, args, clusterer):
        self.args = args
        self.clusterer = clusterer
        self.model = MLPNet(args.max_components)

        self.pca = None

        # optimization
        self.lr = 0.01
        self.momentum = 0.9
        self.weight_decay = 1e-5
        self.num_epochs = 5
        self.batch_size = 64
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
        self.feature_size = self.model.hidden_size

    def fit(self, trajectories):
        # assert isinstance(trajectories, list)
        # trajectories = list(chain(*trajectories))
        # trajectories = torch.cat(trajectories, dim=0)

        trajectories = torch.FloatTensor(trajectories)
        self.num_trajectories = trajectories.shape[0]
        assert self.episode_length == trajectories.shape[1]
        states = trajectories.reshape([-1, trajectories.shape[-1]])

        dataset = torch.utils.data.TensorDataset(trajectories)
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 shuffle=False,
                                                 batch_size=self.batch_size,
                                                 num_workers=self.num_workers,
                                                 pin_memory=False)

        self.model.cuda()

        for epoch in range(self.num_epochs):
            print('epoch {}'.format(epoch))
            # infer features, preprocess features
            features = self.compute_features(dataloader)
            # features = self.preprocess_features(features)

            # cluster features
            assert features.ndim == 3
            features = np.reshape(features, [-1, features.shape[-1]])
            labels = self.clusterer.fit_predict(features, group=self.episode_length)
            print('EM lower bound: {}'.format(self.clusterer.lower_bound_))

            # assign pseudo-labels
            labels = torch.LongTensor(labels)
            train_dataset = torch.utils.data.TensorDataset(states, labels)
            train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=True
            )

            # weigh loss according to inverse-frequency of cluster
            weight = torch.zeros(self.args.max_components, dtype=torch.float32)
            unique_labels, counts = np.unique(labels.numpy(), return_counts=True)
            inverse_counts = 1 / counts
            p_label = inverse_counts / np.sum(inverse_counts)
            for i, label in enumerate(unique_labels):
                weight[label] = p_label[i]

            loss_function = nn.CrossEntropyLoss(weight=weight).cuda()

            # train model
            self.train(train_dataloader, loss_function)

        self.model.cpu()    # save GPU memory

    def compute_features(self, dataloader):
        self.model.prepare_for_inference()
        features = np.zeros((self.num_trajectories, self.episode_length, self.feature_size), dtype=np.float32)
        with torch.no_grad():
            for i, (input_tensor,) in enumerate(dataloader):
                feature_batch = self.model(input_tensor.cuda()).cpu().numpy()
                if i < len(dataloader) - 1:
                    features[i * self.batch_size: (i + 1) * self.batch_size] = feature_batch
                else:
                    features[i * self.batch_size:] = feature_batch
        return features

    def train(self, loader, loss_function):
        self.model.prepare_for_training()

        # create optimizer for top layer
        optimizer_tl = torch.optim.SGD(
            self.model.top_layer.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        for i, (input_tensor, label) in enumerate(loader):
            output = self.model(input_tensor.cuda()).cpu()
            loss = loss_function(output, label.cuda())

            self.model_optimizer.zero_grad()
            optimizer_tl.zero_grad()
            loss.backward()
            self.model_optimizer.step()
            optimizer_tl.step()


if __name__ == '__main__':
    # filename = '/home/kylehsu/experiments/umrl/output/point2d/20190108/context-all_mog_K50_T50_lambda0.5_ent0.1_N1000/history.pkl'
    # history = pickle.load(open(filename, 'rb'))
    # trajectories = history['trajectories']
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.max_components = 50
    args.episode_length = 100
    args.seed = 1

    # set seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    from mixture.models import BayesianGaussianMixture, GaussianMixture
    clusterer = GaussianMixture(n_components=args.max_components,
                                covariance_type='full',
                                verbose=1,
                                verbose_interval=100,
                                max_iter=1000,
                                n_init=1)

    dc = DeepClusterer(args, clusterer)

    trajectories = np.load('./mixture/data_itr20.pkl')

    dc.fit(trajectories)
