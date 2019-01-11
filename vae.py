import torch
import torch.nn as nn
import math

from itertools import chain
import numpy as np
import os
import torch.utils.data
import torch.utils.data.sampler
from deepcluster.util import AverageMeter
from tqdm import tqdm
import matplotlib.pyplot as plt
import ipdb

class VAEModel(nn.Module):
    def __init__(self):
        super(VAEModel, self).__init__()
        self.device = torch.device('cuda:0')

        self.input_size = 2

        self.hidden_size = 256
        self.latent_size = 4

        # encoder
        self.encode_hidden = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU()
        )

        self.mean_z = nn.Linear(self.hidden_size, self.latent_size)
        self.logvar_z = nn.Linear(self.hidden_size, self.latent_size)

        # decoder
        self.decode_hidden = nn.Sequential(
            nn.Linear(self.latent_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
        )

        self.mean_x = nn.Linear(self.hidden_size, self.input_size)
        self.logvar_x = nn.Linear(self.hidden_size, self.input_size)

        self.init_model()

    def init_model(self):
        def weights_init(m):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                nn.init.kaiming_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif classname == 'Linear':
                # nn.init.normal_(m.weight.data, std=0.01)
                # nn.init.eye_(m.weight.data)
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

        self.apply(weights_init)
        return self

    def encode(self, x):
        h = self.encode_hidden(x)
        return self.mean_z(h), self.logvar_z(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul_(std).add_(mu)

    def decode(self, z):
        h = self.decode_hidden(z)
        return self.mean_x(h), self.logvar_x(h)

    def forward(self, x):
        # VAE
        mean_z, logvar_z = self.encode(x)
        z = self.reparameterize(mean_z, logvar_z)
        mean_x, logvar_x = self.decode(z)
        return mean_z, logvar_z, mean_x, logvar_x

        # AE MLP
        # z = self.relu(self.fc1(input))
        # z = self.relu(self.fc21(z))
        # x = self.relu(self.fc3(z))
        # x = self.fc41(x)
        # return x

    def _elbo(self, x, mean_z, logvar_z, mean_x, logvar_x, beta):
        kl_divergence = -0.5 * torch.sum(1 + logvar_z - mean_z.pow(2) - logvar_z.exp(), dim=-1)
        kl_divergence = kl_divergence.mean()

        d = self.input_size
        var = logvar_x.exp()
        likelihood = -0.5 * torch.sum(logvar_x, dim=-1) - 0.5 * torch.sum(torch.pow((x - mean_x), 2) / var, dim=-1)
        likelihood = - 0.5 * d * torch.log(torch.Tensor([2*math.pi])).to(self.device) + likelihood
        likelihood = likelihood.mean()
        return likelihood - beta * kl_divergence


class VAE:

    def __init__(self):
        self.model = VAEModel()
        self.optimizer = torch.optim.SGD(
            filter(lambda x: x.requires_grad, self.model.parameters()),
            lr=1e-3,
            momentum=0.9,
        )

        self.beta = lambda t: min(1, 1e-4 * t)

        self.episode_length = 100
        self.device = torch.device('cuda:0')

        self.loss_function = nn.MSELoss()

    def preprocess_trajectories(self, trajectories):
        if isinstance(trajectories, list) and isinstance(trajectories[0], list):
            trajectories = list(chain(*trajectories))
            trajectories = torch.stack(trajectories, dim=0)
        elif isinstance(trajectories, np.ndarray):
            trajectories = torch.FloatTensor(trajectories)
        else:
            raise ValueError

        assert self.episode_length == trajectories.shape[1]

        return self._train_test_split(trajectories)

    def _train_test_split(self, trajectories):
        num_trajectories_per_skill = 40
        num_skills = trajectories.shape[0] // num_trajectories_per_skill

        indices_train = np.arange(0, 30)
        indices_test = np.arange(30, 40)

        indices_train, indices_test = map(
            lambda x: np.concatenate([x + i * num_trajectories_per_skill for i in range(num_skills)]),
            [indices_train, indices_test])

        trajectories_train, trajectories_test = trajectories[indices_train], trajectories[indices_test]
        return torch.utils.data.TensorDataset(trajectories_train), torch.utils.data.TensorDataset(trajectories_test)

    def fit(self, trajectories):

        dataset_train, dataset_test = self.preprocess_trajectories(trajectories)
        # trajectories = trajectories.reshape([-1, 2])

        loader_train = torch.utils.data.DataLoader(dataset_train, shuffle=True, batch_size=len(dataset_train)//1, num_workers=1)
        loader_test = torch.utils.data.DataLoader(dataset_test, shuffle=False, batch_size=len(dataset_test), num_workers=1)

        self.model.to(self.device)

        t = tqdm(range(20000))
        for i_epoch in t:
            loss_train = self._train(loader_train, i_epoch)

            t.set_description('train loss: {}'.format(loss_train.avg))

            if (i_epoch + 1) % 200 == 0:
                loss_test = self._eval(loader_test, i_epoch)
                # print('epoch: {}\tloss: {}'.format(i_epoch, losses.avg))
                t.write('epoch: {}\ttrain loss: {}\ttest_loss: {}'.format(i_epoch, loss_train.avg, loss_test))

    def _train(self, loader, i_epoch):
        self.model.train()

        losses = AverageMeter()
        for i, (x,) in enumerate(loader):
            x = x.to(self.device)
            self.optimizer.zero_grad()

            # VAE
            mean_z, logvar_z, mean_x, logvar_x = self.model(x)
            loss = -self.model._elbo(x, mean_z, logvar_z, mean_x, logvar_x, self.beta(i_epoch))

            # AE
            # x_recon = self.model(x)
            # loss = self.loss_function(x_recon, x)

            losses.update(loss.item(), x.shape[0])
            loss.backward()
            self.optimizer.step()

        return losses

    def _eval(self, loader, i_epoch):

        self.model.eval()
        loss_function = nn.MSELoss()

        x_orig = []
        x_recon = []

        losses = AverageMeter()
        with torch.no_grad():
            for i, (x,) in enumerate(loader):
                print(i)
                x = x.to(self.device)

                # VAE
                mean_z, logvar_z, mean_x, logvar_x = self.model(x)
                loss = -self.model._elbo(x, mean_z, logvar_z, mean_x, logvar_x, self.beta(i_epoch))
                x_recon.append(mean_x)

                # AE
                # x_hat = self.model(x)
                # loss = loss_function(x_hat, x)
                # x_recon.append(x_hat)

                x_orig.append(x)

                losses.update(loss.item(), x.shape[0])

        x_recon = torch.cat(x_recon, dim=0)
        x_orig = torch.cat(x_orig, dim=0)

        self.model.train()

        def plot(x_orig, x_recon):
            def setup_axes(ax):
                limit = 1
                ax.set_xlim(left=-limit, right=limit)
                ax.set_ylim(bottom=-limit, top=limit)

            fig, axes = plt.subplots(nrows=1, ncols=2, sharex='all', sharey='all',
                                     figsize=[10, 5])
            axes = axes.reshape([-1])

            for i, x_plot in enumerate([x_orig, x_recon]):
                ax = axes[i]
                setup_axes(ax)

                x_plot = x_plot.cpu().numpy().reshape([-1, 2])
                ax.scatter(x_plot[:, 0], x_plot[:, 1], s=1 ** 2)

            plt.savefig(os.path.join(args.log_dir, 'x_test_{}.png'.format(i_epoch)))
            plt.close('all')

        plot(x_orig, x_recon)
        return losses.avg


def visualize(trajectories):
    num_skills = len(trajectories) // 40

    def setup_axes(ax):
        limit = 1
        ax.set_xlim(left=-limit, right=limit)
        ax.set_ylim(bottom=-limit, top=limit)

    fig, axes = plt.subplots(nrows=4, ncols=5, sharex='all', sharey='all', figsize=[10, 8])
    axes = axes.reshape([-1])
    for i in range(num_skills):
        ax = axes[i]
        setup_axes(ax)

        states = trajectories[i*40:(i+1)*40].reshape([-1, 2])
        ax.scatter(states[:, 0], states[:, 1], s=1)
        ax.set_title('skill {}'.format(i))
    plt.savefig('./output/vae/skills_kept.png')
    plt.close('all)')



def validate_vae(args):
    vae = VAE()

    trajectories = np.load('./mixture/data_itr20.pkl')
    trajectories = trajectories / 10
    # i_skill_keep = np.array([1, 3, 9, 10, 11, 13])
    #
    # trajectories = trajectories.reshape([20, 40, 100, 2])
    # trajectories = trajectories[i_skill_keep]
    # trajectories = trajectories.reshape([-1, 100, 2])

    vae.fit(trajectories)
    # visualize(trajectories)


if __name__ == '__main__':
    # filename = '/home/kylehsu/experiments/umrl/output/point2d/20190108/context-all_mog_K50_T50_lambda0.5_ent0.1_N1000/history.pkl'
    # history = pickle.load(open(filename, 'rb'))
    # trajectories = history['trajectories']
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.max_components = 30
    args.episode_length = 100
    args.seed = 1
    # args.log_dir = './output/deepcluster/cluster-kmeans_init-normal_layers5_h4_f2'
    # args.log_dir = './output/vae/debug_ae_mlp_lr1e-3_mo0.9_h64'
    args.log_dir = './output/vae/debug_vae_lr1e-3_mo0.9_h256_l4_beta-linear_skills-all'

    os.makedirs(args.log_dir, exist_ok=True)

    # set seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    validate_vae(args)
