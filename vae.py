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
from utils.early_stopping import EarlyStopping
from visualize import add_time, setup_axes
import pickle

class LinearReLUBatchNorm(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearReLUBatchNorm, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.ReLU(),
            nn.BatchNorm1d(output_size),
        )

    def forward(self, x):
        return self.net(x)


class VAEModel(nn.Module):
    def __init__(self, args, input_size):
        super(VAEModel, self).__init__()
        self.device = args.device

        self.input_size = input_size

        self.hidden_size = args.vae_hidden_size
        self.latent_size = args.vae_latent_size

        self.episode_length = args.episode_length

        # encoder
        self.encode_hidden = nn.Sequential(LinearReLUBatchNorm(self.input_size, self.hidden_size))
        for i in range(args.vae_layers - 1):
            self.encode_hidden.add_module('hidden_{}'.format(i+1),
                                          LinearReLUBatchNorm(self.hidden_size, self.hidden_size))

        self.mean_z = nn.Linear(self.hidden_size, self.latent_size)
        self.logvar_z = nn.Linear(self.hidden_size, self.latent_size)

        # decoder
        self.decode_hidden = nn.Sequential(
            LinearReLUBatchNorm(self.latent_size + self.episode_length, self.hidden_size),
        )
        for i in range(args.vae_layers - 1):
            self.decode_hidden.add_module('hidden_{}'.format(i+1),
                                          LinearReLUBatchNorm(self.hidden_size, self.hidden_size))

        self.mean_x = nn.Linear(self.hidden_size, self.input_size)
        self.logvar_x = nn.Linear(self.hidden_size, self.input_size)

        self.log_2_pi = torch.log(torch.Tensor([2*math.pi])).to(self.device)

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
        orig_shape = x.shape[:-1]
        x = x.reshape([-1, self.input_size])

        h = self.encode_hidden(x)

        mean_z, logvar_z = self.mean_z(h), self.logvar_z(h)
        mean_z = mean_z.reshape(*orig_shape, mean_z.shape[-1])
        logvar_z = logvar_z.reshape(*orig_shape, logvar_z.shape[-1])
        return mean_z, logvar_z

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul_(std).add_(mu)

    def decode(self, z):
        orig_shape = z.shape[:-1]
        z = z.reshape([-1, self.episode_length + self.latent_size])

        h = self.decode_hidden(z)

        mean_x, logvar_x = self.mean_x(h), self.logvar_x(h)
        mean_x = mean_x.reshape(*orig_shape, mean_x.shape[-1])
        logvar_x = logvar_x.reshape(*orig_shape, logvar_x.shape[-1])

        logvar_x = logvar_x.clamp(max=5.0)  # can get to inf
        logvar_x = logvar_x.exp().add(0.01).log()     # add some variance to regularize

        return mean_x, logvar_x

    def _get_time(self):
        return torch.eye(self.episode_length).to(self.device)

    def _aggregate_z_param(self, z_param):
        assert z_param.dim() == 3
        assert z_param.shape[1] == self.episode_length
        return z_param.mean(dim=1).unsqueeze(1)

    def _get_reconstruction_latent(self, z):
        batch_shape = z.shape[:-2]
        t = self._get_time()
        t = t.expand(*batch_shape, *t.shape)
        return torch.cat([z, t], dim=-1)

    def forward(self, x):
        # VAE
        batch_size, episode_size, obs_size = x.shape
        mean_z, logvar_z = self.encode(x)

        # one weird trick
        mean_z = self._aggregate_z_param(mean_z)
        logvar_z = self._aggregate_z_param(logvar_z)

        # condition generation on time
        z = self.reparameterize(mean_z, logvar_z)
        z = z.expand([-1, episode_size, -1])  # conditional independence of states given skill
        latent = self._get_reconstruction_latent(z)

        mean_x, logvar_x = self.decode(latent)
        return mean_z, logvar_z, mean_x, logvar_x

    def _log_likelihood(self, x, mean_x, logvar_x):
        assert x.shape == mean_x.shape == logvar_x.shape
        var = logvar_x.exp()
        log_likelihood = -0.5 * torch.sum(logvar_x, dim=-1) - 0.5 * (x - mean_x).pow(2).div(var).sum(dim=-1)
        log_likelihood = -0.5 * self.input_size * self.log_2_pi + log_likelihood
        return log_likelihood

    def _kl_divergence(self, mean_z, logvar_z):
        kl_divergence = -0.5 * torch.sum(1 + logvar_z - mean_z.pow(2) - logvar_z.exp(), dim=-1)
        return kl_divergence

    def _elbo(self, x, mean_z, logvar_z, mean_x, logvar_x, beta):
        elbo = self._log_likelihood(x, mean_x, logvar_x) - beta * self._kl_divergence(mean_z, logvar_z)
        return elbo


class VAE:

    def __init__(self, args, input_size):
        self.args = args
        self.episode_length = args.episode_length
        self.model = VAEModel(args, input_size)
        # self.optimizer = torch.optim.SGD(
        #     filter(lambda x: x.requires_grad, self.model.parameters()),
        #     lr=1e-3,
        #     momentum=0.9,
        # )
        self.optimizer = torch.optim.Adam(
            filter(lambda x: x.requires_grad, self.model.parameters()),
            lr=self.args.vae_lr,
        )

        self.scale = self.args.vae_scale

        # self.beta = lambda t: min(1, args.vae_beta + 1e-3 * t)
        self.beta = lambda t: args.vae_beta
        self.device = args.device
        self.model.to(self.device)

    def load(self, iteration, load_from=None):
        self._set_logging(iteration)
        if load_from:
            filename = load_from
        else:
            filename = self.filename
        if os.path.isfile(filename):
            self.model.load_state_dict(torch.load(filename))
            self.model.eval()
            print('loaded {}'.format(self.filename))
            self.model.to(self.device)
        else:
            raise ValueError

    def _set_logging(self, iteration):
        self.filename = os.path.join(self.args.log_dir, 'ckpt', 'vae_{}.pt'.format(iteration))
        self.plot_dir = os.path.join(self.args.log_dir, 'plt', 'vae_{}'.format(iteration))
        os.makedirs(os.path.join(self.args.log_dir, 'ckpt'), exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)

    def preprocess_trajectories(self, trajectories):
        if isinstance(trajectories, list) and isinstance(trajectories[0], list):
            trajectories = list(chain(*trajectories))
            trajectories = torch.stack(trajectories, dim=0)
        elif isinstance(trajectories, np.ndarray):
            trajectories = torch.FloatTensor(trajectories)
        elif isinstance(trajectories, torch.Tensor):
            pass
        else:
            raise ValueError

        trajectories = self.scale_data(trajectories)

        assert self.episode_length == trajectories.shape[1]

        return self._train_test_split(trajectories)

    def _train_test_split(self, trajectories):
        if trajectories.shape[0] == 800 and trajectories.shape[1] == 100:
            num_trajectories_per_skill = 40
            num_skills = trajectories.shape[0] // num_trajectories_per_skill

            indices_train = np.arange(0, 30)
            indices_test = np.arange(30, 40)

            indices_train, indices_test = map(
                lambda x: np.concatenate([x + i * num_trajectories_per_skill for i in range(num_skills)]),
                [indices_train, indices_test])

            trajectories_train, trajectories_test = trajectories[indices_train], trajectories[indices_test]

        else:
            trajectories_train = trajectories
            trajectories_test = trajectories

        return torch.utils.data.TensorDataset(trajectories_train), torch.utils.data.TensorDataset(trajectories_test)

    def fit(self, trajectories, iteration):
        self._set_logging(iteration)
        self.optimizer = torch.optim.Adam(
            filter(lambda x: x.requires_grad, self.model.parameters()),
            lr=self.args.vae_lr,
        )
        early_stopping = EarlyStopping(mode='min', min_delta=0.005 if self.scale else 0.02, patience=300)

        dataset_train, dataset_test = self.preprocess_trajectories(trajectories)

        loader_train = torch.utils.data.DataLoader(
            dataset_train, shuffle=True, batch_size=self.args.vae_batch_size, num_workers=2)
        loader_test = torch.utils.data.DataLoader(
            dataset_test, shuffle=False, batch_size=self.args.vae_batch_size, num_workers=2)

        if iteration == 0:
            num_max_epoch = 1000
        else:
            num_max_epoch = self.args.vae_max_fit_epoch

        t = tqdm(range(num_max_epoch))
        for i_epoch in t:
            loss_train = self._train(loader_train, i_epoch)

            t.set_description('train loss: {}'.format(loss_train))

            if (i_epoch + 1) % (num_max_epoch // 10) == 0:
                loss_test = self._eval(loader_test, i_epoch)
                # print('epoch: {}\tloss: {}'.format(i_epoch, losses.avg))
                t.write('epoch: {}\ttrain loss: {}\ttest_loss: {}'.format(i_epoch, loss_train, loss_test))

            if i_epoch > 300:
                if early_stopping.step(loss_train):     # doesn't start tracking until epoch 300
                    t.close()
                    break

        torch.save(self.model.state_dict(), self.filename)
        print('wrote vae model to {}'.format(self.filename))

    def _train(self, loader, i_epoch):
        self.model.train()

        losses = AverageMeter()
        for i, (x,) in enumerate(loader):
            x = x.to(self.device)
            self.optimizer.zero_grad()

            # VAE
            mean_z, logvar_z, mean_x, logvar_x = self.model(x)
            elbo = self.model._elbo(x, mean_z, logvar_z, mean_x, logvar_x, self.beta(i_epoch))
            loss = -elbo.mean()

            if (loss != loss).any() or (loss == float('inf')).any() or (loss == -float('inf')).any():
                raise RuntimeError

            losses.update(loss.item(), x.shape[0])
            loss.backward()
            self.optimizer.step()

        return losses.avg

    def _eval(self, loader, i_epoch):

        self.model.eval()

        x_orig = []
        x_recon = []

        losses = AverageMeter()
        with torch.no_grad():
            for i, (x,) in enumerate(loader):
                x = x.to(self.device)

                # VAE
                mean_z, logvar_z, mean_x, logvar_x = self.model(x)
                elbo = self.model._elbo(x, mean_z, logvar_z, mean_x, logvar_x, self.beta(i_epoch))
                loss = -elbo.mean()
                x_recon_ = self.model.reparameterize(mean_x, logvar_x)
                x_recon.append(x_recon_)

                x_orig.append(x)

                losses.update(loss.item(), x.shape[0])

        x_recon = torch.cat(x_recon, dim=0)
        x_orig = torch.cat(x_orig, dim=0)

        z, x_sample, x_sample_mean = self.sample(num_samples=128)

        self.model.train()

        def plot_point():
            fig, axes = plt.subplots(nrows=1, ncols=4, sharex='all', sharey='all',
                                    figsize=[20, 5])
            axes = axes.reshape([-1])

            for i, (x_plot, title) in enumerate(zip([x_orig, x_sample, x_recon, x_sample_mean], ['raw', 'sample', 'reconstruction', 'sample mean'])):
                ax = axes[i]
                setup_axes(ax, limit=10, walls=True)
                ax.set_title(title)

                if self.scale:
                    x_plot = self.unscale_data(x_plot)

                x_plot = x_plot.reshape([-1, self.episode_length, x_plot.shape[-1]])

                x_plot = add_time(x_plot.cpu().numpy())

                x_plot = x_plot.reshape([-1, x_plot.shape[-1]])
                sc = ax.scatter(x_plot[:, 0], x_plot[:, 1], c=x_plot[:, 2], s=1 ** 2)
                plt.colorbar(sc, ax=ax)

        def plot_cheetah():
            fig, axes = plt.subplots(nrows=3, ncols=4,
                                     sharex='row', sharey='row', figsize=[20, 15])
            for i_col, (x_plot, title) in enumerate(zip([x_orig, x_sample, x_recon, x_sample_mean], ['raw', 'sample', 'reconstruction', 'sample mean'])):
                axes[0, i_col].set_title(title)

                if self.scale:
                    x_plot = self.unscale_data(x_plot)
                x_plot = x_plot.reshape([-1, self.episode_length, x_plot.shape[-1]])
                x_plot = add_time(x_plot.cpu().numpy())

                time = x_plot[:, :, -1].reshape([-1, 1])

                i_row = 0
                rootz = x_plot[:, :, 0].reshape([-1, 1])
                bthigh = x_plot[:, :, 2].reshape([-1, 1])
                sc = axes[i_row, i_col].scatter(rootz, bthigh, c=time, s=1**2)
                # plt.colorbar(sc, ax=axes[i_row, i_col])
                axes[i_row, i_col].set_xlabel('rootz [m]')
                axes[i_row, i_col].set_ylabel('bthigh [rad]')

                i_row = 1
                rootx_vel = x_plot[:, :, 8].reshape([-1, 1])
                rootz_vel = x_plot[:, :, 9].reshape([-1, 1])
                sc = axes[i_row, i_col].scatter(rootx_vel, rootz_vel, c=time, s=1**2)
                # plt.colorbar(sc, ax=axes[i_row, i_col])
                axes[i_row, i_col].set_xlabel('rootx [m/s]')
                axes[i_row, i_col].set_ylabel('rootz [m/s]')

                i_row = 2
                rooty_avel = x_plot[:, :, 10].reshape([-1, 1])
                bthigh_avel = x_plot[:, :, 11].reshape([-1, 1])
                sc = axes[i_row, i_col].scatter(rooty_avel, bthigh_avel, c=time, s=1**2)
                # plt.colorbar(sc, ax=axes[i_row, i_col])
                axes[i_row, i_col].set_xlabel('rooty [rad/s]')
                axes[i_row, i_col].set_ylabel('bthigh [rad/s]')

        if self.args.vae_plot:
            if 'Point2D' in self.args.env_name:
                plot_point()
            elif 'HalfCheetah' in self.args.env_name:
                plot_cheetah()
            plt.savefig(os.path.join(self.plot_dir, 'epoch_{}.png'.format(i_epoch)))
            plt.close('all')

        return losses.avg

    def log_s_given_z(self, s, z):
        assert s.dim() == z.dim() == 2
        assert s.shape[0] == z.shape[0]
        assert s.shape[1] == self.model.input_size
        assert z.shape[1] == self.model.latent_size

        if self.scale:
            s = self.scale_data(s)

        self.model.eval()

        with torch.no_grad():
            x = s.to(self.device).reshape(s.shape[0], 1, s.shape[1]).expand([-1, self.episode_length, -1])
            z = z.to(self.device).reshape(z.shape[0], 1, z.shape[1]).expand([-1, self.episode_length, -1])

            z_and_t = self.model._get_reconstruction_latent(z)
            mean_x, logvar_x = self.model.decode(z_and_t)

            log_likelihood = self.model._log_likelihood(x, mean_x, logvar_x)
            assert log_likelihood.dim() == 2
            assert log_likelihood.shape[1] == self.episode_length
            log_likelihood = log_likelihood.exp().mean(dim=1).log()     # marginalize out t
        return log_likelihood.cpu()

    def log_marginal(self, s):
        assert s.shape[1] == self.model.input_size
        assert s.dim() == 2
        self.model.eval()
        # (i_process, i_repetition, i_t, i_feature)
        num_processes = s.shape[0]
        num_z_samples = 16     # samples of z per x; for the expectation under q(z|x)

        if self.scale:
            s = self.scale_data(s)

        with torch.no_grad():
            x = s.to(self.device).reshape(s.shape[0], 1, 1, s.shape[1]).expand([-1, num_z_samples, self.episode_length, -1])
            mean_z, logvar_z = self.model.encode(x)
            z = self.model.reparameterize(mean_z, logvar_z)
            z = z.expand([-1, -1, self.episode_length, -1])
            z_and_t = self.model._get_reconstruction_latent(z)
            mean_x, logvar_x = self.model.decode(z_and_t)

            log_likelihood = self.model._log_likelihood(x, mean_x, logvar_x)
            assert log_likelihood.shape[2] == self.episode_length
            log_likelihood = log_likelihood.exp_().mean(dim=2).log_()

            kl_divergence = self.model._kl_divergence(mean_z, logvar_z)
            kl_divergence = kl_divergence[:, :, 0]   # duplicated across time

            elbo = log_likelihood - kl_divergence
            assert elbo.shape[1] == num_z_samples
            elbo = elbo.mean(dim=1)

        return elbo.cpu()

    def sample(self, num_samples):
        self.model.eval()
        z = torch.randn(num_samples, 1, self.model.latent_size).to(self.device)
        z = z.expand([-1, self.episode_length, -1])

        latent = self.model._get_reconstruction_latent(z)
        with torch.no_grad():
            mean_x, logvar_x = self.model.decode(latent)
            x = self.model.reparameterize(mean_x, logvar_x)

        x = x.reshape([num_samples, self.episode_length, self.model.input_size])
        mean_x = mean_x.reshape([num_samples, self.episode_length, self.model.input_size])
        return z, x, mean_x

    def to(self, device):
        self.model.to(device)

    def scale_data(self, x):
        return x.div(10)

    def unscale_data(self, x):
        return x.mul(10)


def train_vae(args):
    vae = VAE(args)

    # Abhishek's point mass data
    # trajectories = np.load('./mixture/data_itr20.pkl')

    history = pickle.load(open('./output/point2d/20190108/context-all_mog_K50_T50_lambda0.5_ent0.1_N1000_/history.pkl', 'rb'))
    trajectories = history['trajectories']

    trajectories = list(chain(*trajectories))
    subsample_num = 1000
    indices = np.random.choice(len(trajectories), subsample_num, replace=True)
    trajectories = torch.stack([trajectories[index] for index in indices])

    trajectories = trajectories / 10

    # i_skill_keep = np.array([1, 3, 9, 10, 11, 13])
    #
    # trajectories = trajectories.reshape([20, 40, 100, 2])
    # trajectories = trajectories[i_skill_keep]
    # trajectories = trajectories.reshape([-1, 100, 2])

    vae.fit(trajectories)
    # visualize(trajectories)


def validate_vae(args):
    vae = VAE(args)

    # sample trajectories from prior
    z, x, x_mean = vae.sample(num_samples=200)

    def plot():

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=[10, 5])
        axes = axes.reshape([-1])

        for i, x_plot in enumerate([x, x_mean]):
            ax = axes[i]
            setup_axes(ax, limit=1 if args.scale else 10, walls=False if args.scale else True)

            x_plot = add_time(x_plot.cpu().numpy())

            x_plot = x_plot.reshape([-1, x_plot.shape[-1]])
            sc = ax.scatter(x_plot[:, 0], x_plot[:, 1], c=x_plot[:, 2], s=1 ** 2)

            plt.colorbar(sc, ax=ax)

        plt.savefig(os.path.join(args.log_dir, 'samples.png'))
        plt.close('all')

    plot()


    # evaluate

    z = z[:, 0, :]    # align
    t = 25
    s = x_mean[:, t, :]

    vae.model.to(vae.model.device)

    # s_given_z

    s_given_z = vae.s_given_z(s, z)

    s = x[:, t, :]
    s_given_z_2 = vae.s_given_z(s, z)

    # marginal
    p_s_0 = vae.marginal(x[:, 0, :])  # should be high; everything starts at the same place
    p_s_25 = vae.marginal(x[:, 25, :])    # should be lower

    ipdb.set_trace()
    p = 1


if __name__ == '__main__':
    # filename = '/home/kylehsu/experiments/umrl/output/point2d/20190108/context-all_mog_K50_T50_lambda0.5_ent0.1_N1000/history.pkl'
    # history = pickle.load(open(filename, 'rb'))
    # trajectories = history['trajectories']
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.device = torch.device('cuda:0')
    # args.max_components = 30
    args.episode_length = 50
    args.seed = 1
    args.vae_beta = 0.5
    args.vae_lr = 5e-4
    args.vae_hidden_size = 256
    args.vae_latent_size = 8
    args.vae_plot = True
    args.vae_scale = False
    args.log_dir = './output/vae/20190114/mvae_lr5e-4_adam_h256_l8_enc4_dec4_bn_beta-0.5_walls'
    # args.log_dir = './output/vae/20190110/mvae_lr1e-3_adam_h256_l8_beta-1_skills-all_bn'

    os.makedirs(args.log_dir, exist_ok=True)

    # set seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # train_vae(args)
    validate_vae(args)
