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
import copy

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

        self.mean_h = nn.Linear(self.hidden_size, self.latent_size)
        self.weight_h = nn.Sequential(
            nn.Linear(self.hidden_size, self.latent_size),
            nn.ReLU()
        )

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

        mean_h, weight_h = self.mean_h(h), self.weight_h(h) + 1e-5
        mean_h = mean_h.reshape(*orig_shape, mean_h.shape[-1])
        weight_h = weight_h.reshape(*orig_shape, weight_h.shape[-1])
        return mean_h, weight_h

    def reparameterize(self, mu, logvar=None, var=None):
        if logvar is not None:
            std = torch.exp(0.5 * logvar)
        elif var is not None:
            std = torch.sqrt(var)
        else:
            raise RuntimeError
        eps = torch.randn_like(std)
        return eps.mul_(std).add_(mu)

    def decode(self, z):
        orig_shape = z.shape[:-1]
        z = z.reshape([-1, self.episode_length + self.latent_size])

        h = self.decode_hidden(z)

        mean_x, logvar_x = self.mean_x(h), self.logvar_x(h)
        mean_x = mean_x.reshape(*orig_shape, mean_x.shape[-1])
        logvar_x = logvar_x.reshape(*orig_shape, logvar_x.shape[-1])

        logvar_x = logvar_x.clamp(max=4.0)  # can get to inf
        logvar_x = logvar_x.exp().add(0.01).log()     # add some variance to regularize

        return mean_x, logvar_x

    def _get_time(self):
        return torch.eye(self.episode_length).to(self.device)

    def _compute_posterior(self, mean_h, weight_h):
        assert mean_h.shape[-2] == weight_h.shape[-2] <= self.episode_length
        weight_h_sum = weight_h.sum(dim=-2, keepdim=True)
        var_z = weight_h_sum.reciprocal()
        mean_z = (mean_h * weight_h).sum(dim=-2, keepdim=True) / weight_h_sum
        return mean_z, var_z

    def _get_reconstruction_latent(self, z):
        batch_shape = z.shape[:-2]
        t = self._get_time()
        t = t.expand(*batch_shape, *t.shape)
        return torch.cat([z, t], dim=-1)

    def forward(self, x):
        # VAE
        batch_shape = x.shape[:-2]
        mean_h, weight_h = self.encode(x)

        # one weird trick
        mean_z, var_z = self._compute_posterior(mean_h, weight_h)
        mean_z = mean_z.expand(*batch_shape, self.episode_length, -1)
        var_z = var_z.expand(*batch_shape, self.episode_length, -1)

        # condition generation on time
        z = self.reparameterize(mean_z, var=var_z)
        latent = self._get_reconstruction_latent(z)

        mean_x, logvar_x = self.decode(latent)
        return mean_z, var_z, mean_x, logvar_x

    def _log_likelihood(self, x, mean_x, logvar_x):
        assert x.shape == mean_x.shape == logvar_x.shape
        var = logvar_x.exp()
        log_likelihood = -0.5 * torch.sum(logvar_x, dim=-1) - 0.5 * (x - mean_x).pow(2).div(var).sum(dim=-1)
        log_likelihood = -0.5 * self.input_size * self.log_2_pi + log_likelihood
        return log_likelihood

    def _kl_divergence(self, mean_z, var_z):
        kl_divergence = -0.5 * torch.sum(1 + var_z.log() - mean_z.pow(2) - var_z, dim=-1)
        return kl_divergence

    def _elbo(self, x, mean_z, var_z, mean_x, logvar_x, beta):
        elbo = self._log_likelihood(x, mean_x, logvar_x) - beta * self._kl_divergence(mean_z, var_z)
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

        self.normalize_strategy = self.args.vae_normalize_strategy

        # self.beta = lambda t: min(1, args.vae_beta + 1e-3 * t)
        self.beta = lambda t: args.vae_beta
        self.device = args.device
        self.model.to(self.device)
        self.mean = None
        self.std = None

    def load(self, iteration, load_from=''):
        self._set_logging(iteration)
        if load_from != '':
            filename = load_from
        else:
            filename = self.filename
        if os.path.isfile(filename):
            vae_dict = torch.load(filename)
            self.model = vae_dict['model']
            self.mean = vae_dict['mean']
            self.std = vae_dict['std']
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

        self._calculate_statistics(trajectories)
        trajectories = self.normalize_data(trajectories)

        assert self.episode_length == trajectories.shape[1]

        return self._train_test_split(trajectories)

    def _train_test_split(self, trajectories):
        if trajectories.shape[0] == 800 and trajectories.shape[1] == 100:   # for testing only
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
        # self.optimizer = torch.optim.Adam(
        #     filter(lambda x: x.requires_grad, self.model.parameters()),
        #     lr=self.args.vae_lr,
        # )

        dataset_train, dataset_test = self.preprocess_trajectories(trajectories)

        num_trajectories = len(dataset_train)
        batch_size = num_trajectories // self.args.vae_batches

        loader_train = torch.utils.data.DataLoader(
            dataset_train, shuffle=True, batch_size=batch_size, num_workers=2)
        loader_test = torch.utils.data.DataLoader(
            dataset_test, shuffle=False, batch_size=batch_size, num_workers=2)

        num_max_epoch = self.args.vae_max_fit_epoch

        if 'Point2D' in self.args.env_name:
            min_delta = 0.05
        else:
            min_delta = 0.005   # TODO: tune

        early_stopping = EarlyStopping(mode='min', min_delta=min_delta, patience=num_max_epoch // 10)

        t = tqdm(range(num_max_epoch))
        for i_epoch in t:
            loss_train = self._train(loader_train, i_epoch)

            t.set_description('train loss: {}'.format(loss_train))

            if i_epoch == 0 or (i_epoch + 1) % (num_max_epoch // 5) == 0:
                loss_test = self._eval(loader_test, i_epoch)
                # print('epoch: {}\tloss: {}'.format(i_epoch, losses.avg))
                t.write('epoch: {}\ttrain loss: {}\ttest_loss: {}'.format(i_epoch, loss_train, loss_test))

            if i_epoch > num_max_epoch // 5:
                if early_stopping.step(loss_train):     # doesn't start tracking until epoch 300
                    t.close()
                    break

        model = copy.deepcopy(self.model).cpu()
        mean = self.mean.clone()
        std = self.std.clone()
        torch.save(dict(model=model, mean=mean, std=std), self.filename)
        print('wrote vae model to {}'.format(self.filename))

    def _train(self, loader, i_epoch):
        self.model.train()

        losses = AverageMeter()
        for i, (x,) in enumerate(loader):
            x = x.to(self.device)
            self.optimizer.zero_grad()

            # VAE
            mean_z, var_z, mean_x, logvar_x = self.model(x)
            elbo = self.model._elbo(x, mean_z, var_z, mean_x, logvar_x, self.beta(i_epoch))
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
                mean_z, var_z, mean_x, logvar_x = self.model(x)
                elbo = self.model._elbo(x, mean_z, var_z, mean_x, logvar_x, self.beta(i_epoch))
                loss = -elbo.mean()
                x_recon_ = self.model.reparameterize(mean_x, logvar=logvar_x)
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

                x_plot = self.unnormalize_data(x_plot)

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

                x_plot = self.unnormalize_data(x_plot)
                x_plot = x_plot.reshape([-1, self.episode_length, x_plot.shape[-1]])
                x_plot = add_time(x_plot.cpu().numpy())

                rootz = x_plot[:, :, 0].reshape([-1, 1])
                rootx_vel = x_plot[:, :, 8].reshape([-1, 1])
                rootz_vel = x_plot[:, :, 9].reshape([-1, 1])
                time = x_plot[:, :, -1].reshape([-1, 1])

                i_row = 0
                sc = axes[i_row, i_col].scatter(rootz, rootx_vel, c=time, s=1**2)
                axes[i_row, i_col].set_xlabel('rootz [m]')
                axes[i_row, i_col].set_ylabel('rootx_vel [m/s]')

                i_row = 1
                sc = axes[i_row, i_col].scatter(rootx_vel, rootz_vel, c=time, s=1**2)
                axes[i_row, i_col].set_xlabel('rootx_vel [m/s]')
                axes[i_row, i_col].set_ylabel('rootz_vel [m/s]')

                i_row = 2
                sc = axes[i_row, i_col].scatter(rootz, rootz_vel, c=time, s=1**2)
                axes[i_row, i_col].set_xlabel('rootz [m]')
                axes[i_row, i_col].set_ylabel('rootz_vel [m/s]')

        def plot_ant():
            fig, axes = plt.subplots(nrows=3, ncols=4,
                                     sharex='row', sharey='row', figsize=[20, 15])
            for i_col, (x_plot, title) in enumerate(zip([x_orig, x_sample, x_recon, x_sample_mean], ['raw', 'sample', 'reconstruction', 'sample mean'])):
                axes[0, i_col].set_title(title)

                x_plot = self.unnormalize_data(x_plot)
                x_plot = x_plot.reshape([-1, self.episode_length, x_plot.shape[-1]])
                x_plot = add_time(x_plot.cpu().numpy())

                qpos_0 = x_plot[:, :, 0].reshape([-1, 1])
                qpos_1 = x_plot[:, :, 1].reshape([-1, 1])
                qpos_2 = x_plot[:, :, 2].reshape([-1, 1])
                time = x_plot[:, :, -1].reshape([-1, 1])

                i_row = 0
                sc = axes[i_row, i_col].scatter(qpos_0, qpos_1, c=time, s=1**2)
                axes[i_row, i_col].set_xlabel('qpos_0 [m]')
                axes[i_row, i_col].set_ylabel('qpos_1 [m]')

                i_row = 1
                sc = axes[i_row, i_col].scatter(qpos_0, qpos_2, c=time, s=1**2)
                axes[i_row, i_col].set_xlabel('qpos_0 [m]')
                axes[i_row, i_col].set_ylabel('qpos_2 [m]')

                i_row = 2
                sc = axes[i_row, i_col].scatter(qpos_1, qpos_2, c=time, s=1**2)
                axes[i_row, i_col].set_xlabel('qpos_1 [m]')
                axes[i_row, i_col].set_ylabel('qpos_2 [m]')

        if self.args.vae_plot:
            if 'Point2D' in self.args.env_name:
                plot_point()
            elif 'HalfCheetah' in self.args.env_name:
                plot_cheetah()
            elif 'Ant' in self.args.env_name:
                plot_ant()
            plt.savefig(os.path.join(self.plot_dir, 'epoch_{}.png'.format(i_epoch)))
            plt.close('all')

        return losses.avg

    def log_s_given_z(self, s, z):
        assert s.dim() == z.dim() == 2
        assert s.shape[0] == z.shape[0]
        assert s.shape[1] == self.model.input_size
        assert z.shape[1] == self.model.latent_size

        s = self.normalize_data(s)

        self.model.eval()

        with torch.no_grad():
            x = s.to(self.device).reshape(s.shape[0], 1, s.shape[1]).expand([-1, self.episode_length, -1])
            z = z.to(self.device).reshape(z.shape[0], 1, z.shape[1]).expand([-1, self.episode_length, -1])

            z_and_t = self.model._get_reconstruction_latent(z)
            mean_x, logvar_x = self.model.decode(z_and_t)

            log_likelihood = self.model._log_likelihood(x, mean_x, logvar_x)
            assert log_likelihood.dim() == 2
            assert log_likelihood.shape[1] == self.episode_length
            log_likelihood = log_likelihood.exp().clamp(min=0, max=1).mean(dim=1).log()     # marginalize out t
        return log_likelihood.cpu()

    def log_marginal(self, s, traj):
        assert s.shape[1] == self.model.input_size
        assert s.dim() == 2
        assert traj.shape[0] == s.shape[0] == self.args.num_processes
        assert traj.shape[2] == self.model.input_size
        assert traj.dim() == 3

        self.model.eval()
        # (i_process, i_repetition, i_t, i_feature)
        num_z_samples = self.args.vae_marginal_samples     # samples of z per x; for the expectation under q(z|x)

        s = self.normalize_data(s)
        traj = self.normalize_data(traj)

        with torch.no_grad():
            traj = traj.to(self.device).reshape(traj.shape[0], 1, traj.shape[1], traj.shape[2])
            traj = traj.expand([-1, num_z_samples, -1, -1])
            mean_z, var_z, mean_x, logvar_x = self.model(traj)

            s = s.to(self.device).reshape(s.shape[0], 1, 1, s.shape[1])
            s = s.expand([-1, num_z_samples, self.episode_length, -1])

            log_likelihood = self.model._log_likelihood(s, mean_x, logvar_x)
            assert log_likelihood.shape[2] == self.episode_length
            log_likelihood = log_likelihood.exp_().clamp(min=0, max=1).mean(dim=2).log_()

            kl_divergence = self.model._kl_divergence(mean_z, var_z)
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
            x = self.model.reparameterize(mean_x, logvar=logvar_x)

        x = x.reshape([num_samples, self.episode_length, self.model.input_size])
        mean_x = mean_x.reshape([num_samples, self.episode_length, self.model.input_size])
        return z, x, mean_x

    def to(self, device):
        self.model.to(device)

    def _calculate_statistics(self, traj):
        if self.args.vae_normalize_strategy == 'none':
            self.mean = torch.zeros(traj.shape[-1])
            self.std = torch.ones(traj.shape[-1])
        elif self.args.vae_normalize_strategy == 'first':
            if self.mean is None and self.std is None:
                self.mean = traj.mean(dim=0).mean(dim=0)
                self.std = traj.sub(self.mean).pow(2).sum(dim=0).sum(dim=0).div(traj.shape[0] * traj.shape[1] - 1).sqrt()
        elif self.args.vae_normalize_strategy == 'adaptive':
            self.mean = traj.mean(dim=0).mean(dim=0)
            self.std = traj.sub(self.mean).pow(2).sum(dim=0).sum(dim=0).div(traj.shape[0] * traj.shape[1] - 1).sqrt()
        elif self.args.vae_normalize_strategy == 'fixed':
            if 'Point2D' in self.args.env_name:
                self.mean = torch.zeros(traj.shape[-1])
                self.std = 10 * torch.ones(traj.shape[-1])
            else:
                raise ValueError
        else:
            raise ValueError

    def normalize_data(self, x):
        # if 'Point2D' in self.args.env_name:
        #     x = x.div(10)
        # else:
        x = x.cpu().sub(self.mean).div(self.std + 1e-6)
        return x

    def unnormalize_data(self, x):
        # if 'Point2D' in self.args.env_name:
        #     x = x.mul(10)
        # else:
        x = x.cpu().mul(self.std + 1e-6).add(self.mean)
        return x


def train_vae(args):
    vae = VAE(args, input_size=2)

    # Abhishek's point mass data
    # trajectories = np.load('./mixture/data_itr20.pkl')

    history = pickle.load(open('./output/point2d/20190108/context-all_mog_K50_T50_lambda0.5_ent0.1_N1000_/history.pkl', 'rb'))
    trajectories = history['trajectories']

    trajectories = list(chain(*trajectories))
    subsample_num = 1000
    indices = np.random.choice(len(trajectories), subsample_num, replace=True)
    trajectories = torch.stack([trajectories[index] for index in indices])

    # i_skill_keep = np.array([1, 3, 9, 10, 11, 13])
    #
    # trajectories = trajectories.reshape([20, 40, 100, 2])
    # trajectories = trajectories[i_skill_keep]
    # trajectories = trajectories.reshape([-1, 100, 2])

    vae.fit(trajectories, iteration=1)


def validate_vae(args):
    vae = VAE(args, input_size=2)
    vae.load(iteration=1)

    history = pickle.load(open('./output/point2d/20190108/context-all_mog_K50_T50_lambda0.5_ent0.1_N1000_/history.pkl', 'rb'))
    trajectories = history['trajectories']

    trajectories = list(chain(*trajectories))
    subsample_num = 1000
    indices = np.random.choice(len(trajectories), subsample_num, replace=True)
    trajectories = torch.stack([trajectories[index] for index in indices])

    args.num_processes = 5

    traj = trajectories[-5:]
    traj_part = traj[:, :25, :]
    state = traj_part[:, -1, :]

    for i in range(10):
        print(vae.log_marginal(state, traj_part).exp())

    z, x, mean_x = vae.sample(5)
    x = vae.unnormalize_data(x)
    mean_x = vae.unnormalize_data(mean_x)
    z = z[:, 0, :]
    print(vae.log_s_given_z(x[:, -1, :], z).exp())


if __name__ == '__main__':
    # filename = '/home/kylehsu/experiments/umrl/output/point2d/20190108/context-all_mog_K50_T50_lambda0.5_ent0.1_N1000/history.pkl'
    # history = pickle.load(open(filename, 'rb'))
    # trajectories = history['trajectories']
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.device = 'cuda:0'
    # args.num_components = 30
    args.episode_length = 50
    args.seed = 1
    args.vae_beta = 0.6
    args.vae_lr = 5e-4
    args.vae_hidden_size = 256
    args.vae_latent_size = 8
    args.vae_layers = 4
    args.vae_plot = True
    args.vae_normalize = True
    args.vae_max_fit_epoch = 1000
    args.vae_weights = None
    args.vae_batches = 16
    args.vae_marginal_samples = 16
    args.log_dir = './output/vae/20190118/traj_debug_beta0.6_lr5e-4_h256_l8_layers4_b8'

    args.env_name = 'Point2DWalls-corner-v0'

    os.makedirs(args.log_dir, exist_ok=True)

    # set seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # train_vae(args)
    validate_vae(args)
