import os
import ipdb
import re
import torch
import copy
from a2c_ppo_acktr.utils import get_vec_normalize
import numpy as np


def save_model(args, policy, envs, iteration, sub_dir='ckpt'):
    os.makedirs(os.path.join(args.log_dir, sub_dir), exist_ok=True)
    if 'cuda' in args.device:
        policy = copy.deepcopy(policy).cpu()    # apparently a really ugly way to save to CPU
    save_model = [policy, getattr(get_vec_normalize(envs.envs), 'ob_rms', None)]
    torch.save(save_model, os.path.join(args.log_dir, sub_dir, 'iteration_{}.pt'.format(iteration)))


def load_model(log_dir, iteration, sub_dir='ckpt'):
    return torch.load(os.path.join(log_dir, sub_dir, 'iteration_{}.pt'.format(iteration)))


def guard_against_underflow(x):
    if isinstance(x, np.ndarray):

        assert x.dtype == np.float64
        if np.all(x <= 0):  # log-space
            x[x < -600] = -600
        elif np.all(x >= 0):    #
            x[x < 1e-300] = 1e-300
        else:
            raise NotImplementedError

    elif isinstance(x, torch.Tensor):
        x[x == -float('inf')] = -300
    else:
        raise NotImplementedError

    return x


def calculate_state_entropy(args, trajectories):
    if 'Point2D' in args.env_name:
        bins = 100
        bounds = (np.array([-10, 10]), np.array([-10, 10]))
        data = torch.cat(trajectories, dim=0).numpy()
        p_s = np.histogramdd(data, bins=bins, range=bounds, density=True)[0]
        H_s = -np.sum(p_s * np.ma.log(p_s))
    elif 'HalfCheetah' in args.env_name:
        bins = 100
        bounds = (np.array([-30, 30]), np.array([-5, 5]))
        data = torch.cat(trajectories, dim=0).numpy()[:, :2]
        p_s = np.histogramdd(data, bins=bins, range=bounds, density=True)[0]
        H_s = -np.sum(p_s * np.ma.log(p_s))
    elif 'Ant' in args.env_name:
        bins = 100
        bounds = (np.array([-10, 10]), np.array([-10, 10]), np.array([-10, 10]))
        data = torch.cat(trajectories, dim=0).numpy()[:, -3:]
        p_s = np.histogramdd(data, bins=bins, range=bounds, density=True)[0]
        H_s = -np.sum(p_s * np.ma.log(p_s))
    else:
        raise ValueError
    return H_s


class Normalizer:
    def __init__(self):
        self.count = torch.zeros(1)
        self.mean = torch.zeros(1)
        self.M2 = torch.zeros(1)

    def observe(self, x):
        for x_ in x:
            self.count.add_(1)
            delta = x_ - self.mean
            self.mean.add_(delta / self.count)
            delta2 = x_ - self.mean
            self.M2.add_(delta * delta2)

    def normalize(self, x):
        if self.count < 2:
            return x
        variance = self.M2 / self.count
        std = torch.clamp(torch.sqrt(variance), min=0.001)
        return (x - self.mean) / std

    def reset(self):
        self.mean.zero_()
        self.count.zero_()
        self.M2.zero_()


if __name__ == '__main__':
    def test_norm(x):
        norm = Normalizer()
        norm.observe(x)
        x_normed = norm.normalize(x)

        print(x_normed)

    x = torch.Tensor([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
    test_norm(x)

    x = torch.zeros(10)
    test_norm(x)
