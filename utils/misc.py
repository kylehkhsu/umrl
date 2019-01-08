import os
import ipdb
import re
import torch
import copy
from a2c_ppo_acktr.utils import get_vec_normalize
import numpy as np


def save_model(args, policy, envs, iteration, sub_dir='ckpt'):
    os.makedirs(os.path.join(args.log_dir, sub_dir), exist_ok=True)
    if args.cuda:
        policy = copy.deepcopy(policy).cpu()    # apparently a really ugly way to save to CPU
    save_model = [policy, getattr(get_vec_normalize(envs.envs), 'ob_rms', None)]
    torch.save(save_model, os.path.join(args.log_dir, sub_dir, 'iteration_{}.pt'.format(iteration)))


def load_model(log_dir, iteration, sub_dir='ckpt'):
    return torch.load(os.path.join(log_dir, sub_dir, 'iteration_{}.pt'.format(iteration)))


def guard_against_underflow(x):
    assert x.dtype == np.float64
    if np.all(x <= 0):  # log-space
        x[x < -600] = -600
    elif np.all(x >= 0):    #
        x[x < 1e-300] = 1e-300
    else:
        raise NotImplementedError
    return x


def calculate_state_entropy(args, trajectories):
    if 'Point2D' in args.env_name:
        bins = 100
        bounds = (np.array([-10, 0]), np.array([-10, 10]))
    else:
        raise ValueError

    data = torch.cat(trajectories, dim=0).numpy()
    p_s = np.histogramdd(data, bins=bins, range=bounds, density=True)[0]
    H_s = -np.sum(p_s * np.ma.log(p_s))
    return H_s


if __name__ == '__main__':
    pass
