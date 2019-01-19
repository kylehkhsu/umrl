import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb
# from sklearn.mixture import BayesianGaussianMixture
from mixture.models import BayesianGaussianMixture, GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer
from scipy.stats import multivariate_normal
from copy import deepcopy
import pickle
import os
import time
from gmm_abhishek import GMM_estep
from mixture.models.gaussian_mixture import _estimate_log_gaussian_prob
from collections import OrderedDict
from torch.distributions import Categorical
from a2c_ppo_acktr.envs import make_vec_envs
from abc import ABC, abstractmethod
from utils.misc import guard_against_underflow
from itertools import chain
from vae import VAE


class Rewarder:
    def __init__(self, args, **kwargs):
        self.args = args

    @abstractmethod
    def _calculate_reward(self, task, obs_raw, action, **kwargs):
        pass

    @abstractmethod
    def _sample_task_one(self, i_process):
        pass

    @abstractmethod
    def fit(self, **kwargs):
        pass

    def reset_episode(self):
        pass


class UnsupervisedRewarder(Rewarder):
    def __init__(self, args, obs_raw_shape, **kwargs):
        super(UnsupervisedRewarder, self).__init__(args, **kwargs)

        self.fit_counter = 0
        self.fit_counter_to_trajectories = []
        self.fit_counter_to_component_ids = []
        self.fit_counter_to_models = []
        self.episode_rewards = []

        # context type
        if self.args.context == 'mean':
            self.context_shape = obs_raw_shape
        elif self.args.context == 'all':
            self.context_shape = (obs_raw_shape[0] * 3,)    # mean and eigenvectors of cov
        elif self.args.context == 'latent':
            assert self.args.clusterer == 'vae'
            self.context_shape = (self.args.vae_latent_size,)
        else:
            raise ValueError

        if self.args.clusterer == 'mog':
            self.clusterer = \
                GaussianMixture(n_components=self.args.max_components,
                                covariance_type='full',
                                verbose=1,
                                verbose_interval=100,
                                max_iter=1000,
                                n_init=1)
        elif self.args.clusterer == 'dp-mog':
            self.clusterer = \
                BayesianGaussianMixture(n_components=self.args.max_components,
                                        covariance_type='full',
                                        verbose=1,
                                        verbose_interval=100,
                                        max_iter=1000,
                                        n_init=1,
                                        weight_concentration_prior=self.args.weight_concentration_prior,
                                        weight_concentration_prior_type='dirichlet_process')
        elif self.args.clusterer == 'diayn':
            raise NotImplementedError
        elif self.args.clusterer == 'vae':
            self.clusterer = VAE(args, obs_raw_shape[0])
            if self.args.vae_load:
                self.clusterer.load(iteration=0, load_from=self.args.vae_weights)
        else:
            raise ValueError

        self.p_z = np.empty(self.args.max_components, dtype=np.float32)
        self.component_id = np.zeros(self.args.num_processes, dtype=np.int)

        self.obs_raw_shape = obs_raw_shape

    def _calculate_reward(self, task_current, obs_raw, action, **kwargs):
        reward_info = dict()

        if self.fit_counter == 0 and not self.args.vae_load:
            reward = torch.zeros(self.args.num_processes)
            reward_info['log_marginal'] = torch.zeros(self.args.num_processes)
            reward_info['lambda_log_s_given_z'] = torch.zeros(self.args.num_processes)
            return reward, reward_info

        if self.args.reward == 's|z':
            if self.args.clusterer == 'vae':
                if kwargs.get('latent') is not None:
                    z = kwargs['latent']
                else:
                    z = torch.from_numpy(np.stack(task_current, axis=0).astype(dtype=np.float32))

                log_s_given_z = self.clusterer.log_s_given_z(s=obs_raw, z=z)
                log_marginal = self.clusterer.log_marginal(s=obs_raw)

                log_s_given_z = guard_against_underflow(log_s_given_z)
                log_marginal = guard_against_underflow(log_marginal)

                reward = -log_marginal + self.args.conditional_coef * log_s_given_z
                reward_info['log_marginal'] = log_marginal
                reward_info['lambda_log_s_given_z'] = self.args.conditional_coef * log_s_given_z

            elif self.args.clusterer == 'dp-mog' or self.args.clusterer == 'mog':

                X = obs_raw
                log_s_given_z = _estimate_log_gaussian_prob(
                    X, self.clusterer.means_, self.clusterer.precisions_cholesky_, self.clusterer.covariance_type
                )
                s_given_z = np.exp(log_s_given_z)
                s_joint_z = s_given_z * self.p_z[None, :]
                s_joint_z = guard_against_underflow(s_joint_z)
                p_s = s_joint_z.sum(axis=1)
                p_s_given_z = s_given_z[np.arange(s_given_z.shape[0]), self.component_id]
                reward = -np.log(p_s) + self.args.conditional_coef * np.log(p_s_given_z)

            else:
                raise ValueError
        else:
            raise ValueError

        if isinstance(reward, np.ndarray):
            reward = torch.from_numpy(reward)
        self.episode_rewards.append(reward)
        if self.args.cumulative_reward:
            rewards = torch.stack(self.episode_rewards, dim=0)
            reward = rewards.mean(dim=0)
        return reward, reward_info

    def get_assess_tasks(self):
        if self.args.clusterer == 'vae':
            tasks = np.random.randn(10, self.args.vae_latent_size)
        return tasks

    def _sample_task_one(self, i_process):
        if self.fit_counter == 0 and self.args.clusterer != 'vae':
            return None

        if self.args.clusterer == 'dp-mog' or self.args.clusterer == 'mog':

            z = np.random.choice(self.args.max_components, size=1, replace=False, p=self.p_z)[0]
            self.component_id[i_process] = z
            if self.args.context == 'mean':
                task = self.clusterer.means_[z]
            elif self.args.context == 'all':
                task = np.concatenate([self.clusterer.means_[z], self.evecs[z]])

        elif self.args.clusterer == 'vae':

            task = np.random.randn(self.args.vae_latent_size)

        return task

    def fit(self, trajectories, component_ids=None):
        self._pre_fit(trajectories, component_ids)

        if len(self.fit_counter_to_trajectories) > 0 and self.args.subsample_num < len(list(chain(*self.fit_counter_to_trajectories))):
            if self.args.subsample_strategy == 'random':

                data = list(chain(*self.fit_counter_to_trajectories))  # flatten list of lists
                indices = np.random.choice(len(data), self.args.subsample_num, replace=False)
                data = [data[index] for index in indices]

            elif self.args.subsample_strategy == 'skew':
                if self.args.clusterer == 'dp-mog' or self.args.clusterer == 'mog':

                    data = list(chain(*self.fit_counter_to_trajectories))  # flatten list of lists
                    _data = torch.cat(data, dim=0)
                    log_s_given_z = _estimate_log_gaussian_prob(
                        _data, self.clusterer.means_, self.clusterer.precisions_cholesky_, self.clusterer.covariance_type
                    )
                    # (i_trajectory, t, i_component)
                    log_s_given_z = log_s_given_z.reshape([-1, self.args.episode_length, log_s_given_z.shape[-1]])
                    log_tau_given_z = np.sum(log_s_given_z, axis=1)  # (i_trajectory, i_component)
                    log_tau_given_z = guard_against_underflow(log_tau_given_z)
                    tau_given_z = np.exp(log_tau_given_z)
                    tau_joint_z = tau_given_z * (self.p_z[None, :])
                    q_tau = np.sum(tau_joint_z, axis=1)   # (i_trajectory,)
                    assert np.min(q_tau) > 0
                    # p_tau /= np.sum(p_tau)    # redundant
                    skewed_q_tau = np.power(q_tau, self.args.subsample_power)
                    skewed_q_tau /= np.sum(skewed_q_tau)

                    indices = np.random.choice(len(data), self.args.subsample_num, replace=True, p=skewed_q_tau)
                    data = [data[index] for index in indices]

            elif self.args.subsample_strategy == 'last-random':
                data = [trajectories[-self.args.subsample_last_per_fit:] for trajectories in self.fit_counter_to_trajectories]
                data = list(chain(*data))  # flatten list of lists
                num_samples = min(self.args.subsample_num, len(data))
                indices = np.random.choice(len(data), num_samples, replace=False)
                data = [data[index] for index in indices]

            else:
                raise ValueError
        else:
            data = list(chain(*self.fit_counter_to_trajectories))           # else keep it all

        if self.args.clusterer == 'vae':
            # self.clusterer = VAE(self.args, self.obs_raw_shape[0], iteration=self.fit_counter)     # cold-start
            data = torch.stack(data)
            self.clusterer.to(self.args.device)
            self.clusterer.fit(data, iteration=self.fit_counter)

        elif self.args.clusterer == 'dp-mog' or self.args.clusterer == 'mog':
            data = torch.cat(data, dim=0)   # modified EM fitters take 2D input
            self.clusterer.fit(data, group=self.args.episode_length)
            evals, evecs = np.linalg.eigh(self.clusterer.covariances_)
            self.evecs = evecs.reshape([evecs.shape[0], -1])

            self.p_z = self.clusterer.weights_
            self.p_z /= self.p_z.sum()

        self._post_fit()

    def reset_episode(self):
        self.episode_rewards = []

    def append_model(self):
        if self.args.clusterer != 'vae':
            self.fit_counter_to_models.append(deepcopy(self.clusterer))

    def append_trajectories(self, trajectories):
        self.fit_counter_to_trajectories.append(trajectories)

    def dump(self):
        obj = dict(
            trajectories=self.fit_counter_to_trajectories,
            component_ids=self.fit_counter_to_component_ids,
            models=self.fit_counter_to_models,
        )
        filename = os.path.join(self.args.log_dir, 'history.pkl')
        pickle.dump(obj, open(filename, 'wb'))

    def _pre_fit(self, trajectories, component_ids):
        self.fit_counter_to_trajectories.append(trajectories)
        self.fit_counter_to_component_ids.append(component_ids)

    def _post_fit(self):
        self.append_model()
        self.fit_counter += 1
        self.dump()


class SupervisedRewarder(Rewarder):
    def __init__(self, args, **kwargs):
        super(SupervisedRewarder, self).__init__(args, **kwargs)
        if 'HalfCheetah' in self.args.env_name:
            if self.args.task_type == 'direction':
                if self.args.tasks == 'single':
                    self.context_shape = (1,)
                elif self.args.tasks == 'two':
                    self.context_shape = (2,)

        self.component_id = np.zeros(self.args.num_processes, dtype=np.int)

    def _calculate_reward(self, task, obs_raw, action, env_info=None, **kwargs):
        if 'Point2D' in self.args.env_name:
            goal = torch.Tensor(task)
            assert goal.shape == obs_raw.shape
            distance = torch.norm(goal - obs_raw.cpu(), dim=-1)
            dense_reward = -distance
            success_reward = (distance < 2).float()
            reward = self.args.dense_coef * dense_reward + self.args.success_coef * success_reward
        elif 'HalfCheetah' in self.args.env_name:
            velocity = torch.FloatTensor([env_info[i]['velocity'] for i in range(self.args.num_processes)]).unsqueeze(1)
            reward_ctrl = - 0.1 * (action.cpu() ** 2).sum(dim=-1).unsqueeze(1)

            if self.args.task_type == 'goal':
                raise NotImplementedError
                # goal = torch.Tensor(task)
                # vel = obs_raw[:, 8].unsqueeze(1)
                # assert goal.shape == vel.shape
                # distance = torch.norm(goal - vel.cpu(), dim=-1)
                # success_reward = (distance < 1).float()
                # squared_distance = distance ** 2
                # dense_reward = -squared_distance
                # reward = self.args.dense_coef * dense_reward + \
                #          self.args.success_coef * success_reward + reward_ctrl
            elif self.args.task_type == 'direction':
                direction = torch.Tensor(task)[:, 0:1] == 1
                direction = direction.float()
                direction[direction == 0] = -1
                assert torch.all((direction == 1) + (direction == -1))
                # vel = obs_raw[:, 8].unsqueeze(1).cpu()
                dense_reward = direction * velocity.cpu()
                reward = self.args.dense_coef * dense_reward + reward_ctrl
                reward = reward.squeeze(1)

        reward_info = {}

        return reward, reward_info

    def _sample_task_one(self, i_process):
        # rand = 2 * (np.random.random_sample((2,)) - 0.5)    # \in [-1, 1]
        # goal = self.envs.observation_space.spaces['state_observation'].sample()
        # goal = np.array([10, 10])
        goal = None

        if self.args.env_name == 'Point2DWalls-center-v0':
            if self.args.tasks == 'four':
                task_id = np.random.randint(low=0, high=4)
                if task_id == 0:
                    goal = np.array([5, 5])
                elif task_id == 1:
                    goal = np.array([5, -5])
                elif task_id == 2:
                    goal = np.array([-5, -5])
                elif task_id == 3:
                    goal = np.array([-5, 5])
            elif self.args.tasks == 'two':
                task_id = np.random.randint(low=0, high=2)
                if task_id == 0:
                    goal = np.array([5, 5])
                elif task_id == 1:
                    goal = np.array([-5, -5])
            elif self.args.tasks == 'single':
                goal = np.array([5, -5])
            elif self.args.tasks == 'all':
                goal = self.envs.observation_space.spaces['state_observation'].sample()
            elif self.args.tasks == 'ring':
                vec = np.random.normal(size=2)
                goal = 5 * vec / np.linalg.norm(vec, axis=0)

        elif self.args.env_name == 'Point2DWalls-corner-v0':
            if self.args.tasks == 'single':
                goal = np.array([-2.5, -7.5])
            elif self.args.tasks == 'two':
                task_id = np.random.randint(low=0, high=2)
                if task_id == 0:
                    goal = np.array([5, 5])
                elif task_id == 1:
                    goal = np.array([-2.5, -7.5])
            elif self.args.tasks == 'four':
                task_id = np.random.randint(low=0, high=4)
                if task_id == 0:
                    goal = np.array([5, 5])
                elif task_id == 1:
                    goal = np.array([-2.5, -7.5])
                elif task_id == 2:
                    goal = np.array([8, -8])
                elif task_id == 3:
                    goal = np.array([-8, 8])

        elif 'HalfCheetah' in self.args.env_name:
            if self.args.task_type == 'direction':
                if self.args.tasks == 'single':
                    goal = np.array([1])
                elif self.args.tasks == 'two':
                    task_id = np.random.randint(low=0, high=2)
                    if task_id == 0:
                        goal = np.array([1, 0])
                    elif task_id == 1:
                        goal = np.array([0, 1])
            elif self.args.task_type == 'goal':
                if self.args.tasks == 'single':
                    goal = np.array([3])
                elif self.args.tasks == 'two':
                    task_id = np.random.randint(low=0, high=2)
                    if task_id == 0:
                        goal = np.array([3])
                    elif task_id == 1:
                        goal = np.array([-3])
                elif self.args.tasks == 'four':
                    task_id = np.random.randint(low=0, high=4)
                    if task_id == 0:
                        goal = np.array([3])
                    elif task_id == 1:
                        goal = np.array([-3])
                    elif task_id == 2:
                        goal = np.array([5])
                    elif task_id == 3:
                        goal = np.array([-5])

        if goal is None:
            raise ValueError

        return goal

    def get_assess_tasks(self):
        tasks = None
        if 'HalfCheetah' in self.args.env_name:
            if self.args.task_type == 'direction':
                if self.args.tasks == 'single':
                    tasks = np.array([[1]])
                elif self.args.tasks == 'two':
                    tasks = np.array([[1, 0],
                                      [0, 1]])
        elif self.args.env_name == 'Point2DWalls-center-v0':
            if self.args.tasks == 'four':
                tasks = np.array([[5, 5],
                                  [5, -5],
                                  [-5, -5],
                                  [-5, 5]])
            elif self.args.tasks == 'single':
                tasks = np.array([[5, -5]])
            elif self.args.tasks == 'ring':
                x = np.sqrt(25/2)
                tasks = np.array([[0, 5],
                                  [5, 0],
                                  [0, -5],
                                  [-5, 0],
                                  [x, x],
                                  [x, -x],
                                  [-x, x],
                                  [-x, -x]])
        elif self.args.env_name == 'Point2DWalls-corner-v0':
            tasks = np.array([[-10, -10],
                              [-8, -8],
                              [-8, -5],
                              [-8, 0],
                              [-8, 5],
                              [-10, 10],
                              [0, 0],
                              [10, 0],
                              [8, 8]])
        if tasks is None:
            raise ValueError
        return tasks

    def fit(self, *args, **kwargs):
        return

    def reset_episode(self, *args, **kwargs):
        return


