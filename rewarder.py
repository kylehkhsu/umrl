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
from gmm import GMM_estep
from mixture.models.gaussian_mixture import _estimate_log_gaussian_prob
from collections import OrderedDict
from torch.distributions import Categorical
from a2c_ppo_acktr.envs import make_vec_envs
from abc import ABC, abstractmethod
from utils.misc import guard_against_underflow
from itertools import chain


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


class UnsupervisedRewarder(Rewarder):
    def __init__(self, args, obs_raw_shape, **kwargs):
        super(UnsupervisedRewarder, self).__init__(args, **kwargs)

        self.fit_counter = 0
        self.fit_counter_to_trajectories = []
        self.fit_counter_to_component_ids = []
        self.fit_counter_to_models = []
        self.episode_rewards = []

        if self.args.context == 'mean':
            self.context_shape = obs_raw_shape
        elif self.args.context == 'all':
            self.context_shape = (obs_raw_shape[0] * 3,)    # mean and eigenvectors of cov
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
        else:
            raise ValueError

        self.p_z = np.empty(self.args.max_components, dtype=np.float32)
        self.component_id = np.zeros(self.args.num_processes, dtype=np.int)

    def _calculate_reward(self, _, obs_raw, action, **kwargs):
        if self.fit_counter == 0:
            return torch.zeros(self.args.num_processes)

        X = obs_raw

        if self.args.reward == 's|z':
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

        reward = torch.from_numpy(reward)
        self.episode_rewards.append(reward)
        if self.args.cumulative_reward:
            rewards = torch.stack(self.episode_rewards, dim=0)
            reward = rewards.mean(dim=0)

        return reward

    def _sample_task_one(self, i_process):
        if self.fit_counter == 0:
            return None
        z = np.random.choice(self.args.max_components, size=1, replace=False, p=self.p_z)[0]
        self.component_id[i_process] = z
        if self.args.context == 'mean':
            task = self.clusterer.means_[z]
        elif self.args.context == 'all':
            task = np.concatenate([self.clusterer.means_[z], self.evecs[z]])
        return task

    def fit(self, trajectories, component_ids):
        self._pre_fit(trajectories, component_ids)
        data = list(chain(*self.fit_counter_to_trajectories))   # flatten list of lists
        if self.args.subsample_num < len(data):
            if self.args.subsample_strategy == 'random':
                indices = np.random.choice(len(data), self.args.subsample_num, replace=True)
                data = [data[index] for index in indices]
            elif self.args.subsample_strategy == 'skew':
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
        # else keep it all

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
        self.fit_counter_to_models.append(deepcopy(self.clusterer))

    def append_trajectories(self, trajectories):
        self.fit_counter_to_trajectories.append(trajectories)

    def dump(self):
        assert len(self.fit_counter_to_trajectories) == len(self.fit_counter_to_models)
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

    def _calculate_reward(self, task, obs_raw, action, env_info=None):
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
                direction = torch.Tensor(task)
                assert torch.all((direction==1) + (direction==-1))
                # vel = obs_raw[:, 8].unsqueeze(1).cpu()
                dense_reward = direction * velocity.cpu()
                reward = self.args.dense_coef * dense_reward + reward_ctrl
                reward = reward.squeeze(1)

        return reward

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

        elif self.args.env_name == 'HalfCheetahVel-v0':
            if self.args.task_type == 'direction':
                if self.args.tasks == 'single':
                    goal = np.array([1])
                elif self.args.tasks == 'two':
                    task_id = np.random.randint(low=0, high=2)
                    if task_id == 0:
                        goal = np.array([1])
                    elif task_id == 1:
                        goal = np.array([-1])
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
        if self.args.env_name == 'HalfCheetahVel-v0':
            if self.args.task_type == 'direction':
                if self.args.tasks == 'two':
                    tasks = np.array([-1, 1])
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
            x = np.sqrt(25 / 2)
            tasks = np.array([[0, 5],
                              [5, 0],
                              [0, -5],
                              [-5, 0],
                              [x, x],
                              [x, -x],
                              [-x, x],
                              [-x, -x]])
        if tasks is None:
            raise ValueError
        return tasks

    def fit(self):
        return


# class RewarderDeprecated(object):
#
#     def __init__(self,
#                  args,
#                  obs_shape,
#                  logger):
#         self.args = args
#         self.obs_shape = obs_shape
#         self.raw_trajectory_embeddings = []     # data for GMM
#         self.raw_trajectories = []
#         self.raw_trajectory_current = [torch.zeros(size=obs_shape).unsqueeze(0) for i in range(args.num_processes)]
#         self.history = History(args)
#         self.clustering_counter = 0
#         # self.encoding_function = lambda x: x
#
#         self.tasks = []
#         self.task_current = [
#             -1 for i in range(args.num_processes)
#         ]
#         self.task_index_current = np.zeros(args.num_processes, dtype=np.int)
#
#         self.step_counter = [0 for i in range(args.num_processes)]
#
#         group = dict(state=self.args.episode_length,
#                      trajectory_embedding=None)[self.args.cluster_on]
#         if self.args.clusterer == 'bayesian':
#             self.generative_model = \
#                 BayesianGaussianMixture(n_components=self.args.max_components,
#                                         covariance_type='full',
#                                         verbose=1,
#                                         verbose_interval=100,
#                                         max_iter=1000,
#                                         n_init=1,
#                                         weight_concentration_prior_type='dirichlet_process',
#                                         weight_concentration_prior=self.args.weight_concentration_prior,
#                                         group=group)
#         elif self.args.clusterer == 'gaussian':
#             self.generative_model = \
#                 GaussianMixture(n_components=self.args.max_components,
#                                 covariance_type='full',
#                                 verbose=1,
#                                 verbose_interval=100,
#                                 max_iter=1000,
#                                 n_init=1,
#                                 group=group)
#         elif self.args.clusterer == 'discriminator':
#             self.generative_model = Discriminator(input_size=self.obs_shape[0],
#                                                   output_size=self.args.max_components)
#         else:
#             raise ValueError
#
#         self.valid_components = np.array([0])
#         self.p_z = np.array([1])
#
#         self.standardizer = StandardScaler()
#         self.logger = logger
#         self.discriminator_loss = 0
#
#     def fit_generative_model(self):
#         # if self.clustering_counter > 0:
#             # ipdb.set_trace()
#         self.history.new()
#
#         if self.args.cluster_on == 'trajectory_embedding':
#             data_list = self.raw_trajectory_embeddings
#             assert self.generative_model.group is None
#         elif self.args.cluster_on == 'state':
#             data_list = self.raw_trajectories
#         else:
#             raise ValueError
#         tasks = self.tasks
#
#         if self.args.cluster_subsample_strategy == 'last':
#             data_list = data_list[-self.args.cluster_subsample_num:]
#             tasks = tasks[-self.args.cluster_subsample_num:]
#         elif self.args.cluster_subsample_strategy == 'random' and len(data_list) > self.args.cluster_subsample_num:
#             indices = np.random.choice(len(data_list), self.args.cluster_subsample_num, replace=False)
#             data_list = [data_list[index] for index in indices]
#             tasks = [tasks[index] for index in indices]
#         # else keep it all
#
#         data = torch.cat(data_list, dim=0)
#
#         if not self.args.keep_entire_history:
#             self.raw_trajectory_embeddings = []
#             self.raw_trajectories = []
#             self.tasks = []
#
#         if self.args.standardize_data:
#             self.standardizer.fit(data)
#             data = self._get_standardized_data(data)
#
#         if self.args.clusterer == 'discriminator':
#             assert self.args.cluster_on == 'state'
#             self._train_discriminator(data, tasks)
#             self.clustering_counter += 1
#             self.logger.log('clustering_counter: {}'.format(self.clustering_counter))
#             # self.history.save_generative_model(self.generative_model, self.standardizer)
#         else:
#             self.generative_model.fit(data)
#
#             components = np.argwhere(self.generative_model.weights_ >= self.args.component_weight_threshold).reshape([-1])
#             valid_components = []
#             for i, component in enumerate(components):
#                 if i == 0:
#                     valid_components.append(component)
#                     continue
#                 current_mean = self.generative_model.means_[component]
#                 prev_means = self.generative_model.means_[components[:i]]
#                 l_2 = min(np.linalg.norm(prev_means - current_mean, ord=2, axis=1))
#                 l_inf = min(np.linalg.norm(prev_means - current_mean, ord=np.inf, axis=1))
#                 if l_2 >= self.args.component_constraint_l_2 or l_inf >= self.args.component_constraint_l_inf:
#                     valid_components.append(component)
#             self.valid_components = np.array(valid_components)
#
#             if self.args.log_EM:
#                 self.logger.log('clustering_counter: {}'.format(self.clustering_counter))
#                 self.logger.log('raw means of valid components:\n{}'.format(self._get_raw_means(self.valid_components)))
#                 self.logger.log('standardized means of valid components:\n{}'.format(self.generative_model.means_[self.valid_components]))
#
#             self.clustering_counter += 1
#             self.history.save_generative_model(self.generative_model, self.standardizer)
#             self._calculate_sampling_distribution(data)
#
#     def _calculate_sampling_distribution(self, data):
#         assert data.dim() == 2
#
#         # max I
#         log_gauss = _estimate_log_gaussian_prob(data,
#                                                 self.generative_model.means_,
#                                                 self.generative_model.precisions_cholesky_,
#                                                 self.generative_model.covariance_type)
#         gauss = np.exp(log_gauss[:, self.valid_components])
#         joint = gauss * self.generative_model.weights_[None, self.valid_components]
#         joint[joint < 1e-300] = 1e-300
#         denominator = np.sum(joint, axis=1)
#         log_posterior = np.log(joint) - np.log(denominator)[:, None]
#         p_z = np.mean(log_posterior, axis=0)
#         p_z = np.exp(p_z)
#         p_z /= np.sum(p_z)
#
#         # data = torch.reshape(data, [-1, self.args.episode_length, data.shape[1]])
#         # p_z = np.zeros(len(self.valid_components))
#         # N = min(10000, data.shape[0])
#         # for n in range(N):
#         #     for t in range(self.args.episode_length):
#         #         x = data[-n][t]
#         #         joint = np.zeros(len(self.valid_components))
#         #         for i, c in enumerate(self.valid_components):
#         #             density = multivariate_normal.pdf(x=x,
#         #                                               mean=self.generative_model.means_[c],
#         #                                               cov=self.generative_model.covariances_[c])
#         #             joint[i] = density * self.generative_model.weights_[c]
#         #         # if np.any(np.log(joint) == -np.inf):
#         #         #     ipdb.set_trace()
#         #         log_joint = np.log(joint) - np.log(np.sum(joint))
#         #         p_z += log_joint
#         # p_z /= (N * self.args.episode_length)
#         # p_z2 = np.exp(p_z)
#         # p_z2 /= np.sum(p_z2)
#         # ipdb.set_trace()
#
#         # U(z)
#         U_z = np.ones(len(self.valid_components))
#         U_z /= np.sum(U_z)
#
#         # EM
#         p_z_EM = self.generative_model.weights_[self.valid_components]
#         p_z_EM /= np.sum(p_z_EM)
#
#         if self.args.task_sampling == 'max_I':
#             self.p_z = p_z
#         elif self.args.task_sampling == 'uniform':
#             self.p_z = U_z
#         elif self.args.task_sampling == 'EM':
#             self.p_z = p_z_EM
#         else:
#             raise ValueError
#
#         if self.args.log_EM:
#             self.logger.log('max I p_z: {}'.format(p_z))
#             self.logger.log('EM p_z: {}'.format(p_z_EM))
#
#     def _sample_task_one(self, i_process):
#         if self.args.context == 'goal':
#             position = np.random.uniform(low=-5, high=5, size=2)
#             speed = np.random.uniform(low=0.1, high=1, size=1)
#             self.context[i_process] = np.concatenate((position, speed)).astype(np.float32)
#         else:
#             z = np.random.choice(self.valid_components, size=1, replace=False, p=self.p_z)[0]
#             self.task_current[i_process] = z
#             self.task_index_current[i_process] = np.argwhere(self.valid_components == z)[0][0]
#             if self.args.context == 'cluster_mean':
#                 if self.clustering_counter != 0:
#                     self.context[i_process] = self._get_raw_means(z).astype(np.float32)
#             elif self.args.context == 'one_hot':
#                 context = np.zeros(self.args.max_components, dtype=np.float32)
#                 context[z] = 1
#                 self.context[i_process] = context[None, :]
#             else:
#                 raise ValueError
#
#     def _reset_one(self, i_process, raw_obs):
#         self.raw_trajectory_current[i_process] = torch.zeros(size=self.obs_shape).unsqueeze(0)
#         self.raw_trajectory_current[i_process][0][:raw_obs.shape[1]] = raw_obs[i_process]
#         self._sample_task_one(i_process)
#         self.step_counter[i_process] = 0
#
#     def reset(self, raw_obs):
#         for i in range(self.args.num_processes):
#             self._reset_one(i, raw_obs)
#         return torch.cat(
#             [torch.cat(self.raw_trajectory_current, dim=0), torch.from_numpy(np.concatenate(self.context, axis=0))], dim=1
#         )
#
#     def _append_to_trajectory_one(self, i_process, obs):
#         assert self.raw_trajectory_current[i_process] is not None
#         self.raw_trajectory_current[i_process] = torch.cat(
#             (self.raw_trajectory_current[i_process], obs[i_process].unsqueeze(0)), dim=0
#         )
#
#     def _get_raw_means(self, i):
#         mean = self.generative_model.means_[i]
#         if mean.ndim == 1:
#             mean = mean.reshape([1, -1])
#         if self.args.standardize_data:
#             mean = self.standardizer.inverse_transform(mean)
#         return mean
#
#     def step(self, raw_obs, done, infos):
#         obs = self._process_obs(raw_obs)
#         for i in range(self.args.num_processes):
#             self.step_counter[i] += 1
#             done_ = self.step_counter[i] == self.args.episode_length
#             if done_:
#                 self._save_trajectory(i)
#             self._append_to_trajectory_one(i, obs)
#             done[i] = done_
#         reward_start = time.time()
#         reward = self._calculate_reward(done, obs=obs)
#         reward_time = time.time() - reward_start
#
#         reward = torch.from_numpy(reward).unsqueeze(1)
#         context = torch.from_numpy(np.concatenate(self.context, axis=0))
#         return torch.cat([obs, context], dim=1), reward, done, infos, reward_time
#
#     def _calculate_reward(self, done, obs):
#         assert (all(done) or not any(done))
#         if self.args.sparse_reward and not any(done):
#             return 0
#
#         if self.args.cluster_on == 'state':
#             X = obs
#         elif self.args.cluster_on == 'trajectory_embedding':
#             X = torch.cat([self.trajectory_to_embedding(i) for i in range(self.args.num_processes)])
#
#         if self.args.clusterer == 'discriminator':
#             with torch.no_grad():
#                 reward = self.generative_model(obs).log_prob(torch.ones(obs.shape[0]) *
#                                                              torch.from_numpy(self.task_index_current.astype(np.float32)))
#             reward = reward.numpy()
#         else:
#             if self.clustering_counter == 0:
#                 reward = np.zeros(X.shape[0], dtype=np.float32)
#             else:
#                 if self.args.reward == 'z|w':
#                     log_gauss = _estimate_log_gaussian_prob(X,
#                                                             self.generative_model.means_,
#                                                             self.generative_model.precisions_cholesky_,
#                                                             self.generative_model.covariance_type)
#                     gauss = np.exp(log_gauss[:, self.valid_components])
#                     density = gauss * self.p_z[None, :]
#                     density[density < 1e-300] = 1e-300
#                     denominator = np.sum(density, axis=1)
#                     numerator = density[np.arange(self.args.num_processes), self.task_index_current]
#                     reward = np.log(numerator) - np.log(denominator)
#                     reward = reward.astype(np.float32)
#                 elif self.args.reward == 'w|z':
#                     log_gauss = _estimate_log_gaussian_prob(X,
#                                                             self.generative_model.means_,
#                                                             self.generative_model.precisions_cholesky_,
#                                                             self.generative_model.covariance_type)
#                     gauss = np.exp(log_gauss[:, self.valid_components])
#                     joint = gauss * self.p_z[None, :]
#                     joint[joint < 1e-300] = 1e-300
#                     marginal = joint.sum(axis=1)
#                     conditional = joint[np.arange(self.args.num_processes), self.task_index_current]
#                     reward = - np.log(marginal) + self.args.conditional_coef * np.log(conditional)
#         return reward
#
#     def _calculate_reward_old(self, i_process, done):
#         if self.args.sparse_reward and not done:
#             return 0
#
#         time1 = time.time()
#         if self.args.cluster_on == 'state':
#             obs = self.raw_trajectory_current[i_process][-1].unsqueeze(0)
#             x = obs
#         elif self.args.cluster_on == 'trajectory_embedding':
#             embedding = self._get_standardized_data(self.trajectory_to_embedding(i_process))
#             x = embedding
#         else:
#             raise ValueError
#         print('time1: {}'.format(time.time() - time1))
#
#         z = self.task_current[i_process]
#
#         time2 = time.time()
#
#         if self.args.reward == 'z|w':
#             numerator, denominator = 0, 0
#             for i, z_ in enumerate(self.valid_components):
#                 density = multivariate_normal.pdf(x=x,
#                                                   mean=self.generative_model.means_[z_],
#                                                   cov=self.generative_model.covariances_[z_])
#                 denominator += density * self.p_z[i]
#                 if z == z_:
#                     numerator = density * self.p_z[i]
#             if numerator == 0:  # underflow
#                 r = -100
#             else:
#                 r = np.log(numerator) - np.log(denominator)
#         elif self.args.reward == 'w|z':
#             r = multivariate_normal.logpdf(x=x,
#                                            mean=self.generative_model.means_[z],
#                                            cov=self.generative_model.covariances_[z])
#         elif self.args.reward == 'l2':
#             diff = x - torch.from_numpy(self.context[i_process])
#             if self.args.obs == 'pos_speed':
#                 diff = diff * torch.Tensor([1, 1, 10])
#             r = -torch.norm(diff)
#         else:
#             raise ValueError
#         print('time2: {}'.format(time.time() - time2))
#
#         # time3 = time.time()
#         # X = torch.cat([x for i in range(10)], dim=0)
#         #
#         # log_gauss = _estimate_log_gaussian_prob(X,
#         #                                         self.generative_model.means_,
#         #                                         self.generative_model.precisions_cholesky_,
#         #                                         self.generative_model.covariance_type)
#         # print('time3: {}'.format(time.time() - time3))
#         return r
#
#     # def _calculate_reward_parallelized
#
#     def _save_trajectory(self, i_process):
#         self.raw_trajectory_embeddings.append(self.trajectory_to_embedding(i_process))
#         self.raw_trajectories.append(self.raw_trajectory_current[i_process])
#         self.history.save_episode(self.raw_trajectory_current[i_process], self.task_current[i_process])
#         self.tasks.append(self.task_current[i_process])
#
#     def _process_obs(self, raw_obs):
#         if self.args.obs == 'pos_speed':
#             speed = []
#             for i in range(self.args.num_processes):
#                 speed.append(torch.norm((raw_obs[i] - self.raw_trajectory_current[i][-1][:raw_obs.shape[1]]).unsqueeze(0)).unsqueeze(0))
#             return torch.cat([raw_obs, torch.stack(speed, dim=0)], dim=1)
#         elif self.args.obs == 'pos':
#             return raw_obs
#         else:
#             raise ValueError
#
#     def trajectory_to_embedding(self, i_process=0):
#         if self.args.trajectory_embedding_type == 'avg':
#             trajectory_embedding = torch.mean(self.raw_trajectory_current[i_process], dim=0).unsqueeze(0)
#         elif self.args.trajectory_embedding_type == 'final':
#             trajectory_embedding = self.raw_trajectory_current[i_process][-1].unsqueeze(0)
#         else:
#             raise NotImplementedError
#         return trajectory_embedding
#
#     def _get_standardized_data(self, data):
#         if self.args.standardize_data:
#             data = self.standardizer.transform(data)
#         return data
#
#     def _train_discriminator(self, data, tasks):
#         discriminator = self.generative_model
#         assert discriminator.__class__.__name__ == 'Discriminator'
#         assert data.dim() == 2
#
#         num_trajectories = data.shape[0] // self.args.episode_length
#         if self.clustering_counter == 0:
#             tasks = torch.randint(low=0, high=self.args.max_components,
#                                   size=(num_trajectories,))
#         assert len(tasks) == num_trajectories
#         labels = []
#         for task in tasks:
#             labels.append(torch.ones(self.args.episode_length, dtype=torch.long) * task)
#         labels = torch.cat(labels, dim=0)
#         batch_size = 100
#         idx = np.arange(data.shape[0])
#         num_batches = data.shape[0] // batch_size
#         num_epochs = 10
#         loss_function = nn.CrossEntropyLoss()
#         for epoch in range(num_epochs):
#             loss = 0
#             np.random.shuffle(idx)
#             for i in range(num_batches):
#                 idx_batch = idx[i*batch_size : (i+1)*batch_size]
#                 inputs_batch, labels_batch = data[idx_batch], labels[idx_batch]
#                 discriminator.optimizer.zero_grad()
#                 outputs_batch = discriminator(inputs_batch).logits
#                 loss_batch = loss_function(outputs_batch, labels_batch)
#                 loss_batch.backward()
#                 discriminator.optimizer.step()
#                 loss += loss_batch.item()
#             # print('discriminator epoch {}\tloss {}'.format(epoch, loss/data.shape[0]))
#         self.discriminator_loss = loss / data.shape[0]
#
#         self.valid_components = np.arange(self.args.max_components)
#         self.p_z = np.ones(self.args.max_components) / self.args.max_components
#
#
#
#
# import torch.optim as optim
# class Discriminator(nn.Module):
#     def __init__(self, input_size, output_size, hidden_sizes=(64, 64), nonlinearity=F.relu):
#         super(Discriminator, self).__init__()
#         self.input_size = input_size
#         self.output_size = output_size
#         self.hidden_sizes = hidden_sizes
#         self.nonlinearity = nonlinearity
#         self.num_layers = len(hidden_sizes) + 1
#         layer_sizes = (input_size,) + hidden_sizes + (output_size,)
#         for i in range(1, self.num_layers + 1):
#             self.add_module('layer{0}'.format(i),
#                             nn.Linear(layer_sizes[i - 1], layer_sizes[i]))
#
#         def weight_init(module):
#             if isinstance(module, nn.Linear):
#                 nn.init.xavier_uniform_(module.weight)
#                 module.bias.data.zero_()
#
#         self.apply(weight_init)
#         self.optimizer = optim.Adam(self.parameters(), lr=1e-3, betas=(0.9, 0.999))
#
#     def forward(self, input, params=None):
#         if params is None:
#             params = OrderedDict(self.named_parameters())
#         output = input
#         for i in range(1, self.num_layers + 1):
#             output = F.linear(output,
#                               weight=params['layer{0}.weight'.format(i)],
#                               bias=params['layer{0}.bias'.format(i)])
#             if i != self.num_layers:
#                 output = self.nonlinearity(output)
#         return Categorical(logits=output)


