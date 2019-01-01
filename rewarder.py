import numpy as np
import torch
import ipdb
# from sklearn.mixture import BayesianGaussianMixture
from mixture import BayesianGaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer
from scipy.stats import multivariate_normal
from copy import deepcopy
import pickle
import os
import time
from gmm import GMM_estep


class History(object):
    def __init__(self, args=None):
        self.args = args
        self.episodes = []
        self.generative_model = None
        self.standardizer = None
        self.filename = os.path.join(self.args.log_dir, 'history.pkl')
        self.all = []

    def new(self):
        if self.generative_model is not None and len(self.episodes) > 0:
            self.all.append(dict(generative_model=self.generative_model,
                                 standardizer=self.standardizer,
                                 episodes=self.episodes))
        self.generative_model = None
        self.standardizer = None
        self.episodes = []

    def save_generative_model(self, generative_model, standardizer):
        assert self.generative_model is None
        self.generative_model = deepcopy(generative_model)
        self.standardizer = deepcopy(standardizer)

    def save_episode(self, trajectory, task):
        self.episodes.append((trajectory, task))

    def dump(self):
        pickle.dump(self, open(self.filename, 'wb'))

    def load(self):
        return pickle.load(open(self.filename, 'rb'))


def transform(x):
    return x * np.array([0, 0, 1])

def inverse_transform(x):
    return x / np.array([1, 1, 1])

class Rewarder(object):

    def __init__(self,
                 args,
                 obs_shape,
                 logger):
        self.args = args
        self.obs_shape = obs_shape
        self.raw_trajectory_embeddings = []     # data for GMM
        self.raw_trajectories = []
        self.raw_trajectory_current = [torch.zeros(size=obs_shape).unsqueeze(0) for i in range(args.num_processes)]
        self.history = History(args)
        self.clustering_counter = 0
        # self.encoding_function = lambda x: x
        self.context = [
            np.zeros(shape=obs_shape, dtype=np.float32) for i in range(args.num_processes)
        ]
        self.task = [
            -1 for i in range(args.num_processes)
        ]

        self.step_counter = [0 for i in range(args.num_processes)]

        group = dict(state=self.args.episode_length,
                     trajectory_embedding=None)[self.args.cluster_on]
        self.generative_model = \
            BayesianGaussianMixture(n_components=self.args.max_components,
                                    covariance_type='full',
                                    verbose=1,
                                    verbose_interval=100,
                                    max_iter=5000,
                                    n_init=1,
                                    weight_concentration_prior_type='dirichlet_process',
                                    group=group)

        self.valid_components = None
        self.standardizer = StandardScaler()
        self.p_z = None
        self.logger = logger

    def fit_generative_model(self, data=None):
        # if self.clustering_counter > 0:
            # ipdb.set_trace()
        self.history.new()
        if data is not None:    # first fitting
            assert self.clustering_counter == 0
            if self.args.cluster_on == 'trajectory_embedding' and self.args.trajectory_embedding_type == 'avg':
                assert data.dim() == 3 and data.shape[1] == self.args.episode_length
                data = torch.mean(data, dim=1)
            elif self.args.cluster_on == 'state':
                # data = torch.cat(data, dim=0)
                data = torch.reshape(data, [-1, data.shape[-1]])
            else:
                raise ValueError
        else:
            if self.args.cluster_on == 'trajectory_embedding':
                data = torch.cat(self.raw_trajectory_embeddings, dim=0)
                assert self.generative_model.group is None
            elif self.args.cluster_on == 'state':
                data = torch.cat(self.raw_trajectories[-self.args.cluster_subsample:], dim=0)
                assert self.generative_model.group == self.args.episode_length

        if not self.args.fit_on_entire_history:
            self.raw_trajectory_embeddings = []
            self.raw_trajectories = []

        if self.args.standardize_data:
            self.standardizer.fit(data)

        data = self._get_standardized_data(data)

        self.generative_model.fit(data)

        components = np.argwhere(self.generative_model.weights_ >= self.args.component_weight_threshold).reshape([-1])
        valid_components = []
        for i, component in enumerate(components):
            if i == 0:
                valid_components.append(component)
                continue
            current_mean = self.generative_model.means_[component]
            prev_means = self.generative_model.means_[components[:i]]
            l_2 = min(np.linalg.norm(prev_means - current_mean, ord=2, axis=1))
            l_inf = min(np.linalg.norm(prev_means - current_mean, ord=np.inf, axis=1))
            if l_2 >= self.args.component_constraint_l_2 or l_inf >= self.args.component_constraint_l_inf:
                valid_components.append(component)
        self.valid_components = np.array(valid_components)

        self.logger.log('clustering_counter: {}'.format(self.clustering_counter))
        self.logger.log('raw means of valid components:\n{}'.format(self._get_raw_means(self.valid_components)))
        self.logger.log('standardized means of valid components:\n{}'.format(self.generative_model.means_[self.valid_components]))

        self.clustering_counter += 1
        self.history.save_generative_model(self.generative_model, self.standardizer)
        self._calculate_sampling_distribution(data)

    def _calculate_sampling_distribution(self, data):
        assert data.dim() == 2

        # max I
        data = torch.reshape(data, [-1, self.args.episode_length, data.shape[1]])
        p_z = np.zeros(len(self.valid_components))
        N = min(10000, data.shape[0])
        for n in range(N):
            for t in range(self.args.episode_length):
                x = data[-n][t]
                joint = np.zeros(len(self.valid_components))
                for i, c in enumerate(self.valid_components):
                    density = multivariate_normal.pdf(x=x,
                                                      mean=self.generative_model.means_[c],
                                                      cov=self.generative_model.covariances_[c])
                    joint[i] = density * self.generative_model.weights_[c]
                log_joint = np.log(joint) - np.log(np.sum(joint))
                p_z += log_joint
        p_z /= (N * self.args.episode_length)
        p_z = np.exp(p_z)
        p_z /= np.sum(p_z)

        # U(z)
        U_z = np.ones(len(self.valid_components))
        U_z /= np.sum(U_z)

        # EM
        p_z_EM = self.generative_model.weights_[self.valid_components]
        p_z_EM /= np.sum(p_z_EM)

        if self.args.task_sampling == 'max_I':
            self.p_z = p_z
        elif self.args.task_sampling == 'uniform':
            self.p_z = U_z
        elif self.args.task_sampling == 'EM':
            self.p_z = p_z_EM
        else:
            raise ValueError

        self.logger.log('max I p_z: {}'.format(p_z))
        self.logger.log('EM p_z: {}'.format(p_z_EM))

    def _sample_task_one(self, i_process):
        if self.args.context == 'goal':
            position = np.random.uniform(low=-5, high=5, size=2)
            speed = np.random.uniform(low=0.1, high=1, size=1)
            self.context[i_process] = np.concatenate((position, speed)).astype(np.float32)
        elif self.args.context == 'cluster_mean':
            z = np.random.choice(self.valid_components, size=1, replace=False, p=self.p_z)[0]
            self.task[i_process] = z
            self.context[i_process] = self._get_raw_means(z).astype(np.float32)

    def _reset_one(self, i_process, raw_obs):
        self.raw_trajectory_current[i_process] = torch.zeros(size=self.obs_shape).unsqueeze(0)
        self.raw_trajectory_current[i_process][0][:raw_obs.shape[1]] = raw_obs[i_process]
        self._sample_task_one(i_process)
        self.step_counter[i_process] = 0

    def reset(self, raw_obs):
        for i in range(self.args.num_processes):
            self._reset_one(i, raw_obs)
        return torch.cat(
            [torch.cat(self.raw_trajectory_current, dim=0), torch.from_numpy(np.concatenate(self.context, axis=0))], dim=1
        )

    def _append_to_trajectory_one(self, i_process, obs):
        assert self.raw_trajectory_current[i_process] is not None
        self.raw_trajectory_current[i_process] = torch.cat(
            (self.raw_trajectory_current[i_process], obs[i_process].unsqueeze(0)), dim=0
        )

    def _get_raw_means(self, i):
        mean = self.generative_model.means_[i]
        if mean.ndim == 1:
            mean = mean.reshape([1, -1])
        if self.args.standardize_data:
            mean = self.standardizer.inverse_transform(mean)
        return mean

    def step(self, raw_obs, done, infos):
        obs = self._process_obs(raw_obs)
        reward = []
        reward_time = 0
        for i in range(self.args.num_processes):
            self.step_counter[i] += 1
            done_ = self.step_counter[i] == self.args.episode_length
            if done_:
                self._save_trajectory(i)
            self._append_to_trajectory_one(i, obs)
            done[i] = done_
            reward_start = time.time()
            reward.append(self._calculate_reward(i, done_))
            reward_time += time.time() - reward_start
        reward = torch.from_numpy(np.array(reward, dtype=np.float32)).unsqueeze(1)
        context = torch.from_numpy(np.concatenate(self.context, axis=0))
        return torch.cat([obs, context], dim=1), reward, done, infos, reward_time

    def _calculate_reward(self, i_process, done):
        if self.args.sparse_reward and not done:
            return 0
        embedding = self._get_standardized_data(self.trajectory_to_embedding(i_process))
        obs = self.raw_trajectory_current[i_process][-1].unsqueeze(0)

        if self.args.cluster_on == 'state':
            x = obs
        elif self.args.cluster_on == 'trajectory_embedding':
            x = embedding
        else:
            raise ValueError

        z = self.task[i_process]

        if self.args.reward == 'z|w':
            numerator, denominator = 0, 0
            for i, z_ in enumerate(self.valid_components):
                density = multivariate_normal.pdf(x=x,
                                                  mean=self.generative_model.means_[z_],
                                                  cov=self.generative_model.covariances_[z_])
                denominator += density * self.p_z[i]
                if z == z_:
                    numerator = density * self.p_z[i]
            if numerator == 0:  # underflow
                r = -100
            else:
                r = np.log(numerator) - np.log(denominator)
            # from mixture.gaussian_mixture import _estimate_log_gaussian_prob
            # log_gauss = _estimate_log_gaussian_prob(x,
            #                                         self.generative_model.means_,
            #                                         self.generative_model.precisions_cholesky_,
            #                                         self.generative_model.covariance_type)
            #
            #
            #
            # ipdb.set_trace()
        elif self.args.reward == 'w|z':
            r = multivariate_normal.logpdf(x=x,
                                           mean=self.generative_model.means_[z],
                                           cov=self.generative_model.covariances_[z])
        elif self.args.reward == 'l2':
            diff = x - torch.from_numpy(self.context[i_process])
            if self.args.obs == 'pos_speed':
                diff = diff * torch.Tensor([1, 1, 10])
            r = -torch.norm(diff)
        else:
            raise ValueError

        return r

    # def _calculate_reward_parallelized

    def _save_trajectory(self, i_process):
        self.raw_trajectory_embeddings.append(self.trajectory_to_embedding(i_process))
        self.raw_trajectories.append(self.raw_trajectory_current[i_process])
        self.history.save_episode(self.raw_trajectory_current[i_process], self.task[i_process])

    def _process_obs(self, raw_obs):
        if self.args.obs == 'pos_speed':
            speed = []
            for i in range(self.args.num_processes):
                speed.append(torch.norm((raw_obs[i] - self.raw_trajectory_current[i][-1][:raw_obs.shape[1]]).unsqueeze(0)).unsqueeze(0))
            return torch.cat([raw_obs, torch.stack(speed, dim=0)], dim=1)
        elif self.args.obs == 'pos':
            return raw_obs
        else:
            raise ValueError

    def trajectory_to_embedding(self, i_process=0):
        if self.args.trajectory_embedding_type == 'avg':
            trajectory_embedding = torch.mean(self.raw_trajectory_current[i_process], dim=0).unsqueeze(0)
        elif self.args.trajectory_embedding_type == 'final':
            trajectory_embedding = self.raw_trajectory_current[i_process][-1].unsqueeze(0)
        else:
            raise NotImplementedError
        return trajectory_embedding

    def _get_standardized_data(self, data):
        if self.args.standardize_data:
            data = self.standardizer.transform(data)
        return data



