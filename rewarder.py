import numpy as np
import torch
import ipdb
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import StandardScaler
from scipy.stats import multivariate_normal
from copy import deepcopy
import pickle
import os


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


class Rewarder(object):

    def __init__(self,
                 args,
                 obs_shape):
        self.args = args
        self.obs_shape = obs_shape
        self.raw_trajectory_embeddings = []
        self.history = History(args)
        self.clustering_counter = 0
        # self.encoding_function = lambda x: x
        self.raw_trajectory_current = [torch.zeros(size=obs_shape).unsqueeze(0) for i in range(args.num_processes)]
        self.context = [
            np.zeros(shape=obs_shape, dtype=np.float32) for i in range(args.num_processes)
        ]
        self.task = [
            -1 for i in range(args.num_processes)
        ]

        self.step_counter = [0 for i in range(args.num_processes)]
        self.dpgmm = BayesianGaussianMixture(n_components=self.args.max_components,
                                             covariance_type='full',
                                             verbose=1,
                                             verbose_interval=1000,
                                             max_iter=5000,
                                             n_init=2,
                                             weight_concentration_prior_type='dirichlet_process')
        self.valid_components = None
        self.standardizer = StandardScaler()
    # def encode_trajectory(self, trajectory):
    #     return self.encoding_function(trajectory)

    def fit_generative_model(self, trajectories=None):
        self.history.new()
        if trajectories is not None:    # first fitting
            assert self.clustering_counter == 0
            trajectory_embeddings = torch.cat(trajectories, dim=0)
        else:
            trajectory_embeddings = torch.cat(self.raw_trajectory_embeddings, dim=0)

        if not self.args.fit_on_entire_history:
            self.raw_trajectory_embeddings = []

        if self.args.standardize_embeddings:
            self.standardizer.fit(trajectory_embeddings)

        trajectory_embeddings = self._get_standardized_trajectory_embedding(trajectory_embeddings)

        self.dpgmm.fit(trajectory_embeddings)

        components = np.argwhere(self.dpgmm.weights_ >= self.args.component_weight_threshold).reshape([-1])
        valid_components = []
        for i, component in enumerate(components):
            if i == 0:
                valid_components.append(component)
                continue
            current_mean = self.dpgmm.means_[component]
            prev_means = self.dpgmm.means_[components[:i]]
            l_2 = min(np.linalg.norm(prev_means - current_mean, ord=2, axis=1))
            l_inf = min(np.linalg.norm(prev_means - current_mean, ord=np.inf, axis=1))
            if l_2 >= self.args.component_constraint_l_2 or l_inf >= self.args.component_constraint_l_inf:
                valid_components.append(component)
        self.valid_components = np.array(valid_components)

        print('raw means of valid components:\n{}'.format(self._get_raw_means(self.valid_components)))
        print('standardized means of valid components:\n{}'.format(self.dpgmm.means_[self.valid_components]))

        self.clustering_counter += 1
        self.history.save_generative_model(self.dpgmm, self.standardizer)

    def skew_generative_model(self):
        pass

    def _insert_one(self, i_process, embedding):
        assert self.raw_trajectory_current[i_process] is not None
        self.raw_trajectory_current[i_process] = torch.cat(
            (self.raw_trajectory_current[i_process], embedding[i_process].unsqueeze(0)), dim=0
        )

    def _get_raw_means(self, i):
        mean = self.dpgmm.means_[i]
        if self.args.standardize_embeddings:
            mean = self.standardizer.inverse_transform(mean, copy=None)
        return mean

    def _sample_task_one(self, i_process):
        if self.args.context == 'goal':
            position = np.random.uniform(low=-5, high=5, size=2)
            speed = np.random.uniform(low=0.1, high=1, size=1)
            self.context[i_process] = np.concatenate((position, speed)).astype(np.float32)
        elif self.args.context == 'cluster_mean':
            if self.args.skew:
                raise NotImplementedError
            else:
                if self.args.uniform_cluster_categorical:
                    z = np.random.choice(self.valid_components, size=1, replace=False)[0]   # U(z)
                else:
                    weights = self.dpgmm.weights_[self.valid_components]
                    weights = weights / sum(weights)
                    z = np.random.choice(self.valid_components, size=1, replace=False, p=weights)[0]
                self.task[i_process] = z
                self.context[i_process] = self._get_raw_means(z)

    def _reset_one(self, i_process, raw_obs):
        self.raw_trajectory_current[i_process] = torch.zeros(size=self.obs_shape).unsqueeze(0)
        self.raw_trajectory_current[i_process][0][:raw_obs.shape[1]] = raw_obs[i_process]
        self._sample_task_one(i_process)
        self.step_counter[i_process] = 0

    def reset(self, raw_obs):
        for i in range(self.args.num_processes):
            self._reset_one(i, raw_obs)
        return torch.cat(
            (torch.cat(self.raw_trajectory_current, dim=0), torch.from_numpy(np.array(self.context, dtype=np.float32))), dim=1
        )

    def step(self, raw_obs, done, infos):
        embedding = self._process_obs(raw_obs)
        reward = []
        for i in range(self.args.num_processes):
            self.step_counter[i] += 1
            done_ = self.step_counter[i] == self.args.episode_length
            if done_:
                self._save_trajectory(i)
            self._insert_one(i, embedding)
            done[i] = done_
            reward.append(self._reward(i, done_))
        reward = torch.from_numpy(np.array(reward, dtype=np.float32)).unsqueeze(1)
        context = torch.from_numpy(np.array(self.context, dtype=np.float32))
        return torch.cat([embedding, context], dim=1), reward, done, infos

    def _reward(self, i_process, done):
        if self.args.sparse_reward and not done:
            return 0

        trajectory_embedding = self._get_standardized_trajectory_embedding(self.trajectory_to_embedding(i_process))
        if self.args.reward == 'l2':
            difference = trajectory_embedding - torch.from_numpy(self.context[i_process])
            scaled_difference = difference * torch.Tensor([1, 1, 10])
            d = torch.norm(scaled_difference)
            r = -d
        elif self.args.reward == 'w|z':
            z = self.task[i_process]
            r = multivariate_normal.logpdf(x=trajectory_embedding,
                                           mean=self.dpgmm.means_[z],
                                           cov=self.dpgmm.covariances_[z])
        elif self.args.reward == 'z|w':
            w = trajectory_embedding
            z = self.task[i_process]

            denominator = 0
            for i in range(len(self.valid_components)):
                x = multivariate_normal.pdf(x=w,
                                            mean=self.dpgmm.means_[i],
                                            cov=self.dpgmm.covariances_[i])
                denominator += x
                if z == i:
                    numerator = x
            if denominator == 0 or numerator == 0:
                r = -1000
            else:
                r = np.log(numerator) - np.log(denominator)
        else:
            raise ValueError

        return r

    def _save_trajectory(self, i_process):
        self.raw_trajectory_embeddings.append(self.trajectory_to_embedding(i_process).unsqueeze(0))
        self.history.save_episode(self.raw_trajectory_current[i_process], self.task[i_process])

    def _process_obs(self, raw_obs):
        speed = []
        for i in range(self.args.num_processes):
            speed.append(torch.norm((raw_obs[i] - self.raw_trajectory_current[i][-1][:raw_obs.shape[1]]).unsqueeze(0)).unsqueeze(0))
        return torch.cat([raw_obs, torch.stack(speed, dim=0)], dim=1)

    def trajectory_to_embedding(self, i_process=0):
        if self.args.trajectory_embedding_type == 'avg':
            trajectory_embedding = torch.mean(self.raw_trajectory_current[i_process], dim=0)
        elif self.args.trajectory_embedding_type == 'final':
            trajectory_embedding = self.raw_trajectory_current[i_process][-1]
        else:
            raise NotImplementedError
        return trajectory_embedding

    def _get_standardized_trajectory_embedding(self, trajectory_embedding):
        if self.args.standardize_embeddings:
            trajectory_embedding = self.standardizer.transform(trajectory_embedding, copy=None)
        return trajectory_embedding



