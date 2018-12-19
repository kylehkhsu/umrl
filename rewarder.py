import numpy as np
import torch
import ipdb
from sklearn.mixture import BayesianGaussianMixture
from scipy.stats import multivariate_normal
from copy import deepcopy
import pickle
import os


class History(object):
    def __init__(self, args):
        self.args = args
        self.episodes = []
        self.generative_model = None
        self.filename = os.path.join(self.args.log_dir, 'history.pkl')
        self.all = []

    def new(self):
        if self.generative_model is not None and len(self.episodes) > 0:
            self.all.append(dict(generative_model=self.generative_model,
                                 episodes=self.episodes))
        self.generative_model = None
        self.episodes = []

    def save_generative_model(self, generative_model):
        assert self.generative_model is None
        self.generative_model = deepcopy(generative_model)

    def save_episode(self, trajectory, task):
        self.episodes.append((trajectory, task))

    def dump(self):
        pickle.dump(self, open(self.filename, 'wb'))

    def load(self):
        return pickle.load(open(self.filename, 'rb'))


class Rewarder(object):

    def __init__(self,
                 args,
                 embedding_shape,
                 starting_trajectories=[]):
        self.args = args
        self.num_processes = args.num_processes
        self.embedding_shape = embedding_shape
        self.episode_length = args.episode_length
        self.trajectory_embedding = args.trajectory_embedding
        self.trajectories = starting_trajectories
        self.history = History(args)
        self.clustering_counter = 0
        # self.encoding_function = lambda x: x
        self.current_trajectory = [torch.zeros(size=embedding_shape).unsqueeze(0) for i in range(args.num_processes)]
        self.context = [
            np.zeros(shape=embedding_shape, dtype=np.float32) for i in range(args.num_processes)
        ]
        self.task = [
            -1 for i in range(args.num_processes)
        ]

        self.step_counter = [0 for i in range(args.num_processes)]
        self.dpgmm = BayesianGaussianMixture(n_components=50,
                                             covariance_type='full',
                                             verbose=0,
                                             max_iter=10000,
                                             weight_concentration_prior_type='dirichlet_process')
        self.weights = None
    # def encode_trajectory(self, trajectory):
    #     return self.encoding_function(trajectory)

    def fit_generative_model(self):
        self.history.new()

        trajectory_embeddings = torch.cat(self.trajectories, dim=0)
        self.trajectories = []  #TODO: change, maybe
        self.dpgmm.fit(trajectory_embeddings)
        self.weights = np.compress(self.dpgmm.weights_ >= 1e-8,
                                   self.dpgmm.weights_)
        self.weights = self.weights / sum(self.weights)

        self.history.save_generative_model(self.dpgmm)

    def skew_generative_model(self):
        pass

    def _insert_one(self, i_process, embedding):
        assert self.current_trajectory[i_process] is not None
        self.current_trajectory[i_process] = torch.cat(
            (self.current_trajectory[i_process], embedding[i_process].unsqueeze(0)), dim=0
        )

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
                    z = np.random.choice(len(self.weights), size=1, replace=False)[0]   # U(z)
                else:
                    z = np.random.choice(len(self.weights), size=1, replace=False, p=self.weights)[0]
                self.task[i_process] = z
                self.context[i_process] = self.dpgmm.means_[z]

    def _reset_one(self, i_process, raw_obs):
        self.current_trajectory[i_process] = torch.zeros(size=self.embedding_shape).unsqueeze(0)
        self.current_trajectory[i_process][0][:raw_obs.shape[1]] = raw_obs[i_process]
        self._sample_task_one(i_process)
        self.step_counter[i_process] = 0

    def reset(self, raw_obs):
        for i in range(self.num_processes):
            self._reset_one(i, raw_obs)
        return torch.cat(
            (torch.cat(self.current_trajectory, dim=0), torch.from_numpy(np.array(self.context, dtype=np.float32))), dim=1
        )

    def step(self, raw_obs, done, infos):
        embedding = self._process_obs(raw_obs)
        reward = []
        for i in range(self.num_processes):
            self.step_counter[i] += 1
            done_ = self.step_counter[i] == self.episode_length
            if done_:
                self._save_trajectory(i)
                # self._reset_one(i, embedding)
                # ipdb.set_trace()
            self._insert_one(i, embedding)
            done[i] = done_
            reward.append(self._reward(i, done_))
        reward = torch.from_numpy(np.array(reward, dtype=np.float32)).unsqueeze(1)
        context = torch.from_numpy(np.array(self.context, dtype=np.float32))
        return torch.cat([embedding, context], dim=1), reward, done, infos

    def _reward(self, i_process, done):
        if self.args.sparse_reward and not done:
            return 0

        traj_embedding = self.get_traj_embedding(i_process)
        if self.args.reward == 'l2':
            difference = traj_embedding - torch.from_numpy(self.context[i_process])
            scaled_difference = difference * torch.Tensor([1, 1, 10])
            d = torch.norm(scaled_difference)
            r = -d
        elif self.args.reward == 'w|z':
            z = self.task[i_process]
            r = multivariate_normal.logpdf(x=traj_embedding,
                                           mean=self.dpgmm.means_[z],
                                           cov=self.dpgmm.covariances_[z])
        elif self.args.reward == 'z|w':
            w = traj_embedding
            z = self.task[i_process]

            denominator = 0
            for i in range(len(self.weights)):
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
        self.trajectories.append(self.get_traj_embedding(i_process).unsqueeze(0))
        self.history.save_episode(self.current_trajectory[i_process], self.task[i_process])

    def _process_obs(self, raw_obs):
        speed = []
        for i in range(self.num_processes):
            speed.append(torch.norm((raw_obs[i] - self.current_trajectory[i][-1][:raw_obs.shape[1]]).unsqueeze(0)).unsqueeze(0))
        return torch.cat([raw_obs, torch.stack(speed, dim=0)], dim=1)

    def get_traj_embedding(self, i_process=0):
        if self.trajectory_embedding == 'avg':
            return torch.mean(self.current_trajectory[i_process], dim=0)
        elif self.trajectory_embedding == 'final':
            return self.current_trajectory[i_process][-1]
        else:
            raise NotImplementedError


if __name__ == '__main__':
    pass
