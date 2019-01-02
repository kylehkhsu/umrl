import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb
# from sklearn.mixture import BayesianGaussianMixture
from mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer
from scipy.stats import multivariate_normal
from copy import deepcopy
import pickle
import os
import time
from gmm import GMM_estep
from mixture.gaussian_mixture import _estimate_log_gaussian_prob
from collections import OrderedDict
from torch.distributions import Categorical


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
        self.tasks = []
        self.task_current = [
            -1 for i in range(args.num_processes)
        ]
        self.task_index_current = np.zeros(args.num_processes, dtype=np.int)

        self.step_counter = [0 for i in range(args.num_processes)]

        group = dict(state=self.args.episode_length,
                     trajectory_embedding=None)[self.args.cluster_on]
        if self.args.clusterer == 'bayesian':
            self.generative_model = \
                BayesianGaussianMixture(n_components=self.args.max_components,
                                        covariance_type='full',
                                        verbose=1,
                                        verbose_interval=100,
                                        max_iter=1000,
                                        n_init=1,
                                        weight_concentration_prior_type='dirichlet_process',
                                        weight_concentration_prior=self.args.weight_concentration_prior,
                                        group=group)
        elif self.args.clusterer == 'gaussian':
            self.generative_model = \
                GaussianMixture(n_components=self.args.max_components,
                                covariance_type='full',
                                verbose=1,
                                verbose_interval=100,
                                max_iter=1000,
                                n_init=1,
                                group=group)
        elif self.args.clusterer == 'discriminator':
            self.generative_model = Discriminator(input_size=self.obs_shape[0],
                                                  output_size=self.args.max_components)

        self.valid_components = None
        self.standardizer = StandardScaler()
        self.p_z = None
        self.logger = logger
        self.discriminator_loss = 0

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
            tasks = []
        else:
            if self.args.cluster_on == 'trajectory_embedding':
                data = torch.cat(self.raw_trajectory_embeddings, dim=0)
                assert self.generative_model.group is None
            elif self.args.cluster_on == 'state':
                if self.args.cluster_subsample_strategy == 'last':
                    data = torch.cat(self.raw_trajectories[-self.args.cluster_subsample_num:], dim=0)
                    tasks = self.tasks[-self.args.cluster_subsample_num:]
                elif self.args.cluster_subsample_strategy == 'random' and len(self.raw_trajectories) > self.args.cluster_subsample_num:
                    indices = np.random.choice(len(self.raw_trajectories), self.args.cluster_subsample_num, replace=False)
                    subset = [self.raw_trajectories[index] for index in indices]
                    data = torch.cat(subset, dim=0)
                    tasks = [self.tasks[index] for index in indices]
                else:
                    data = torch.cat(self.raw_trajectories, dim=0)
                    tasks = self.tasks

        if not self.args.keep_entire_history:
            self.raw_trajectory_embeddings = []
            self.raw_trajectories = []
            self.tasks = []

        if self.args.standardize_data:
            self.standardizer.fit(data)

        data = self._get_standardized_data(data)

        if self.args.clusterer == 'discriminator':
            assert self.args.cluster_on == 'state'
            self._train_discriminator(data, tasks)
            self.valid_components = np.arange(self.args.max_components)
            self.p_z = np.ones(self.args.max_components) / self.args.max_components
            self.logger.log('clustering_counter: {}'.format(self.clustering_counter))
            self.clustering_counter += 1
            self.history.save_generative_model(self.generative_model, self.standardizer)
        else:
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
        log_gauss = _estimate_log_gaussian_prob(data,
                                                self.generative_model.means_,
                                                self.generative_model.precisions_cholesky_,
                                                self.generative_model.covariance_type)
        gauss = np.exp(log_gauss[:, self.valid_components])
        joint = gauss * self.generative_model.weights_[None, self.valid_components]
        joint[joint < 1e-300] = 1e-300
        denominator = np.sum(joint, axis=1)
        log_posterior = np.log(joint) - np.log(denominator)[:, None]
        p_z = np.mean(log_posterior, axis=0)
        p_z = np.exp(p_z)
        p_z /= np.sum(p_z)

        # data = torch.reshape(data, [-1, self.args.episode_length, data.shape[1]])
        # p_z = np.zeros(len(self.valid_components))
        # N = min(10000, data.shape[0])
        # for n in range(N):
        #     for t in range(self.args.episode_length):
        #         x = data[-n][t]
        #         joint = np.zeros(len(self.valid_components))
        #         for i, c in enumerate(self.valid_components):
        #             density = multivariate_normal.pdf(x=x,
        #                                               mean=self.generative_model.means_[c],
        #                                               cov=self.generative_model.covariances_[c])
        #             joint[i] = density * self.generative_model.weights_[c]
        #         # if np.any(np.log(joint) == -np.inf):
        #         #     ipdb.set_trace()
        #         log_joint = np.log(joint) - np.log(np.sum(joint))
        #         p_z += log_joint
        # p_z /= (N * self.args.episode_length)
        # p_z2 = np.exp(p_z)
        # p_z2 /= np.sum(p_z2)
        # ipdb.set_trace()

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
        else:
            z = np.random.choice(self.valid_components, size=1, replace=False, p=self.p_z)[0]
            self.task_current[i_process] = z
            self.task_index_current[i_process] = np.argwhere(self.valid_components == z)[0][0]
            if self.args.context == 'cluster_mean':
                self.context[i_process] = self._get_raw_means(z).astype(np.float32)
            elif self.args.context == 'one_hot':
                context = np.zeros(self.args.max_components, dtype=np.float32)
                context[z] = 1
                self.context[i_process] = context[None, :]
            else:
                raise ValueError

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
        for i in range(self.args.num_processes):
            self.step_counter[i] += 1
            done_ = self.step_counter[i] == self.args.episode_length
            if done_:
                self._save_trajectory(i)
            self._append_to_trajectory_one(i, obs)
            done[i] = done_
        reward_start = time.time()
        reward = self._calculate_reward(done, obs=obs)
        reward_time = time.time() - reward_start

        reward = torch.from_numpy(reward).unsqueeze(1)
        context = torch.from_numpy(np.concatenate(self.context, axis=0))
        return torch.cat([obs, context], dim=1), reward, done, infos, reward_time

    def _calculate_reward(self, done, obs):
        assert (all(done) or not any(done))
        if self.args.sparse_reward and not any(done):
            return 0

        if self.args.cluster_on == 'state':
            X = obs
        elif self.args.cluster_on == 'trajectory_embedding':
            X = torch.cat([self.trajectory_to_embedding(i) for i in range(self.args.num_processes)])

        if self.args.clusterer == 'discriminator':
            with torch.no_grad():
                reward = self.generative_model(obs).log_prob(torch.ones(obs.shape[0]) *
                                                             torch.from_numpy(self.task_index_current.astype(np.float32)))
            reward = reward.numpy()
        else:

            log_gauss = _estimate_log_gaussian_prob(X,
                                                    self.generative_model.means_,
                                                    self.generative_model.precisions_cholesky_,
                                                    self.generative_model.covariance_type)
            gauss = np.exp(log_gauss[:, self.valid_components])
            density = gauss * self.p_z[None, :]
            density[density < 1e-300] = 1e-300
            denominator = np.sum(density, axis=1)
            numerator = density[np.arange(self.args.num_processes), self.task_index_current]
            reward = np.log(numerator) - np.log(denominator)
            reward = reward.astype(np.float32)
        return reward

    def _calculate_reward_old(self, i_process, done):
        if self.args.sparse_reward and not done:
            return 0

        time1 = time.time()
        if self.args.cluster_on == 'state':
            obs = self.raw_trajectory_current[i_process][-1].unsqueeze(0)
            x = obs
        elif self.args.cluster_on == 'trajectory_embedding':
            embedding = self._get_standardized_data(self.trajectory_to_embedding(i_process))
            x = embedding
        else:
            raise ValueError
        print('time1: {}'.format(time.time() - time1))

        z = self.task_current[i_process]

        time2 = time.time()

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
        print('time2: {}'.format(time.time() - time2))

        # time3 = time.time()
        # X = torch.cat([x for i in range(10)], dim=0)
        #
        # log_gauss = _estimate_log_gaussian_prob(X,
        #                                         self.generative_model.means_,
        #                                         self.generative_model.precisions_cholesky_,
        #                                         self.generative_model.covariance_type)
        # print('time3: {}'.format(time.time() - time3))
        return r

    # def _calculate_reward_parallelized

    def _save_trajectory(self, i_process):
        self.raw_trajectory_embeddings.append(self.trajectory_to_embedding(i_process))
        self.raw_trajectories.append(self.raw_trajectory_current[i_process])
        self.history.save_episode(self.raw_trajectory_current[i_process], self.task_current[i_process])
        self.tasks.append(self.task_current[i_process])

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

    def _train_discriminator(self, data, tasks):
        discriminator = self.generative_model
        assert discriminator.__class__.__name__ == 'Discriminator'
        assert data.dim() == 2

        num_trajectories = data.shape[0] // self.args.episode_length
        if tasks == []:
            # tasks = torch.randint(low=0, high=self.args.max_components,
            #                       size=(num_trajectories,))
            return
        else:
            assert len(tasks) == num_trajectories
        labels = []
        for task in tasks:
            labels.append(torch.ones(self.args.episode_length, dtype=torch.long) * task)
        labels = torch.cat(labels, dim=0)
        batch_size = 100
        idx = np.arange(data.shape[0])
        num_batches = data.shape[0] // batch_size
        num_epochs = 10
        loss_function = nn.CrossEntropyLoss()
        for epoch in range(num_epochs):
            loss = 0
            np.random.shuffle(idx)
            for i in range(num_batches):
                idx_batch = idx[i*batch_size : (i+1)*batch_size]
                inputs_batch, labels_batch = data[idx_batch], labels[idx_batch]
                discriminator.optimizer.zero_grad()
                outputs_batch = discriminator(inputs_batch).logits
                loss_batch = loss_function(outputs_batch, labels_batch)
                loss_batch.backward()
                discriminator.optimizer.step()
                loss += loss_batch.item()
            # print('discriminator epoch {}\tloss {}'.format(epoch, loss/data.shape[0]))
        self.discriminator_loss = loss / data.shape[0]


def weight_init(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        module.bias.data.zero_()

import torch.optim as optim
class Discriminator(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes=(64, 64), nonlinearity=F.relu):
        super(Discriminator, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.nonlinearity = nonlinearity
        self.num_layers = len(hidden_sizes) + 1
        layer_sizes = (input_size,) + hidden_sizes + (output_size,)
        for i in range(1, self.num_layers + 1):
            self.add_module('layer{0}'.format(i),
                            nn.Linear(layer_sizes[i - 1], layer_sizes[i]))
        self.apply(weight_init)
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3, betas=(0.9, 0.999))

    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())
        output = input
        for i in range(1, self.num_layers + 1):
            output = F.linear(output,
                              weight=params['layer{0}.weight'.format(i)],
                              bias=params['layer{0}.bias'.format(i)])
            if i != self.num_layers:
                output = self.nonlinearity(output)
        return Categorical(logits=output)


