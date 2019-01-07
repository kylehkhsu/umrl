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
from a2c_ppo_acktr.envs import make_vec_envs
from abc import ABC, abstractmethod


class History(object):
    def __init__(self, args=None):
        self.args = args
        self.episodes = []
        self.generative_model = None
        self.standardizer = None
        self.filename = os.path.join(self.args.log_dir, 'history.pkl')
        self.all = []

    def new(self):
        assert len(self.episodes) > 0
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


class MultiTaskEnvInterface(ABC):
    def __init__(self, args, mode='train'):
        if mode == 'val':
            args.cuda = False
            args.num_processes = 1
        #     self.num_processes = 1
        # else:
        #     cuda = args.cuda
        #     self.num_processes = args.num_processes

        if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

        device = torch.device('cuda:0' if args.cuda else 'cpu')
        env_id = args.env_name

        self.envs = make_vec_envs(env_name=env_id,
                                  seed=args.seed,
                                  num_processes=args.num_processes,
                                  gamma=args.gamma,
                                  log_dir=args.log_dir,
                                  add_timestep=False,
                                  device=device,
                                  allow_early_resets=True)

        if args.obs == 'raw':
            self.obs_shape = self.envs.observation_space.shape
        else:
            raise ValueError

        self.action_space = self.envs.action_space

        self.tasks = []
        self.trajectories_pre = []
        self.trajectories_post = []
        # self.trajectory_embeddings_pre = []
        # self.trajectory_embeddings_post = []

        self.trajectory_current = [torch.zeros(size=[1, self.obs_shape[0]]) for i in range(args.num_processes)]
        self.task_current = [None for i in range(args.num_processes)]

        self.episode_counter = [0 for i in range(args.num_processes)]
        self.step_counter = [0 for i in range(args.num_processes)]
        self.args = args

    def _reset_one(self, i_process, obs_raw, trial_done):
        self.trajectory_current[i_process] = obs_raw[i_process].unsqueeze(0)
        if trial_done[i_process]:
            self._sample_task_one(i_process)
            self.episode_counter[i_process] = 0
        self.step_counter[i_process] = 0

    def reset(self, trial_done=None):
        if trial_done is None:
            trial_done = torch.ones(self.args.num_processes)
        obs_raw = self.envs.reset()
        for i_process in range(self.args.num_processes):
            self._reset_one(i_process, obs_raw, trial_done)
        return self._get_obs_reset(obs_raw)

    def _get_obs_reset(self, obs_raw, obs_act=None, obs_rew=None, obs_flag=None):
        if obs_act is None:
            obs_act = torch.zeros(self.args.num_processes, self.envs.action_space.shape[0])
        if obs_rew is None:
            obs_rew = torch.zeros(self.args.num_processes, 1)
        if obs_flag is None:
            obs_flag = 2 * torch.ones(self.args.num_processes, 1)  # TODO: 0 or 2?
        return obs_raw, obs_act, obs_rew, obs_flag

    def step(self, action):
        env_step_start = time.time()
        obs_raw, _, _done, _ = self.envs.step(action)
        env_step_time = time.time() - env_step_start
        assert not any(_done), 'environments should not reset on their own'

        episode_done = torch.zeros(self.args.num_processes)
        trial_done = torch.zeros(self.args.num_processes)

        for i_process in range(self.args.num_processes):
            self.step_counter[i_process] += 1
            episode_done_ = int(self.step_counter[i_process] == self.args.episode_length)
            if episode_done_:
                self.episode_counter[i_process] += 1
                if self.episode_counter[i_process] == 1:
                    save_to = 'pre'
                else:
                    save_to = 'post'
                self._save_episode(i_process, save_to)
            else:
                self._append_to_trajectory_one(i_process, obs_raw)
            trial_done_ = int(self.episode_counter[i_process] == self.args.trial_length)
            episode_done[i_process] = episode_done_
            trial_done[i_process] = trial_done_
        reward_start = time.time()
        reward = self._calculate_reward(obs_raw, action, episode_done)
        reward_time = time.time() - reward_start
        reward = reward.unsqueeze(1)

        flag = torch.FloatTensor([[1.0] if (episode_done_ and not trial_done_) else [0.0]
                                  for (episode_done_, trial_done_) in zip(episode_done, trial_done)])
        done = dict(episode=episode_done, trial=trial_done)

        assert all(episode_done) or not any(episode_done)
        assert all(trial_done) or not any(trial_done)
        if all(episode_done):
            obs_raw_reset, obs_act_reset, obs_rew_reset, obs_flag_reset = self.reset(trial_done)
            if all(trial_done):
                obs = obs_raw_reset, obs_act_reset, obs_rew_reset, obs_flag_reset
            elif not any(trial_done):
                obs = obs_raw_reset, action, reward, flag
            else:
                raise ValueError
        else:
            obs = obs_raw, action, reward, flag

        info = dict(reward_time=reward_time, env_step_time=env_step_time)

        return obs, reward, done, info

    @abstractmethod
    def _calculate_reward(self, obs_raw, action, done):
        pass

    @abstractmethod
    def _sample_task_one(self, i_process):
        pass

    def _save_episode(self, i_process, save_to):
        assert save_to in ['pre', 'post']
        if save_to == 'pre':
            trajectories = self.trajectories_pre
        else:
            trajectories = self.trajectories_post
        trajectories.append(self.trajectory_current[i_process])

        # self.raw_trajectory_embeddings.append(self.trajectory_to_embedding(i_process))
        # self.history.save_episode(self.raw_trajectory_current[i_process], self.task_current[i_process])
        # self.tasks.append(self.task_current[i_process])

    def _append_to_trajectory_one(self, i_process, obs_raw):
        assert self.trajectory_current[i_process] is not None
        self.trajectory_current[i_process] = torch.cat(
            (self.trajectory_current[i_process], obs_raw[i_process].unsqueeze(0)), dim=0
        )


class SupervisedRewarder(MultiTaskEnvInterface):
    def __init__(self, args, **kwargs):
        super(SupervisedRewarder, self).__init__(args, **kwargs)

    def _calculate_reward(self, obs_raw, action, done):
        if 'Point2D' in self.args.env_name:
            goal = torch.Tensor(self.task_current)
            assert goal.shape == obs_raw.shape
            distance = torch.norm(goal - obs_raw.cpu(), dim=-1)
            dense_reward = -distance
            success_reward = (distance < 2).float()
            reward = self.args.dense_coef * dense_reward + self.args.success_coef * success_reward
        elif 'HalfCheetah' in self.args.env_name:

            reward_ctrl = - 0.1 * (action.cpu() ** 2).sum(dim=-1).unsqueeze(1)

            if self.args.task_type == 'goal':
                goal = torch.Tensor(self.task_current)
                vel = obs_raw[:, 8].unsqueeze(1)
                assert goal.shape == vel.shape
                distance = torch.norm(goal - vel.cpu(), dim=-1)
                success_reward = (distance < 1).float()
                squared_distance = distance ** 2
                dense_reward = -squared_distance
                reward = self.args.dense_coef * dense_reward + \
                         self.args.success_coef * success_reward + reward_ctrl
            elif self.args.task_type == 'direction':
                direction = torch.Tensor(self.task_current)
                vel = obs_raw[:, 8].unsqueeze(1).cpu()
                dense_reward = direction * vel
                reward = self.args.dense_coef * dense_reward + reward_ctrl
                reward = reward.squeeze(1)

        return reward

    def _sample_task_one(self, i_process):
        # rand = 2 * (np.random.random_sample((2,)) - 0.5)    # \in [-1, 1]
        # goal = self.envs.observation_space.spaces['state_observation'].sample()
        # goal = np.array([10, 10])

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
        self.task_current[i_process] = goal


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
            np.zeros(shape=(1, obs_shape[0]), dtype=np.float32) for i in range(args.num_processes)
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
        else:
            raise ValueError

        self.valid_components = np.array([0])
        self.p_z = np.array([1])

        self.standardizer = StandardScaler()
        self.logger = logger
        self.discriminator_loss = 0

    def fit_generative_model(self):
        # if self.clustering_counter > 0:
            # ipdb.set_trace()
        self.history.new()

        if self.args.cluster_on == 'trajectory_embedding':
            data_list = self.raw_trajectory_embeddings
            assert self.generative_model.group is None
        elif self.args.cluster_on == 'state':
            data_list = self.raw_trajectories
        else:
            raise ValueError
        tasks = self.tasks

        if self.args.cluster_subsample_strategy == 'last':
            data_list = data_list[-self.args.cluster_subsample_num:]
            tasks = tasks[-self.args.cluster_subsample_num:]
        elif self.args.cluster_subsample_strategy == 'random' and len(data_list) > self.args.cluster_subsample_num:
            indices = np.random.choice(len(data_list), self.args.cluster_subsample_num, replace=False)
            data_list = [data_list[index] for index in indices]
            tasks = [tasks[index] for index in indices]
        # else keep it all

        data = torch.cat(data_list, dim=0)

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
            self.clustering_counter += 1
            self.logger.log('clustering_counter: {}'.format(self.clustering_counter))
            # self.history.save_generative_model(self.generative_model, self.standardizer)
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

            if self.args.log_EM:
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

        if self.args.log_EM:
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
                if self.clustering_counter != 0:
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
            if self.clustering_counter == 0:
                reward = np.zeros(X.shape[0], dtype=np.float32)
            else:
                if self.args.reward == 'z|w':
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
                elif self.args.reward == 'w|z':
                    log_gauss = _estimate_log_gaussian_prob(X,
                                                            self.generative_model.means_,
                                                            self.generative_model.precisions_cholesky_,
                                                            self.generative_model.covariance_type)
                    gauss = np.exp(log_gauss[:, self.valid_components])
                    joint = gauss * self.p_z[None, :]
                    joint[joint < 1e-300] = 1e-300
                    marginal = joint.sum(axis=1)
                    conditional = joint[np.arange(self.args.num_processes), self.task_index_current]
                    reward = - np.log(marginal) + self.args.conditional_coef * np.log(conditional)
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
        if self.clustering_counter == 0:
            tasks = torch.randint(low=0, high=self.args.max_components,
                                  size=(num_trajectories,))
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

        self.valid_components = np.arange(self.args.max_components)
        self.p_z = np.ones(self.args.max_components) / self.args.max_components


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


