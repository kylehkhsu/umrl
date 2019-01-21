from vae import VAE
import numpy as np
import torch
from utils.misc import guard_against_underflow
import pickle
import os
import ipdb

class History:
    def __init__(self, args):
        self.args = args
        self.episodes = dict(pre_update=[], post_update=[])
        # pre_update or post_update -> fit counter -> task

    def dump(self):
        filename = os.path.join(self.args.log_dir, 'history.pkl')
        pickle.dump(self.episodes, open(filename, 'wb'))


class UnsupervisedRewarder:
    def __init__(self, args, obs_raw_shape):
        self.args = args
        self.obs_raw_shape = obs_raw_shape
        self.tasks = None
        self.latents = None
        self.trajectory_current = None
        self.rewards_current = None
        self.fit_counter = 0
        self.history = History(args)
        self.clusterer = VAE(args, obs_raw_shape[0])
        self.fitted = False

    def save_episodes(self, episodes, is_pre_update):
        key = 'pre_update' if is_pre_update else 'post_update'
        fit_counter_to_task_to_episodes = self.history.episodes[key]

        if len(fit_counter_to_task_to_episodes) == self.fit_counter:
            fit_counter_to_task_to_episodes.append([])

        fit_counter_to_task_to_episodes[self.fit_counter].append(episodes)

    def reset(self):
        self.trajectory_current = []
        self.rewards_current = []

    def append(self, obs):
        self.trajectory_current.append(obs)

    def sample_tasks(self, num_tasks):
        # MAML interface wants tasks as list of dictionaries
        if self.args.clusterer == 'vae':
            latents = np.random.randn(num_tasks, self.args.vae_latent_size).astype(np.float32)
            tasks = [dict(latent=latent) for latent in latents]
        else:
            raise ValueError
        return tasks

    def set_tasks(self, tasks):
        self.tasks = tasks
        self.latents = torch.stack([torch.from_numpy(task['latent']) for task in tasks])

    def _get_fitting_data(self):
        # to get balanced/optimized data across tasks/fitting iterations
        fit_counter_to_task_to_episodes = self.history.episodes['post_update']

        if self.args.subsample_strategy == 'random':
            trajs = [episodes.observations.cpu().transpose(0, 1) for task_to_episodes in fit_counter_to_task_to_episodes
                     for episodes in task_to_episodes]

            raise NotImplementedError
        elif self.args.subsample_strategy == 'last-random':
            # take only trajectories sampled in the last meta-update
            fit_counter_to_task_to_episodes = [task_to_episodes[-self.args.meta_batch_size:] for task_to_episodes in fit_counter_to_task_to_episodes]

            trajs = [episodes.observations.cpu().transpose(0, 1) for task_to_episodes in fit_counter_to_task_to_episodes
                     for episodes in task_to_episodes]
            trajs = torch.cat(trajs, dim=0)

            num_samples = min(self.args.subsample_num, len(trajs))
            indices = torch.from_numpy(np.random.choice(len(trajs), num_samples, replace=False))
            trajs = trajs.index_select(dim=0, index=indices)
        else:
            raise ValueError

        return trajs

    def fit(self):
        trajs = self._get_fitting_data()
        self.clusterer.to(self.args.device)
        self.clusterer.fit(trajs, iteration=self.fit_counter)
        self.fitted = True
        self.fit_counter += 1
        self.history.dump()

    def calculate_reward(self, obs, actions):
        reward_info = dict()
        if not self.fitted:
            reward = torch.zeros(self.args.num_processes)
            reward_info['log_marginal'] = torch.zeros(self.args.num_processes)
            reward_info['lambda_log_s_given_z'] = torch.zeros(self.args.num_processes)
            return reward, reward_info

        if self.args.reward == 's_given_z':
            if self.args.clusterer == 'vae':
                z = self.latents
                traj = torch.stack(self.trajectory_current, dim=1)

                log_s_given_z = self.clusterer.log_s_given_z(s=obs, z=z)
                log_marginal = self.clusterer.log_marginal(s=obs, traj=traj)

                log_s_given_z = guard_against_underflow(log_s_given_z)
                log_marginal = guard_against_underflow(log_marginal)

                reward = -log_marginal + self.args.conditional_coef * log_s_given_z
                reward_info['log_marginal'] = log_marginal
                reward_info['lambda_log_s_given_z'] = self.args.conditional_coef * log_s_given_z
            else:
                raise ValueError
        else:
            raise ValueError

        if isinstance(reward, np.ndarray):
            reward = torch.from_numpy(reward)

        self.rewards_current.append(reward)
        if self.args.cumulative_reward:
            rewards = torch.stack(self.rewards_current, dim=0)
            reward = rewards.mean(dim=0)
        return reward, reward_info
