import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb
import time
from a2c_ppo_acktr.envs import make_vec_envs
from abc import ABC, abstractmethod
from rewarder import SupervisedRewarder


class MultiTaskEnvInterface(ABC):
    def __init__(self, args, mode='train'):
        if mode == 'val':
            args.cuda = False
            args.num_processes = 1

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
            self.obs_raw_shape = self.envs.observation_space.shape
        else:
            raise ValueError

        self.action_space = self.envs.action_space
        self.tasks = []

        self.trajectory_current = [torch.zeros(size=[1, self.obs_raw_shape[0]]) for i in range(args.num_processes)]
        self.task_current = [None for i in range(args.num_processes)]
        self.step_counter = [0 for i in range(args.num_processes)]
        self.args = args

        if self.args.rewarder == 'supervised':
            self.rewarder = SupervisedRewarder(args)
        else:
            raise ValueError

    def _append_to_trajectory_one(self, i_process, obs_raw):
        assert self.trajectory_current[i_process] is not None
        self.trajectory_current[i_process] = torch.cat(
            (self.trajectory_current[i_process], obs_raw[i_process].unsqueeze(0)), dim=0)

    @abstractmethod
    def _reset_one(self, i_process, obs_raw, **kwargs):
        pass

    @abstractmethod
    def reset(self, **kwargs):
        pass

    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def _save_episode(self, i_process, **kwargs):
        pass

    @abstractmethod
    def set_task_one(self, task, i_process):
        pass

    def _calculate_reward(self, task, obs_raw, action, **kwargs):
        return self.rewarder._calculate_reward(task, obs_raw, action, **kwargs)

    def _sample_task_one(self):
        return self.rewarder._sample_task_one()


class RL2EnvInterface(MultiTaskEnvInterface):
    def __init__(self, args, **kwargs):
        super(RL2EnvInterface, self).__init__(args, **kwargs)
        self.obs_shape = self.obs_raw_shape
        self.trajectories_pre = []
        self.trajectories_post = []
        # self.trajectory_embeddings_pre = []
        # self.trajectory_embeddings_post = []
        self.episode_counter = [0 for i in range(args.num_processes)]

    def _reset_one(self, i_process, obs_raw, **kwargs):
        self.trajectory_current[i_process] = obs_raw[i_process].unsqueeze(0)
        if kwargs['trial_done'][i_process]:
            self.set_task_one(self._sample_task_one(), i_process)
            self.episode_counter[i_process] = 0
        self.step_counter[i_process] = 0

    def reset(self, trial_done=None):
        if trial_done is None:
            trial_done = torch.ones(self.args.num_processes)
        obs_raw = self.envs.reset()
        for i_process in range(self.args.num_processes):
            self._reset_one(i_process, obs_raw, trial_done=trial_done)
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
        obs_raw, _, done_raw, info_raw = self.envs.step(action)
        env_step_time = time.time() - env_step_start
        assert not any(done_raw), 'environments should not reset on their own'

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
        reward = self._calculate_reward(self.task_current, obs_raw, action, env_info=info_raw)
        reward_time = time.time() - reward_start
        reward = reward.unsqueeze(1)

        flag = torch.FloatTensor([[1.0] if (episode_done_ and not trial_done_) else [0.0]
                                  for (episode_done_, trial_done_) in zip(episode_done, trial_done)])
        done = dict(episode=episode_done, trial=trial_done)

        assert all(episode_done) or not any(episode_done)
        assert all(trial_done) or not any(trial_done)
        if all(episode_done):
            obs_raw_reset, obs_act_reset, obs_rew_reset, obs_flag_reset = self.reset(trial_done=trial_done)
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

    def _save_episode(self, i_process, save_to=None):
        assert save_to in ['pre', 'post']
        if save_to == 'pre':
            trajectories = self.trajectories_pre
        else:
            trajectories = self.trajectories_post
        trajectories.append(self.trajectory_current[i_process])

        # self.raw_trajectory_embeddings.append(self.trajectory_to_embedding(i_process))
        # self.history.save_episode(self.trajectory_current[i_process], self.task_current[i_process])
        # self.tasks.append(self.task_current[i_process])

    def set_task_one(self, task, i_process=0):
        self.task_current[i_process] = task


class ContextualEnvInterface(MultiTaskEnvInterface):
    def __init__(self, args, **kwargs):
        super(ContextualEnvInterface, self).__init__(args, **kwargs)
        self.context = [
            np.zeros(shape=(1, self.obs_raw_shape[0]), dtype=np.float32) for i in range(args.num_processes)
        ]
        self.obs_shape = (self.obs_raw_shape[0] * 2,)

    def _reset_one(self, i_process, obs_raw, **kwargs):
        self.trajectory_current[i_process] = torch.zeros(size=self.obs_raw_shape).unsqueeze(0)
        self.trajectory_current[i_process][0][:obs_raw.shape[1]] = obs_raw[i_process]
        self.set_task_one(self._sample_task_one(), i_process)
        self.step_counter[i_process] = 0

    def _get_obs(self, obs_raw):
        context = torch.from_numpy(np.concatenate(self.context, axis=0).astype(dtype=np.float32))
        return torch.cat([obs_raw, context], dim=1)

    def reset(self):
        obs_raw = self.envs.reset()
        for i_process in range(self.args.num_processes):
            self._reset_one(i_process, obs_raw)
        obs = self._get_obs(obs_raw)
        return obs

    def step(self, action):
        env_step_start = time.time()
        obs_raw, _, done_raw, info_raw = self.envs.step(action)
        env_step_time = time.time() - env_step_start
        assert not any(done_raw), 'environments should not reset on their own'

        done = torch.zeros(self.args.num_processes)
        for i_process in range(self.args.num_processes):
            self.step_counter[i_process] += 1
            done_ = int(self.step_counter[i_process] == self.args.episode_length)
            if done_:
                self._save_episode(i_process)
            else:
                self._append_to_trajectory_one(i_process, obs_raw)
            done[i_process] = done_
        reward_start = time.time()
        reward = self._calculate_reward(self.task_current, obs_raw, action, env_info=info_raw)
        reward_time = time.time() - reward_start
        reward = reward.unsqueeze(1)

        assert all(done) or not any(done)
        if all(done):
            obs = self.reset()
        else:
            obs = self._get_obs(obs_raw)

        info = dict(reward_time=reward_time, env_step_time=env_step_time)
        return obs, reward, done, info

    def _save_episode(self, i_process, **kwargs):
        pass

    def set_task_one(self, task, i_process=0):
        self.task_current[i_process] = task
        self.context[i_process] = np.expand_dims(task, axis=0)