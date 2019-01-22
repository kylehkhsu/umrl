import gym
import torch
import multiprocessing as mp
import ipdb
from maml_rl.rewarder import UnsupervisedRewarder, SupervisedRewarder

from maml_rl.envs.subproc_vec_env import SubprocVecEnv
from maml_rl.episode import BatchEpisodes

from a2c_ppo_acktr.envs import make_vec_envs
import time

def make_env(env_name):
    def _make_env():
        return gym.make(env_name)
    return _make_env

class BatchSampler(object):
    def __init__(self, env_name, batch_size, num_workers=mp.cpu_count() - 1):
        self.env_name = env_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.queue = mp.Queue()
        self.envs = SubprocVecEnv([make_env(env_name) for _ in range(num_workers)],
            queue=self.queue)
        self._env = gym.make(env_name)

    def sample(self, policy, params=None, gamma=0.95, device='cpu'):
        episodes = BatchEpisodes(batch_size=self.batch_size, gamma=gamma, device=device)
        for i in range(self.batch_size):
            self.queue.put(i)
        for _ in range(self.num_workers):
            self.queue.put(None)
        observations, batch_ids = self.envs.reset()
        dones = [False]
        while (not all(dones)) or (not self.queue.empty()):
            with torch.no_grad():
                observations_tensor = torch.from_numpy(observations).to(device=device)
                actions_tensor = policy(observations_tensor, params=params).sample()
                actions = actions_tensor.cpu().numpy()
            new_observations, rewards, dones, new_batch_ids, _ = self.envs.step(actions)
            episodes.append(observations, actions, rewards, batch_ids)
            observations, batch_ids = new_observations, new_batch_ids
        ipdb.set_trace()
        return episodes

    def reset_task(self, task):
        tasks = [task for _ in range(self.num_workers)]
        reset = self.envs.reset_task(tasks)
        return all(reset)

    def sample_tasks(self, num_tasks):
        tasks = self._env.unwrapped.sample_tasks(num_tasks)
        return tasks


class BatchSamplerMultiworld(object):
    def __init__(self, args, val=False):
        if not val:
            self.batch_size = args.fast_batch_size
            self.num_processes = args.num_processes
        else:
            self.batch_size = args.fast_batch_size_val
            self.num_processes = 1

        self.envs = make_vec_envs(env_name=args.env_name,
                                  seed=args.seed,
                                  num_processes=self.num_processes,
                                  gamma=None,
                                  log_dir=args.log_dir,
                                  add_timestep=False,
                                  device=args.device,
                                  allow_early_resets=True)

        if val or args.rewarder == 'supervised':
            self.rewarder = SupervisedRewarder(args)
        else:
            obs_raw_shape = self.envs.observation_space.shape
            self.rewarder = UnsupervisedRewarder(args, obs_raw_shape=obs_raw_shape)

        self.args = args
        self.logging_info = dict(pre_update=[], post_update=[])

    def sample(self, policy, params=None, gamma=0.95, device=None):
        if device is None:
            device = self.args.device

        episodes = BatchEpisodes(batch_size=self.batch_size, gamma=gamma, device=device)

        assert self.batch_size % self.num_processes == 0, "for looping to work"
        episodes_per_process = self.batch_size // self.num_processes

        for i_episode_per_process in range(episodes_per_process):
            batch_ids = [(i_episode_per_process * self.num_processes) + p for p in range(self.num_processes)]
            obs_tensor = self.envs.reset()
            self.rewarder.reset()
            self.rewarder.append(obs_tensor)    # one extra append at end of for loop, but that's okay

            for t in range(self.args.episode_length):
                with torch.no_grad():
                    actions_tensor = policy(obs_tensor.to(device), params=params).sample()
                new_obs_tensor, _, _, info_raw = self.envs.step(actions_tensor)
                rewards_tensor, rewards_info = self.rewarder.calculate_reward(obs_tensor, actions_tensor)

                episodes.append(obs_tensor.cpu().numpy(), actions_tensor.cpu().numpy(),
                                rewards_tensor.cpu().numpy(), batch_ids)
                self.rewarder.append(obs_tensor)
                obs_tensor = new_obs_tensor

                self._append_to_log(rewards_info, is_pre_update=params is None)

        self.rewarder.save_episodes(episodes, is_pre_update=params is None)
        return episodes

    def sample_tasks(self, num_tasks=-1):
        tasks = self.rewarder.sample_tasks(num_tasks)
        return tasks

    def reset_task(self, task):
        tasks = [task for _ in range(self.num_processes)]
        self.rewarder.set_tasks(tasks)

    def fit_rewarder(self, logger):
        start = time.time()
        self.rewarder.fit()
        logger.logkv('time_rewarder_fit', time.time() - start)

    def _append_to_log(self, info, is_pre_update):
        if is_pre_update:
            self.logging_info['pre_update'].append(info)
        else:
            self.logging_info['post_update'].append(info)

    def log_unsupervised(self, logger):

        for quantity_name in ['log_marginal', 'lambda_log_s_given_z']:
            quantity_pre = [info[quantity_name] for info in self.logging_info['pre_update']]
            quantity_post = [info[quantity_name] for info in self.logging_info['post_update']]

            quantity_avg_pre = torch.stack(quantity_pre).mean().item() * self.args.episode_length
            quantity_avg_post = torch.stack(quantity_post).mean().item() * self.args.episode_length

            logger.logkv(f"{quantity_name}_avg_pre", quantity_avg_pre)
            logger.logkv(f"{quantity_name}_avg_post", quantity_avg_post)

        self._clear_log()

    def _clear_log(self):
        self.logging_info = dict(pre_update=[], post_update=[])

