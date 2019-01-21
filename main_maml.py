import maml_rl.envs
import multiworld.envs.pygame
import multiworld.envs.mujoco
import gym
import numpy as np
import torch
import json
from utils import logger
from utils.misc import save_model_maml
import ipdb
import os
import time
import pickle
from subprocess import Popen

from maml_rl.metalearner import MetaLearner
from maml_rl.policies import CategoricalMLPPolicy, NormalMLPPolicy
from maml_rl.baseline import LinearFeatureBaseline
from maml_rl.sampler import BatchSampler, BatchSamplerMultiworld


def calculate_returns(episodes_rewards, aggregation=torch.mean):
    returns = torch.stack([aggregation(torch.sum(rewards, dim=0)) for rewards in episodes_rewards], dim=0)
    return returns


def log_main(logger, episodes, batch, args, start_time, metalearner):
    num_tasks = (batch + 1) * args.meta_batch_size
    num_trials = num_tasks * args.fast_batch_size
    num_steps = num_trials * args.episode_length * 2

    returns_pre = calculate_returns([ep.rewards for ep, _ in episodes])
    returns_post = calculate_returns([ep.rewards for _, ep in episodes])

    obs_pre = torch.cat([ep.observations.cpu() for ep, _ in episodes])
    obs_post = torch.cat([ep.observations.cpu() for _, ep in episodes])

    def calculate_entropy(obs):
        data = obs.reshape([-1, obs.shape[-1]])
        if 'Point2D' in args.env_name:
            bins = 100
            bounds = (np.array([-10, 10]), np.array([-10, 10]))
            p_s = np.histogramdd(data, bins=bins, range=bounds, density=True)[0]
            H_s = -np.sum(p_s * np.ma.log(p_s))
        else:
            raise ValueError
        return H_s

    state_entropy_pre = calculate_entropy(obs_pre)
    state_entropy_post = calculate_entropy(obs_post)

    logger.logkv('dist_entropy_pre', np.mean(metalearner.entropy_pi_pre))
    logger.logkv('dist_entropy_post', np.mean(metalearner.entropy_pi_post))
    logger.logkv('num_policy_updates', (batch + 1))
    logger.logkv('num_trials', num_trials)
    logger.logkv('num_steps', num_steps)
    logger.logkv('return_avg_pre', returns_pre.mean().item())
    logger.logkv('return_avg_post', returns_post.mean().item())
    logger.logkv('return_std_pre', returns_pre.std().item())
    logger.logkv('return_std_post', returns_post.std().item())
    logger.logkv('state_entropy_pre', state_entropy_pre)
    logger.logkv('state_entropy_post', state_entropy_post)
    logger.logkv('time', time.time() - start_time)


def val(args, sampler_val, policy, baseline, batch):
    start_time = time.time()

    from maml_rl.utils.torch_utils import weighted_normalize, weighted_mean
    tasks_val = sampler_val.sample_tasks()
    task_to_episodes = dict()
    for task in tasks_val:
        task_episodes = []
        sampler_val.reset_task(task)
        for i_episode in range(args.num_adapt_val + 1):
            episodes = sampler_val.sample(policy, gamma=args.gamma, device=args.device)
            baseline.fit(episodes)
            values = baseline(episodes)
            advantages = episodes.gae(values, tau=args.tau)
            advantages = weighted_normalize(advantages, weights=episodes.mask)
            if i_episode == 0:
                params = None
            pi = policy(episodes.observations, params=params)
            log_probs = pi.log_prob(episodes.actions)
            if log_probs.dim() > 2:
                log_probs = torch.sum(log_probs, dim=2)
            entropy = pi.entropy().mean()
            loss = -weighted_mean(log_probs * advantages, dim=0,
                                  weights=episodes.mask) - args.entropy_coef_val * entropy
            fast_lr = args.fast_lr if i_episode == 0 else args.fast_lr_val_after_one
            params = policy.update_params(loss, step_size=fast_lr, first_order=True)
            task_episodes.append(episodes)
        task_to_episodes[str(task)] = task_episodes

    for i_episode in range(args.num_adapt_val + 1):
        returns = calculate_returns([task_episodes[i_episode].rewards for task_episodes in task_to_episodes.values()])
        logger.logkv(f'val_return_avg_adapt{i_episode}', returns.mean().item())
        logger.logkv(f'val_return_std_adapt{i_episode}', returns.std().item())

    logger.logkv('val_time', time.time() - start_time)

    save_dir = os.path.join(args.log_dir, 'val')
    os.makedirs(save_dir, exist_ok=True)
    pickle.dump(task_to_episodes, open(os.path.join(save_dir, f'val_{batch}.pkl'), 'wb'))


def main(args):
    continuous_actions = (args.env_name in ['AntVel-v1', 'AntDir-v1',
        'AntPos-v0', 'HalfCheetahVel-v1', 'HalfCheetahDir-v1',
        '2DNavigation-v0', 'Point2DWalls-corner-v0', 'Ant-v0', 'HalfCheetah-v0'])

    logger.configure(dir=args.log_dir, format_strs=['stdout', 'log', 'csv'])
    logger.log(args)
    json.dump(vars(args), open(os.path.join(args.log_dir, 'params.json',), 'w'), indent=2)

    sampler = BatchSamplerMultiworld(args)
    sampler_val = BatchSamplerMultiworld(args, val=True)

    if continuous_actions:
        policy = NormalMLPPolicy(
            int(np.prod(sampler.envs.observation_space.shape)),
            int(np.prod(sampler.envs.action_space.shape)),
            hidden_sizes=(args.hidden_size,) * args.num_layers,
            bias_transformation_size=args.bias_transformation_size,
            init_gain=args.init_gain,
        )
    else:
        raise NotImplementedError
    baseline = LinearFeatureBaseline(
        int(np.prod(sampler.envs.observation_space.shape)))

    metalearner = MetaLearner(sampler, policy, baseline, gamma=args.gamma,
        fast_lr=args.fast_lr, tau=args.tau, entropy_coef=args.entropy_coef, device=args.device)

    start_time = time.time()

    processes = []

    for batch in range(args.num_batches):
        metalearner.reset()
        tasks = sampler.sample_tasks(num_tasks=args.meta_batch_size)
        episodes = metalearner.sample(tasks, first_order=args.first_order)
        if sampler.rewarder.fit_counter > 0:
            metalearner.step(episodes, max_kl=args.max_kl, cg_iters=args.cg_iters,
                cg_damping=args.cg_damping, ls_max_steps=args.ls_max_steps,
                ls_backtrack_ratio=args.ls_backtrack_ratio)

        if batch % args.rewarder_fit_period == 0:
            sampler.fit_rewarder(logger)

        sampler.log(logger)
        log_main(logger, episodes, batch, args, start_time, metalearner)

        if batch % args.save_period == 0 or batch == args.num_batches - 1:
            save_model_maml(args, policy, batch)

        if batch % args.val_period == 0 or batch == args.num_batches - 1:
            val(args, sampler_val, policy, baseline, batch)

        if batch % args.vis_period == 0 or batch == args.num_batches - 1:
            if args.plot:
                p = Popen('python maml_rl/utils/visualize.py --log-dir {}'.format(args.log_dir), shell=True)
                processes.append(p)

        logger.dumpkvs()


if __name__ == '__main__':

    from arguments_maml import get_args
    args = get_args()

    # set seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    main(args)
