import os
import time
import json
import ipdb
import torch
import datetime
import numpy as np
import doodad as dd

from env_interface import ContextualEnvInterface
from utils import logger
from arguments import get_args

from a2c_ppo_acktr import algo
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from a2c_ppo_acktr.utils import get_vec_normalize, update_linear_schedule
from multiworld.envs.pygame.point2d import Point2DEnv, Point2DTrajectoryEnv
from multiworld.envs.mujoco.classic_mujoco.half_cheetah import HalfCheetahEnv
from utils.misc import save_model, calculate_state_entropy
from utils.looker import Looker
from subprocess import Popen

args = get_args()

# set seeds
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)

assert args.trial_length == 1

def initialize_policy(envs):
    actor_critic = Policy(envs.obs_shape, envs.action_space,
                          base_kwargs=dict(recurrent=args.recurrent_policy,
                                           hidden_size=args.policy_hidden_size,
                                           init_gain=args.init_gain))
    actor_critic.to(args.device)
    agent = algo.PPO(actor_critic, args.clip_param, args.ppo_epoch, args.num_mini_batch,
                     args.value_loss_coef, args.entropy_coef, lr=args.lr,
                     eps=args.eps,
                     max_grad_norm=args.max_grad_norm)
    return actor_critic, agent


def train():
    processes = []
    if os.path.isdir(args.log_dir):
        ans = input('{} exists\ncontinue and overwrite? y/n: '.format(args.log_dir))
        if ans == 'n':
            return

    logger.configure(dir=args.log_dir, format_strs=['stdout', 'log', 'csv'])
    logger.log(args)
    json.dump(vars(args), open(os.path.join(args.log_dir, 'params.json'), 'w'))

    torch.set_num_threads(2)

    start = time.time()
    policy_update_time, policy_forward_time = 0, 0
    step_time_env, step_time_total, step_time_rewarder = 0, 0, 0
    visualize_time = 0
    rewarder_fit_time = 0

    envs = ContextualEnvInterface(args)
    if args.look:
        looker = Looker(args.log_dir)

    actor_critic, agent = initialize_policy(envs)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                        envs.obs_shape, envs.action_space,
                        actor_critic.recurrent_hidden_state_size)
    rollouts.to(args.device)

    def copy_obs_into_beginning_of_storage(obs):
        rollouts.obs[0].copy_(obs)

    for j in range(args.num_updates):

        obs = envs.reset()  # have to reset here to use updated rewarder to sample tasks
        copy_obs_into_beginning_of_storage(obs)

        if args.use_linear_lr_decay:
            update_linear_schedule(agent.optimizer, j, args.num_updates, args.lr)

        if args.algo == 'ppo' and args.use_linear_clip_decay:
            agent.clip_param = args.clip_param  * (1 - j / float(args.num_updates))

        log_marginal = 0
        lambda_log_s_given_z = 0

        for step in range(args.num_steps):
            # Sample actions
            policy_forward_start = time.time()
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                        rollouts.obs[step],
                        rollouts.recurrent_hidden_states[step],
                        rollouts.masks[step])
            policy_forward_time += time.time() - policy_forward_start

            # Obser reward and next obs
            step_total_start = time.time()
            obs, reward, done, info = envs.step(action)
            step_time_total += time.time() - step_total_start
            step_time_env += info['step_time_env']
            step_time_rewarder += info['reward_time']
            if args.rewarder == 'unsupervised' and args.clusterer == 'vae':
                log_marginal += info['log_marginal'].sum().item()
                lambda_log_s_given_z += info['lambda_log_s_given_z'].sum().item()

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            rollouts.insert(obs, recurrent_hidden_states, action, action_log_prob, value, reward, masks)

        assert all(done)

        # policy update
        with torch.no_grad():
            next_value = actor_critic.get_value(rollouts.obs[-1],
                                                rollouts.recurrent_hidden_states[-1],
                                                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.tau)

        policy_update_start = time.time()
        if args.rewarder != 'supervised' and envs.rewarder.fit_counter == 0:
            value_loss, action_loss, dist_entropy = 0, 0, 0
        else:
            value_loss, action_loss, dist_entropy = agent.update(rollouts)
        policy_update_time += time.time() - policy_update_start
        rollouts.after_update()

        # metrics
        trajectories = envs.trajectories_current_update
        state_entropy = calculate_state_entropy(args, trajectories)

        return_avg = rollouts.rewards.sum() / args.trials_per_update
        reward_avg = return_avg / (args.trial_length * args.episode_length)
        log_marginal_avg = log_marginal / args.trials_per_update / (args.trial_length * args.episode_length)
        lambda_log_s_given_z_avg = lambda_log_s_given_z / args.trials_per_update / (args.trial_length * args.episode_length)

        num_steps = (j + 1) * args.num_steps * args.num_processes
        num_episodes = num_steps // args.episode_length
        num_trials = num_episodes // args.trial_length

        logger.logkv('state_entropy', state_entropy)
        logger.logkv('value_loss', value_loss)
        logger.logkv('action_loss', action_loss)
        logger.logkv('dist_entropy', dist_entropy)
        logger.logkv('return_avg', return_avg.item())
        logger.logkv('reward_avg', reward_avg.item())
        logger.logkv('steps', num_steps)
        logger.logkv('episodes', num_episodes)
        logger.logkv('trials', num_trials)
        logger.logkv('policy_updates', (j + 1))
        logger.logkv('time', time.time() - start)
        logger.logkv('policy_forward_time', policy_forward_time)
        logger.logkv('policy_update_time', policy_update_time)
        logger.logkv('step_time_rewarder', step_time_rewarder)
        logger.logkv('step_time_env', step_time_env)
        logger.logkv('step_time_total', step_time_total)
        logger.logkv('visualize_time', visualize_time)
        logger.logkv('rewarder_fit_time', rewarder_fit_time)
        if args.rewarder == 'unsupervised' and args.clusterer == 'vae':
            logger.logkv('log_marginal_avg', log_marginal_avg)
            logger.logkv('lambda_log_s_given_z_avg', lambda_log_s_given_z_avg)
        logger.dumpkvs()

        if (j % args.save_period == 0 or j == args.num_updates - 1) and args.log_dir != '':
            save_model(args, actor_critic, envs, iteration=j)

        if j % args.rewarder_fit_period == 0:
            rewarder_fit_start = time.time()
            envs.fit_rewarder()
            rewarder_fit_time += time.time() - rewarder_fit_start

        if (j % args.vis_period == 0 or j == args.num_updates - 1) and args.log_dir != '':
            visualize_start = time.time()
            if args.look:
                looker.look(iteration=j)
            if args.plot:
                p = Popen('python visualize.py --log-dir {}'.format(args.log_dir), shell=True)
                processes.append(p)
            visualize_time += time.time() - visualize_start


if __name__ == '__main__':
    train()
