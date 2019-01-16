import os
import time
import json
import ipdb
import torch
import datetime
import numpy as np
import doodad as dd

from env_interface import RL2EnvInterface
from utils import logger
from a2c_ppo_acktr import algo
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.model import Policy, RL2Base
from a2c_ppo_acktr.storage_rl2 import RolloutStorage
from a2c_ppo_acktr.utils import get_vec_normalize, update_linear_schedule
from multiworld.envs.pygame.point2d import Point2DEnv, Point2DTrajectoryEnv
from multiworld.envs.mujoco.classic_mujoco.half_cheetah import HalfCheetahEnv
from utils.misc import save_model, calculate_state_entropy
from utils.looker import Looker
from subprocess import Popen

log_dir_root = './output'
args = get_args()

# objective optimization
args.cuda = True
args.algo = 'ppo'
args.entropy_coef = 0.01  # default: 0.01
args.lr = 3e-4
args.value_loss_coef = 0.1
args.gamma = 0.99
args.tau = 0.95
args.use_gae = True
args.use_linear_lr_decay = True
args.cold_start_policy = False

# environment, reward
args.interface = 'rl2'
args.env_name = 'Point2DWalls-corner-v0'
args.rewarder = 'unsupervised'    # supervised or unsupervised
args.obs = 'raw'

# specific to args.rewarder == 'unsupervised'
args.cumulative_reward = False
args.clusterer = 'vae'  # mog or dp-mog or diayn or vae
args.max_components = 25    # irrelevant for vae
args.reward = 's|z'     # s|z or z|s
args.conditional_coef = 0.8
args.rewarder_fit_period = 25
args.subsample_num = 2048
args.weight_concentration_prior = 1e5   # specific to dp-mog
args.subsample_strategy = 'last-random'    # skew or random or last-random
args.subsample_last_per_fit = 300
args.subsample_power = -0.01   # specific to skewing
args.context = 'latent'   # mean or all or latent

# specific to args.clusterer == 'vae'
args.vae_beta = 0.5
args.vae_lr = 5e-4
args.vae_hidden_size = 256
args.vae_latent_size = 8
args.vae_plot = True
args.vae_scale = True
args.vae_max_fit_epoch = 1000
args.vae_weights = 'point2d/20190115/context_vae-warm_policy-warm_lambda0.5_P25_N1000/ckpt/vae_39.pt'
args.vae_weights = os.path.join(log_dir_root, args.vae_weights)
args.vae_freeze = True
args.vae_load = True

# specific to args.rewarder == 'supervised' or supervised evaluation
args.dense_coef = 1
args.success_coef = 10
args.tasks = 'single'
args.task_type = 'goal'

# steps, processes
args.num_mini_batch = 5
args.num_processes = 5
args.trial_length = 2
args.episode_length = 30
args.trials_per_update = 100
args.trials_per_process_per_update = args.trials_per_update // args.num_processes
args.num_steps = args.episode_length * args.trial_length * args.trials_per_process_per_update
args.num_updates = 1000

# logging, saving, visualization
args.save_period = 1
args.vis_period = 1
args.experiment_name = 'point2d/20190117/pipeline'
args.log_dir = os.path.join(log_dir_root, args.experiment_name)
args.look = True
args.plot = False

# set seeds
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)

args.device = 'cuda:0' if args.cuda else 'cpu'


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

    envs = RL2EnvInterface(args)
    if args.look:
        looker = Looker(args.log_dir)

    actor_critic = Policy(envs.obs_shape, envs.action_space,
                          base=RL2Base, base_kwargs={'recurrent': True,
                                                     'num_act_dim': envs.action_space.shape[0]})
    actor_critic.to(args.device)
    agent = algo.PPO(actor_critic, args.clip_param, args.ppo_epoch, args.num_mini_batch,
                     args.value_loss_coef, args.entropy_coef, lr=args.lr,
                     eps=args.eps,
                     max_grad_norm=args.max_grad_norm)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                        envs.obs_shape, envs.action_space,
                        actor_critic.recurrent_hidden_state_size)
    rollouts.to(args.device)

    def copy_obs_into_beginning_of_storage(obs):
        obs_raw, obs_act, obs_rew, obs_flag = obs
        rollouts.obs[0].copy_(obs_raw)
        rollouts.obs_act[0].copy_(obs_act)
        rollouts.obs_rew[0].copy_(obs_rew)
        rollouts.obs_flag[0].copy_(obs_flag)

    for j in range(args.num_updates):
        obs = envs.reset()
        copy_obs_into_beginning_of_storage(obs)

        if args.use_linear_lr_decay:
            update_linear_schedule(agent.optimizer, j, args.num_updates, args.lr)

        if args.algo == 'ppo' and args.use_linear_clip_decay:
            agent.clip_param = args.clip_param  * (1 - j / float(args.num_updates))

        episode_returns = [0 for i in range(args.trial_length)]
        episode_final_reward = [0 for i in range(args.trial_length)]
        i_episode = 0

        log_marginal = 0
        lambda_log_s_given_z = 0

        for step in range(args.num_steps):
            # Sample actions
            policy_forward_start = time.time()
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                        rollouts.get_obs(step),
                        rollouts.recurrent_hidden_states[step],
                        rollouts.masks[step])
            policy_forward_time += time.time() - policy_forward_start

            # Obser reward and next obs
            step_total_start = time.time()
            obs, reward, done, info = envs.step(action)
            step_time_total += time.time() - step_total_start
            step_time_env += info['step_time_env']
            step_time_rewarder += info['reward_time']
            log_marginal += info['log_marginal'].sum().item()
            lambda_log_s_given_z += info['lambda_log_s_given_z'].sum().item()

            episode_returns[i_episode] += reward.sum().item()
            if all(done['episode']):
                episode_final_reward[i_episode] += reward.sum().item()
                i_episode = (i_episode + 1) % args.trial_length

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done['trial']])
            rollouts.insert(obs, recurrent_hidden_states, action, action_log_prob, value, reward, masks)

        assert all(done['trial'])

        with torch.no_grad():
            next_value = actor_critic.get_value(rollouts.get_obs(-1),
                                                rollouts.recurrent_hidden_states[-1],
                                                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.tau)

        policy_update_start = time.time()
        value_loss, action_loss, dist_entropy = agent.update(rollouts)
        policy_update_time += time.time() - policy_update_start
        rollouts.after_update()

        # metrics
        trajectories_pre = envs.trajectories_pre
        state_entropy_pre = calculate_state_entropy(args, trajectories_pre)

        trajectories_post = envs.trajectories_post
        state_entropy_post = calculate_state_entropy(args, trajectories_post)

        return_avg = rollouts.rewards.sum() / args.trials_per_update
        reward_avg = return_avg / (args.trial_length * args.episode_length)
        log_marginal_avg = log_marginal / args.trials_per_update / (args.trial_length * args.episode_length)
        lambda_log_s_given_z_avg = lambda_log_s_given_z / args.trials_per_update / (args.trial_length * args.episode_length)

        num_steps = (j + 1) * args.num_steps * args.num_processes
        num_episodes = num_steps // args.episode_length
        num_trials = num_episodes // args.trial_length

        logger.logkv('state_entropy_pre', state_entropy_pre)
        logger.logkv('state_entropy_post', state_entropy_post)
        logger.logkv('value_loss', value_loss)
        logger.logkv('action_loss', action_loss)
        logger.logkv('dist_entropy', dist_entropy)
        logger.logkv('return_avg', return_avg.item())
        logger.logkv('reward_avg', reward_avg.item())
        logger.logkv('steps', (j + 1) * args.num_steps * args.num_processes)
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
        logger.logkv('log_marginal_avg', log_marginal_avg)
        logger.logkv('lambda_log_s_given_z_avg', lambda_log_s_given_z_avg)
        for i_episode in range(args.trial_length):
            logger.logkv('episode_return_avg_{}'.format(i_episode),
                         episode_returns[i_episode] / args.trials_per_update)
            logger.logkv('episode_final_reward_{}'.format(i_episode),
                         episode_final_reward[i_episode] / args.trials_per_update)

        if (j % args.save_period == 0 or j == args.num_updates - 1) and args.log_dir != '':
            save_model(args, actor_critic, envs, iteration=j)

        if not args.vae_freeze and j % args.rewarder_fit_period == 0:
            rewarder_fit_start = time.time()
            envs.fit_rewarder()
            rewarder_fit_time += time.time() - rewarder_fit_start

        if (j % args.vis_period == 0 or j == args.num_updates - 1) and args.log_dir != '':
            visualize_start = time.time()
            if args.look:
                eval_return_avg, eval_episode_returns, eval_episode_final_reward = looker.look(iteration=j)
                logger.logkv('eval_return_avg', eval_return_avg)
                for i_episode in range(args.trial_length):
                    logger.logkv('eval_episode_return_avg_{}'.format(i_episode),
                                 eval_episode_returns[i_episode] / args.trials_per_update)
                    logger.logkv('eval_episode_final_reward_{}'.format(i_episode),
                                 eval_episode_final_reward[i_episode] / args.trials_per_update)

            if args.plot:
                p = Popen('python visualize.py --log-dir {}'.format(args.log_dir), shell=True)
                processes.append(p)
            visualize_time += time.time() - visualize_start

        logger.dumpkvs()



if __name__ == '__main__':
    train()
