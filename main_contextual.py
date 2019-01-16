import os
import time
import json
import ipdb
import torch
import datetime
import numpy as np

from env_interface import ContextualEnvInterface
from utils import logger
from a2c_ppo_acktr import algo
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from a2c_ppo_acktr.utils import get_vec_normalize, update_linear_schedule
from multiworld.envs.pygame.point2d import Point2DEnv, Point2DTrajectoryEnv
from multiworld.envs.mujoco.classic_mujoco.half_cheetah import HalfCheetahEnv
from utils.misc import save_model, calculate_state_entropy
from utils.looker import Looker
# from visualize import visualize
from visualize import plot_and_save
from subprocess import Popen

args = get_args()
# args.clusterer = 'gaussian'     # bayesian or gaussian or discriminator
#
# if args.clusterer == 'discriminator':
#
#     args.cuda = False
#     args.algo = 'ppo'
#     # args.env_name = 'Point2DEnv-traj-v0'
#     # args.env_name = 'Point2DEnv-v0'
#     args.env_name = 'Point2DWalls-corner-v0'
#     args.use_gae = True
#     args.gamma = 1
#     args.action_limit = 1.0    # change in env creation if not 1.0
#     args.obs = 'pos'    # pos or pos_speed
#     args.fixed_start = True
#     args.sparse_reward = False
#
#     args.episode_length = 30
#     args.num_processes = 10
#     args.trajectories_per_update = 200
#     args.trajectories_per_process_per_update = args.trajectories_per_update // args.num_processes
#     args.num_steps = args.episode_length * args.trajectories_per_process_per_update
#     args.num_updates = 500
#     args.num_clusterings = 250
#     args.clustering_period = args.num_updates // args.num_clusterings
#     args.trajectories_per_clustering = args.trajectories_per_process_per_update * args.num_processes \
#         * args.clustering_period
#     args.cluster_subsample_num = 10000    # trajectories
#     args.cluster_subsample_strategy = 'all'  # random or last or something_else=all
#     args.keep_entire_history = False
#
#     args.context = 'one_hot'     # cluster_mean or goal or one_hot
#     args.trajectory_embedding_type = 'avg'
#     args.task_sampling = 'uniform'     # max_I or uniform or EM
#     args.save_period = 50
#     args.component_weight_threshold = 1e-7
#     args.reward = 'z|w'     # w|z or l2 or z|w
#     args.entropy_coef = 0.003  # default: 0.01
#     args.standardize_data = False
#     args.component_constraint_l_inf = 0.01
#     args.component_constraint_l_2 = 0.01
#     args.max_components = 50
#     args.cluster_on = 'state'   # state or trajectory_embedding
#     args.weight_concentration_prior = 1e8    # 1 or 1000 or 100000
#     args.vis_type = 'png'
#     args.visualize_period = 10
#
#     args.log_dir = './output/point2d/20190102/disc_entropy0.003'
#     # args.log_dir = './output/debug/20190102/disc'
# else:
#     args.cuda = False
#     args.algo = 'ppo'
#     # args.env_name = 'Point2DEnv-traj-v0'
#     # args.env_name = 'Point2DEnv-v0'
#     args.env_name = 'Point2DWalls-corner-v0'
#     args.use_gae = True
#     args.gamma = 1
#     args.action_limit = 1.0    # change in env creation if not 1.0
#     args.obs = 'pos'    # pos or pos_speed
#     args.fixed_start = True
#     args.sparse_reward = False
#
#     args.episode_length = 30
#     args.num_processes = 10
#     args.trajectories_per_update = 200
#     args.trajectories_per_process_per_update = args.trajectories_per_update // args.num_processes
#     args.num_steps = args.episode_length * args.trajectories_per_process_per_update
#     args.num_updates = 1000
#     args.num_clusterings = 50
#     args.clustering_period = args.num_updates // args.num_clusterings
#     args.trajectories_per_clustering = args.trajectories_per_process_per_update * args.num_processes \
#         * args.clustering_period
#     args.cluster_subsample_num = 100000    # trajectories
#     args.cluster_subsample_strategy = 'random'  # random or last or skew or something_else=all
#     args.keep_entire_history = True
#
#     args.context = 'cluster_mean'     # cluster_mean or goal or one_hot
#     args.trajectory_embedding_type = 'avg'
#     args.task_sampling = 'EM'     # max_I or uniform or EM
#     args.save_period = 50
#     args.component_weight_threshold = 0
#     args.reward = 'w|z'     # w|z or l2 or z|w
#     args.conditional_coef = 0.8
#     args.entropy_coef = 0  # default: 0.01
#     args.component_constraint_l_inf = 0.01
#     args.component_constraint_l_2 = 0.01
#     args.max_components = 50
#     args.cluster_on = 'state'   # state or trajectory_embedding
#     args.weight_concentration_prior = 1e8    # 1 or 1000 or 100000
#     args.vis_type = 'png'
#     args.visualize_period = args.clustering_period
#     args.log_EM = True
#
#
#     args.log_dir = './output/point2d/20190103/gen-state_entropy0_conditional0.8_pz-EM'
#     # args.log_dir = './output/debug/20190103/w_given_z'
#
# if args.clusterer == 'discriminator':
#     assert args.cluster_on == 'state'
#     assert args.context == 'one_hot'

# objective optimization
args.cuda = True
args.algo = 'ppo'
args.entropy_coef = 0.03
args.lr = 3e-4
args.value_loss_coef = 0.1
args.gamma = 0.99
args.tau = 0.95
args.use_gae = True
args.use_linear_lr_decay = False
args.cold_start_policy = False

# environment, reward
args.interface = 'contextual'
# args.env_name = 'Point2DWalls-center-v0'
args.env_name = 'Point2DWalls-corner-v0'
args.rewarder = 'unsupervised'    # supervised or unsupervised
args.obs = 'raw'

# specific to args.rewarder == 'unsupervised'
args.cumulative_reward = False
args.clusterer = 'vae'  # mog or dp-mog or diayn or vae
args.max_components = 25    # irrelevant for vae
args.reward = 's|z'     # s|z or z|s
args.conditional_coef = 0.2
args.rewarder_fit_period = 50
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

# specific to args.rewarder == 'supervised' or supervised evaluation
args.dense_coef = 1
args.success_coef = 10
args.tasks = 'single'
args.task_type = 'goal'

# steps, processes
args.num_mini_batch = 5
args.num_processes = 5
args.trial_length = 1
args.episode_length = 30
args.trials_per_update = 100
args.trials_per_process_per_update = args.trials_per_update // args.num_processes
args.num_steps = args.episode_length * args.trial_length * args.trials_per_process_per_update
args.num_updates = 1000

# logging, saving, visualization
args.save_period = args.rewarder_fit_period
args.vis_period = args.rewarder_fit_period
args.log_dir = './output/point2d/20190115/context_vae-warm_policy-warm_lambda0.5_P50_N1000'
# args.log_dir = './output/debug/point2d/201901114/context_mog'
args.look = False
args.plot = True

# set seeds
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)

assert args.trial_length == 1

args.device = 'cuda:0' if args.cuda else 'cpu'


def initialize_policy(envs):
    actor_critic = Policy(envs.obs_shape, envs.action_space,
                          base_kwargs={'recurrent': args.recurrent_policy})
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
        if args.clusterer == 'vae':
            envs.rewarder.clusterer.to(args.device)

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
            log_marginal += info['log_marginal'].sum().item()
            lambda_log_s_given_z += info['lambda_log_s_given_z'].sum().item()

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            rollouts.insert(obs, recurrent_hidden_states, action, action_log_prob, value, reward, masks)

        assert all(done)

        if args.clusterer == 'vae':
            envs.rewarder.clusterer.to(torch.device('cpu'))

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

        # if j % args.visualize_period == 0:
        #     print('making visualization')
        #     visualize_start = time.time()
        #     visualize(rewarder.history, which='last')
        #     visualize_time += time.time() - visualize_start

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
        # logger.logkv('num_valid_components', len(rewarder.valid_components))
        logger.logkv('policy_forward_time', policy_forward_time)
        logger.logkv('policy_update_time', policy_update_time)
        logger.logkv('step_time_rewarder', step_time_rewarder)
        logger.logkv('step_time_env', step_time_env)
        logger.logkv('step_time_total', step_time_total)
        # logger.logkv('discriminator_loss', rewarder.discriminator_loss)
        logger.logkv('visualize_time', visualize_time)
        logger.logkv('rewarder_fit_time', rewarder_fit_time)
        logger.logkv('log_marginal_avg', log_marginal_avg)
        logger.logkv('lambda_log_s_given_z_avg', lambda_log_s_given_z_avg)
        logger.dumpkvs()

        if (j % args.save_period == 0 or j == args.num_updates - 1) and args.log_dir != '':
            save_model(args, actor_critic, envs, iteration=j)

        if j % args.rewarder_fit_period == 0:
            rewarder_fit_start = time.time()
            envs.fit_rewarder()
            rewarder_fit_time += time.time() - rewarder_fit_start
            if args.cold_start_policy:
                actor_critic, agent = initialize_policy(envs)

        if (j % args.vis_period == 0 or j == args.num_updates - 1) and args.log_dir != '':
            visualize_start = time.time()
            if args.look:
                looker.look(iteration=j)
            if args.plot:
                p = Popen('python visualize.py --log-dir {}'.format(args.log_dir), shell=True)
                processes.append(p)
            visualize_time += time.time() - visualize_start

            # if j == args.num_updates - 1:
            #     rewarder.history.new()    # there won't be a new fit_generative_model() call
            #     visualize(rewarder.history, which='all')
            # rewarder.history.dump()
            # print('saved model and history')
    #
    # while len(processes) != 0:
    #     for p in processes:
    #         p.wait()


if __name__ == '__main__':
    train()
