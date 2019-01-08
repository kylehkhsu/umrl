import os
import time
import json
import ipdb
import torch
import datetime
import numpy as np

from env_interface import RL2EnvInterface
from utils import logger
from a2c_ppo_acktr import algo
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.model import Policy, RL2Base
from a2c_ppo_acktr.storage_rl2 import RolloutStorage
from a2c_ppo_acktr.utils import update_linear_schedule
from multiworld.envs.pygame.point2d import Point2DEnv, Point2DTrajectoryEnv
from multiworld.envs.mujoco.classic_mujoco.half_cheetah import HalfCheetahEnv
from utils.misc import save_model
from utils.looker import Looker

args = get_args()
args.clusterer = 'gaussian'     # bayesian or gaussian or discriminator

args.cuda = True
args.algo = 'ppo'
args.entropy_coef = 0.01  # default: 0.01
args.lr = 3e-4
args.value_loss_coef = 0.1
args.gamma = 0.99
args.tau = 0.95
args.use_gae = True
args.use_linear_lr_decay = True

# args.env_name = 'Point2DEnv-traj-v0'
# args.env_name = 'Point2DEnv-v0'
args.interface = 'rl2'
args.env_name = 'Point2DWalls-center-v0'
# args.env_name = 'HalfCheetahVel-v0'
args.rewarder = 'supervised'
args.action_limit = 1.0    # change in env creation if not 1.0
args.obs = 'raw'    # raw
args.fixed_start = True
args.sparse_reward = False

args.num_mini_batch = 5
args.num_processes = 5
args.episode_length = 10
args.trial_length = 5
args.trials_per_update = 100
args.trials_per_process_per_update = args.trials_per_update // args.num_processes
args.num_steps = args.episode_length * args.trial_length * args.trials_per_process_per_update
args.num_updates = 100
args.num_clusterings = 50
args.clustering_period = args.num_updates // args.num_clusterings
# args.trajectories_per_clustering = args.trajectories_per_process_per_update * args.num_processes \
#     * args.clustering_period
# args.cluster_subsample_num = 100000    # trajectories
# args.cluster_subsample_strategy = 'random'  # random or last or skew or something_else=all
# args.keep_entire_history = True

args.context = 'cluster_mean'     # cluster_mean or goal or one_hot
args.trajectory_embedding_type = 'avg'
args.task_sampling = 'EM'     # max_I or uniform or EM
args.save_interval = 5
args.component_weight_threshold = 0
args.reward = 'w|z'     # w|z or l2 or z|w
args.conditional_coef = 0.8

args.component_constraint_l_inf = 0.01
args.component_constraint_l_2 = 0.01
args.max_components = 50
args.cluster_on = 'state'   # state or trajectory_embedding
args.weight_concentration_prior = 1e8    # 1 or 1000 or 100000
# args.vis_type = 'png'
args.visualize_period = args.clustering_period
args.log_EM = True
args.tasks = 'four'
args.task_type = 'goal'
args.dense_coef = 0.1
args.success_coef = 0
args.look = True

# args.log_dir = './output/point2d/20190103/gen-state_entropy0_conditional0.8_pz-EM'
args.log_dir = './output/debug/point2d/20190107/rl2_tasks-four_run1'

# assert args.num_processes > 1   # shenanigans...
if args.clusterer == 'discriminator':
    assert args.cluster_on == 'state'
    assert args.context == 'one_hot'

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)


def train():
    if os.path.isdir(args.log_dir):
        ans = input('{} exists\ncontinue and overwrite? y/n: '.format(args.log_dir))
        if ans == 'n':
            return

    logger.configure(dir=args.log_dir, format_strs=['stdout', 'log', 'csv'])
    logger.log(args)
    json.dump(vars(args), open(os.path.join(args.log_dir, 'params.json'), 'w'))

    torch.set_num_threads(2)
    device = torch.device('cuda:0' if args.cuda else 'cpu')

    start = time.time()
    rewarder_step_time, rewarder_fit_time, rewarder_reward_time = 0, 0, 0
    policy_update_time, policy_forward_time = 0, 0
    env_step_time, total_step_time = 0, 0
    visualize_time = 0

    envs = RL2EnvInterface(args)
    if args.look:
        looker = Looker(args.log_dir)

    actor_critic = Policy(envs.obs_shape, envs.action_space,
                          base=RL2Base, base_kwargs={'recurrent': True,
                                                     'num_act_dim': envs.action_space.shape[0]})
    actor_critic.to(device)
    agent = algo.PPO(actor_critic, args.clip_param, args.ppo_epoch, args.num_mini_batch,
                     args.value_loss_coef, args.entropy_coef, lr=args.lr,
                     eps=args.eps,
                     max_grad_norm=args.max_grad_norm)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                        envs.obs_shape, envs.action_space,
                        actor_critic.recurrent_hidden_state_size)
    rollouts.to(device)

    def copy_obs_into_beginning_of_storage(obs):
        obs_raw, obs_act, obs_rew, obs_flag = obs
        rollouts.obs[0].copy_(obs_raw)
        rollouts.obs_act[0].copy_(obs_act)
        rollouts.obs_rew[0].copy_(obs_rew)
        rollouts.obs_flag[0].copy_(obs_flag)

    # #TODO: remove
    # step = 0
    # _, action, _, _ = actor_critic.act(
    #                     rollouts.get_obs(step),
    #                     rollouts.recurrent_hidden_states[step],
    #                     rollouts.masks[step])
    # obs_raw = envs.envs.reset()
    # obs_raw, reward, done, info = envs.envs.step(action)
    # ipdb.set_trace()

    obs = envs.reset()
    copy_obs_into_beginning_of_storage(obs)
    for j in range(args.num_updates):
        # if j % args.clustering_period == 0 and j != 0:
        #     rewarder_fit_start = time.time()
        #     rewarder.fit_generative_model()
        #     rewarder_fit_time += time.time() - rewarder_fit_start

        if args.use_linear_lr_decay:
            update_linear_schedule(agent.optimizer, j, args.num_updates, args.lr)

        if args.algo == 'ppo' and args.use_linear_clip_decay:
            agent.clip_param = args.clip_param  * (1 - j / float(args.num_updates))

        episode_returns = [0 for i in range(args.trial_length)]
        episode_final_reward = [0 for i in range(args.trial_length)]
        i_episode = 0

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
            total_step_start = time.time()
            obs, reward, done, info = envs.step(action)
            total_step_time += time.time() - total_step_start
            env_step_time += info['env_step_time']
            rewarder_reward_time += info['reward_time']

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
        # if rewarder.clustering_counter == 0:
        #     value_loss, action_loss, dist_entropy = 0, 0, 0
        # else:
        value_loss, action_loss, dist_entropy = agent.update(rollouts)
        policy_update_time += time.time() - policy_update_start
        rollouts.after_update()

        return_avg = rollouts.rewards.sum() / (args.trials_per_update)
        reward_avg = return_avg / (args.trial_length * args.episode_length)

        # if j % args.visualize_period == 0:
        #     print('making visualization')
        #     visualize_start = time.time()
        #     visualize(rewarder.history, which='last')
        #     visualize_time += time.time() - visualize_start

        logger.logkv('value_loss', value_loss)
        logger.logkv('action_loss', action_loss)
        logger.logkv('dist_entropy', dist_entropy)
        logger.logkv('return_avg', return_avg.item())
        logger.logkv('reward_avg', reward_avg.item())
        logger.logkv('steps', (j + 1) * args.num_steps * args.num_processes)
        logger.logkv('policy_updates', (j + 1))
        logger.logkv('time', time.time() - start)
        # logger.logkv('num_valid_components', len(rewarder.valid_components))
        logger.logkv('rewarder_step_time', rewarder_step_time)
        logger.logkv('rewarder_fit_time', rewarder_fit_time)
        logger.logkv('rewarder_reward_time', rewarder_reward_time)
        logger.logkv('policy_forward_time', policy_forward_time)
        logger.logkv('policy_update_time', policy_update_time)
        logger.logkv('env_step_time', env_step_time)
        logger.logkv('total_step_time', total_step_time)
        # logger.logkv('discriminator_loss', rewarder.discriminator_loss)
        logger.logkv('visualize_time', visualize_time)
        for i_episode in range(args.trial_length):
            logger.logkv('episode_return_avg_{}'.format(i_episode),
                         episode_returns[i_episode] / args.trials_per_update)
            logger.logkv('episode_final_reward_{}'.format(i_episode),
                         episode_final_reward[i_episode] / args.trials_per_update)
        logger.dumpkvs()

        if (j % args.save_interval == 0 or j == args.num_updates - 1) and args.log_dir != '':
            save_model(args, actor_critic, envs, iteration=j)
            visualize_start = time.time()
            if args.look:
                looker.look(iteration=j)
            visualize_time += time.time() - visualize_start


if __name__ == '__main__':
    train()
