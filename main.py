import copy
import os
import time
from collections import deque
from collections import namedtuple
import json
import ipdb
import torch
import datetime

from rewarder import Rewarder
from utils import logger
from a2c_ppo_acktr import algo
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from a2c_ppo_acktr.utils import get_vec_normalize, update_linear_schedule
from multiworld.envs.pygame.point2d import Point2DEnv, Point2DTrajectoryEnv
from visualize import visualize

args = get_args()
args.clusterer = 'discriminator'     # bayesian or gaussian or discriminator

if args.clusterer == 'discriminator':

    args.cuda = False
    args.algo = 'ppo'
    # args.env_name = 'Point2DEnv-traj-v0'
    # args.env_name = 'Point2DEnv-v0'
    args.env_name = 'Point2DWalls-corner-v0'
    args.use_gae = True
    args.gamma = 1
    args.action_limit = 1.0    # change in env creation if not 1.0
    args.obs = 'pos'    # pos or pos_speed
    args.fixed_start = True
    args.sparse_reward = False

    args.episode_length = 100
    args.num_processes = 10
    args.trajectories_per_update = 200
    args.trajectories_per_process_per_update = args.trajectories_per_update // args.num_processes
    args.num_steps = args.episode_length * args.trajectories_per_process_per_update
    args.num_updates = 500
    args.num_clusterings = 250
    args.clustering_period = args.num_updates // args.num_clusterings
    args.trajectories_per_clustering = args.trajectories_per_process_per_update * args.num_processes \
        * args.clustering_period
    args.cluster_subsample_num = 10000    # trajectories
    args.cluster_subsample_strategy = 'all'  # random or last or something_else=all
    args.keep_entire_history = False

    args.visualize_period = 50
    args.context = 'one_hot'     # cluster_mean or goal or one_hot
    args.trajectory_embedding_type = 'avg'
    args.task_sampling = 'uniform'     # max_I or uniform or EM
    args.save_interval = 50
    args.component_weight_threshold = 1e-7
    args.reward = 'z|w'     # w|z or l2 or z|w
    args.entropy_coef = 0.02  # default: 0.01
    args.standardize_data = False
    args.component_constraint_l_inf = 0.01
    args.component_constraint_l_2 = 0.01
    args.max_components = 100
    args.cluster_on = 'state'   # state or trajectory_embedding
    args.weight_concentration_prior = 1e8    # 1 or 1000 or 100000
    args.vis_type = 'png'

    args.log_dir = './output/point2d/20190101/disc_entropy0.02_components100_length100'
else:
    args.cuda = False
    args.algo = 'ppo'
    # args.env_name = 'Point2DEnv-traj-v0'
    # args.env_name = 'Point2DEnv-v0'
    args.env_name = 'Point2DWalls-corner-v0'
    args.use_gae = True
    args.gamma = 1
    args.action_limit = 1.0    # change in env creation if not 1.0
    args.obs = 'pos'    # pos or pos_speed
    args.fixed_start = True
    args.sparse_reward = False

    args.episode_length = 30
    args.num_processes = 10
    args.trajectories_per_update = 100
    args.trajectories_per_process_per_update = args.trajectories_per_update // args.num_processes
    args.num_steps = args.episode_length * args.trajectories_per_process_per_update
    args.num_updates = 1000
    args.num_clusterings = 20
    args.clustering_period = args.num_updates // args.num_clusterings
    args.trajectories_per_clustering = args.trajectories_per_process_per_update * args.num_processes \
        * args.clustering_period
    args.cluster_subsample_num = 1000    # trajectories
    args.cluster_subsample_strategy = 'all'  # random or last or something_else=all
    args.keep_entire_history = True

    args.context = 'cluster_mean'     # cluster_mean or goal or one_hot
    args.trajectory_embedding_type = 'avg'
    args.task_sampling = 'uniform'     # max_I or uniform or EM
    args.save_interval = args.clustering_period
    args.component_weight_threshold = 1e-7
    args.reward = 'z|w'     # w|z or l2 or z|w
    args.entropy_coef = 0.01  # default: 0.01
    args.standardize_data = False
    args.component_constraint_l_inf = 0.01
    args.component_constraint_l_2 = 0.01
    args.max_components = 25
    args.cluster_on = 'trajectory_embedding'   # state or trajectory_embedding
    args.weight_concentration_prior = 1e8    # 1 or 1000 or 100000

    # args.log_dir = './output/point2d/20181231/entropy0.01_gamma1e8_clusteron-trajemb_sample-all'

assert not args.standardize_data
assert args.reward == 'z|w'
if args.clusterer == 'discriminator':
    assert args.cluster_on == 'state'
    assert args.context == 'one_hot'

# num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

def train():
    logger.configure(dir=args.log_dir, format_strs=['stdout', 'log', 'csv'])
    logger.log(args)
    json.dump(vars(args), open(os.path.join(args.log_dir, 'params.json'), 'w'))

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    torch.set_num_threads(1)
    device = torch.device('cuda:0' if args.cuda else 'cpu')
    env_id = args.env_name
    envs = make_vec_envs(env_id,
                         args.seed,
                         args.num_processes,
                         args.gamma,
                         args.log_dir,
                         args.add_timestep,
                         device,
                         True)

    if args.obs == 'pos_speed':
        obs_shape = (envs.observation_space.shape[0] + 1,)
    elif args.obs == 'pos':
        obs_shape = envs.observation_space.shape
    else:
        raise ValueError

    if args.context == 'goal':
        obs_context_shape = (obs_shape[0] * 2,)
    elif args.context == 'cluster_mean':
        obs_context_shape = (obs_shape[0] * 2,)
    elif args.context == 'one_hot':
        obs_context_shape = (obs_shape[0] + args.max_components,)
    else:
        raise ValueError

    start = time.time()
    rewarder_step_time, rewarder_fit_time, rewarder_reward_time = 0, 0, 0
    policy_update_time, policy_forward_time = 0, 0
    env_step_time = 0
    visualize_time = 0

    starting_trajectories = []
    for i in range(10):
        obs = envs.reset()
        trajectory = []
        for j in range(args.episode_length):
            trajectory.append(obs)
            action = (torch.rand(args.num_processes, envs.action_space.shape[0]) - 0.5) * 2 * args.action_limit
            obs, _, _, _ = envs.step(action)
        starting_trajectories.append(torch.stack(trajectory, dim=1))
    starting_trajectories = torch.cat(starting_trajectories, dim=0)

    rewarder = Rewarder(args,
                        obs_shape=obs_shape,
                        logger=logger)
    rewarder_fit_start = time.time()
    rewarder.fit_generative_model(starting_trajectories)
    rewarder_fit_time += time.time() - rewarder_fit_start

    actor_critic = Policy(obs_context_shape, envs.action_space,
                          base_kwargs={'recurrent': args.recurrent_policy})
    actor_critic.to(device)
    agent = algo.PPO(actor_critic, args.clip_param, args.ppo_epoch, args.num_mini_batch,
                     args.value_loss_coef, args.entropy_coef, lr=args.lr,
                     eps=args.eps,
                     max_grad_norm=args.max_grad_norm)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                        obs_context_shape, envs.action_space,
                        actor_critic.recurrent_hidden_state_size)

    episode_rewards = deque(maxlen=10)

    for j in range(args.num_updates):
        if j % args.clustering_period == 0 and j != 0:
            rewarder_fit_start = time.time()
            rewarder.fit_generative_model()
            rewarder_fit_time += time.time() - rewarder_fit_start
        raw_obs = envs.reset()
        obs_context = rewarder.reset(raw_obs)
        rollouts.obs[0].copy_(obs_context)
        rollouts.to(device)

        if args.use_linear_lr_decay:
            update_linear_schedule(agent.optimizer, j, args.num_updates, args.lr)

        if args.algo == 'ppo' and args.use_linear_clip_decay:
            agent.clip_param = args.clip_param  * (1 - j / float(args.num_updates))

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
            env_step_start = time.time()
            raw_obs, _, done, infos = envs.step(action)
            env_step_time += time.time() - env_step_start
            rewarder_step_start = time.time()
            obs_context, reward, done, infos, reward_time = rewarder.step(raw_obs, done, infos)
            rewarder_reward_time += reward_time
            rewarder_step_time += time.time() - rewarder_step_start
            assert (all(done) or not any(done)) # synchronous reset
            if all(done) and step != args.num_steps - 1:
                raw_obs = envs.reset()
                obs_context = rewarder.reset(raw_obs)

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0]
                                       for done_ in done])
            rollouts.insert(obs_context, recurrent_hidden_states, action, action_log_prob, value, reward, masks)

        assert all(done), "can't allow leakage of episodes"

        with torch.no_grad():
            next_value = actor_critic.get_value(rollouts.obs[-1],
                                                rollouts.recurrent_hidden_states[-1],
                                                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.tau)

        policy_update_start = time.time()
        value_loss, action_loss, dist_entropy = agent.update(rollouts)
        policy_update_time += time.time() - policy_update_start

        rollouts.after_update()

        reward_avg = rollouts.rewards.sum() / len((rollouts.masks == 0).nonzero())

        logger.logkv('value_loss', value_loss)
        logger.logkv('action_loss', action_loss)
        logger.logkv('dist_entropy', dist_entropy)
        logger.logkv('reward_avg', reward_avg.item())
        logger.logkv('steps', (j + 1) * args.num_steps * args.num_processes)
        logger.logkv('policy_updates', (j + 1))
        logger.logkv('time', time.time() - start)
        logger.logkv('num_valid_components', len(rewarder.valid_components))
        logger.logkv('rewarder_step_time', rewarder_step_time)
        logger.logkv('rewarder_fit_time', rewarder_fit_time)
        logger.logkv('rewarder_reward_time', rewarder_reward_time)
        logger.logkv('policy_forward_time', policy_forward_time)
        logger.logkv('policy_update_time', policy_update_time)
        logger.logkv('env_step_time', env_step_time)
        logger.logkv('discriminator_loss', rewarder.discriminator_loss)
        logger.logkv('visualize_time', visualize_time)
        logger.dumpkvs()

        if (j % args.save_interval == 0 or j == args.num_updates - 1) and args.log_dir != '':

            # A really ugly way to save a model to CPU
            save_model = actor_critic
            if args.cuda:
                save_model = copy.deepcopy(actor_critic).cpu()

            save_model = [save_model,
                          getattr(get_vec_normalize(envs), 'ob_rms', None)]

            torch.save(save_model, os.path.join(args.log_dir, args.env_name + ".pt"))
            if j == args.num_updates - 1:
                rewarder.history.new()    # there won't be a new fit_generative_model() call
            rewarder.history.dump()
            print('saved model and history')

        if j % args.visualize_period == 0 or j == args.num_updates - 1:
            print('making visualization')
            visualize_start = time.time()
            visualize(args, rewarder.history)
            visualize_time += time.time() - visualize_start

from a2c_ppo_acktr.utils import get_render_func

def look():
    args.load_dir = args.log_dir
    args.det = True
    env = make_vec_envs(args.env_name, args.seed + 1000, 1, None, None,
                        args.add_timestep, device='cpu', allow_early_resets=True)

    render_func = get_render_func(env)

    # We need to use the same statistics for normalization as used in training
    actor_critic, ob_rms = torch.load(os.path.join(args.load_dir, args.env_name + ".pt"))
    recurrent_hidden_states = torch.zeros(1, actor_critic.recurrent_hidden_state_size)
    masks = torch.zeros(1, 1)

    embedding_shape = (env.observation_space.shape[0] + 1,)
    embedding_context_shape = (embedding_shape[0] * 2,)

    rewarder = Rewarder(num_processes=1,
                        embedding_shape=embedding_shape,
                        max_length_per_episode=args.episode_length,
                        reward=args.reward)

    raw_obs = env.reset()
    embedding_context = rewarder.reset(raw_obs)
    print('goal_avg_position: {}\tgoal_avg_speed: {}'.format(embedding_context[0, 3:5], embedding_context[0, 5:6]))
    while True:
        with torch.no_grad():
            value, action, _, recurrent_hidden_states = actor_critic.act(
                embedding_context, recurrent_hidden_states, masks, deterministic=args.det)

        # Obser reward and next obs
        raw_obs, _, done, infos = env.step(action)
        embedding_context, reward, done, infos = rewarder.step(raw_obs, done, infos)

        if done:
            embedding = rewarder.get_traj_embedding()
            print('avg_position: {}\tavg_speed: {}\n'.format(
                embedding[0:2], embedding[2:3]
            ))
            raw_obs = env.reset()
            embedding_context = rewarder.reset(raw_obs)
            print('goal_avg_position: {}\tgoal_avg_speed: {}'.format(embedding_context[0, 3:5], embedding_context[0, 5:6]))

        masks.fill_(0.0 if done else 1.0)

        if render_func is not None:
            # render_func('human')
            render_func()
            time.sleep(0.1)

    render_func(close=True)

if __name__ == "__main__":
    # e = Point2DEnv()
    import matplotlib.pyplot as plt

    # env = ImageEnv(Point2DEnv(render_size=256))
    # run(env)
    # main()
    train()
    # look()


    # e = Point2DWallEnv("-", render_size=84)
#     e = ImageEnv(Point2DWallEnv(wall_shape="u", render_size=84))
#     for i in range(10):
#         e.reset()
#         for j in range(50):
#             e.step(np.random.rand(2))
#             e.render()
#             im = e.get_image()