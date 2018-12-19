# def run(env):
#     ob = env.reset()
#     # ipdb.set_trace()
#
#     for t in range(50):
#         # action = np.array([float(a) for a in input('format: action_dim_1 action_dim_2\n').split()])
#         action = np.random.random(2)
#         print('action', action)
#         ob, reward, done, info = env.step(action)
#         print('ob:', ob, 'reward:', reward, 'done:', done, 'info:', info, sep='\n')
#         logger.logkv('reward', reward)
#         logger.dumpkvs()
#         env.render()
#         im = env.get_image()
#         time.sleep(0.05)
#
#
# def main():
#     import ipdb
#     # env = ImageEnv(Point2DEnv(render_size=256, randomize_position_on_reset=params.randomize_position_on_reset), imsize=256)
#     env = ImageEnv(Point2DEnv(render_size=256,
#                                         randomize_position_on_reset=params.randomize_position_on_reset,
#                                         fixed_goal=params.fixed_goal), imsize=256)
#     for i_update in range(10000):
#         ob = env.reset()
#         for t in range(100):
#             # ipdb.set_trace()
#             state = ob['state_observation']
#             goal = ob['state_desired_goal']
#             state_goal = np.concatenate((state, goal))
#             action = index_to_action(select_action(state_goal))
#             ob, reward, done, info = env.step(action)
#             policy.rewards.append(reward)
#         logger.logkv('reward_sum', sum(policy.rewards))
#         logger.dumpkvs()
#         finish_episode()
#
#     # # test
#     # ob = env.reset()
#     # for t in range(50):
#     #     state = ob['state_observation']
#     #     goal = ob['state_desired_goal']
#     #     state_goal = np.concatenate((state, goal))
#     #     action = index_to_action(select_action(state_goal))
#     #     ob, reward, done, info = env.step(action)
#     #     env.render()
#     #     im = env.get_image()
#     #     time.sleep(0.05)



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

args = get_args()

args.num_processes = 10
args.cuda = False
args.algo = 'ppo'
# args.env_name = 'Point2DEnv-traj-v0'
args.env_name = 'Point2DEnv-v0'
args.use_gae = True
args.gamma = 1
args.save_interval = 50
args.episode_length = 30
args.num_steps = args.episode_length * 10
args.fixed_start = True
args.trajectory_embedding = 'avg'
args.reward = 'z|w'     # or l2 or 'z|w'
args.skew = False   # unused
args.context = 'cluster_mean'     # or 'goal' for debugging
args.obs = 'pos_speed'
args.clustering_period = 10
args.sparse_reward = False
args.uniform_cluster_categorical = True
args.entropy_coef = 0   # default: 0.01

# num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

# logging stuff
# args.log_dir = './output/point2d/20181218/pos_speed_avg'
args.log_dir = './output/debug/20181218/clustering_z|w'


def train():
    logger.configure(dir=args.log_dir, format_strs=['stdout', 'log', 'csv'])
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
        embedding_shape = (envs.observation_space.shape[0] + 1,)
    elif args.obs == 'pos':
        embedding_shape = envs.observation_space.shape
    else:
        raise ValueError

    if args.context == 'goal':
        embedding_context_shape = (embedding_shape[0] * 2,)
    elif args.context == 'cluster_mean':
        embedding_context_shape = (embedding_shape[0] * 2,)
    else:
        raise ValueError

    fake_trajectories = []
    for i in range(args.num_processes * args.num_steps):
        embedding = torch.rand(embedding_shape)
        embedding = embedding * torch.Tensor([1, 1, 0.5])
        embedding = embedding - torch.Tensor([0.5, 0.5, 0])
        embedding = embedding.unsqueeze(0)
        fake_trajectories.append(embedding)

    rewarder = Rewarder(args,
                        embedding_shape=embedding_shape,
                        starting_trajectories=fake_trajectories)

    actor_critic = Policy(embedding_context_shape, envs.action_space,
                          base_kwargs={'recurrent': args.recurrent_policy})
    actor_critic.to(device)
    agent = algo.PPO(actor_critic, args.clip_param, args.ppo_epoch, args.num_mini_batch,
                     args.value_loss_coef, args.entropy_coef, lr=args.lr,
                     eps=args.eps,
                     max_grad_norm=args.max_grad_norm)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                        embedding_context_shape, envs.action_space,
                        actor_critic.recurrent_hidden_state_size)

    episode_rewards = deque(maxlen=10)

    num_updates = 1000

    for j in range(num_updates):
        if j % args.clustering_period == 0:
            rewarder.fit_generative_model()
        raw_obs = envs.reset()
        embedding_context = rewarder.reset(raw_obs)
        rollouts.obs[0].copy_(embedding_context)
        rollouts.to(device)

        if args.use_linear_lr_decay:
            update_linear_schedule(agent.optimizer, j, num_updates, args.lr)

        if args.algo == 'ppo' and args.use_linear_clip_decay:
            agent.clip_param = args.clip_param  * (1 - j / float(num_updates))

        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                        rollouts.obs[step],
                        rollouts.recurrent_hidden_states[step],
                        rollouts.masks[step])

            # Obser reward and next obs
            raw_obs, _, done, infos = envs.step(action)
            embedding_context, reward, done, infos = rewarder.step(raw_obs, done, infos)
            assert (all(done) or not any(done)) # synchronous reset
            if all(done) and step != args.num_steps - 1:
                raw_obs = envs.reset()
                embedding_context = rewarder.reset(raw_obs)

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0]
                                       for done_ in done])
            rollouts.insert(embedding_context, recurrent_hidden_states, action, action_log_prob, value, reward, masks)

        assert all(done), "can't allow leakage of episodes"

        with torch.no_grad():
            next_value = actor_critic.get_value(rollouts.obs[-1],
                                                rollouts.recurrent_hidden_states[-1],
                                                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.tau)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        reward_avg = rollouts.rewards.sum() / len((rollouts.masks == 0).nonzero())

        logger.logkv('value_loss', value_loss)
        logger.logkv('action_loss', action_loss)
        logger.logkv('dist_entropy', dist_entropy)
        logger.logkv('reward_avg', reward_avg.item())
        logger.logkv('steps', (j + 1) * args.num_steps * args.num_processes)
        logger.dumpkvs()

        if (j % args.save_interval == 0 or j == num_updates - 1) and args.log_dir != '':

            # A really ugly way to save a model to CPU
            save_model = actor_critic
            if args.cuda:
                save_model = copy.deepcopy(actor_critic).cpu()

            save_model = [save_model,
                          getattr(get_vec_normalize(envs), 'ob_rms', None)]

            torch.save(save_model, os.path.join(args.log_dir, args.env_name + ".pt"))
            rewarder.history.dump()

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