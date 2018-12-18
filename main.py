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
args.log_dir = None
args.algo = 'ppo'
args.num_steps = 500
args.env_name = 'Point2DEnv-traj-v0'
args.use_gae = False

num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

# logging stuff
args.log_dir = './output/debug/debug'

def train():

    params = {'randomize_position_on_reset': False,
              'fixed_goal': True
              }
    logger.configure(dir=args.log_dir, format_strs=['stdout', 'log', 'csv'])
    json.dump(params, open(os.path.join(args.log_dir, 'params.json'), 'w'))
    params = namedtuple('Params', params.keys())(*params.values())

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
                         False)

    rewarder = Rewarder(num_processes=args.num_processes,
                        observation_shape=envs.observation_space.shape)

    actor_critic = Policy(envs.observation_space.shape, envs.action_space,
                          base_kwargs={'recurrent': args.recurrent_policy})
    actor_critic.to(device)
    agent = algo.PPO(actor_critic, args.clip_param, args.ppo_epoch, args.num_mini_batch,
                     args.value_loss_coef, args.entropy_coef, lr=args.lr,
                     eps=args.eps,
                     max_grad_norm=args.max_grad_norm)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                        envs.observation_space.shape, envs.action_space,
                        actor_critic.recurrent_hidden_state_size)

    obs = envs.reset()
    rewarder.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)


    for j in range(100):
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
            obs, reward, done, infos = envs.step(action)
            # ipdb.set_trace()

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0]
                                       for done_ in done])
            rollouts.insert(obs, recurrent_hidden_states, action, action_log_prob, value, reward, masks)

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

    # A really ugly way to save a model to CPU
    save_model = actor_critic
    if args.cuda:
        save_model = copy.deepcopy(actor_critic).cpu()

    save_model = [save_model,
                  getattr(get_vec_normalize(envs), 'ob_rms', None)]

    torch.save(save_model, os.path.join(args.log_dir, args.env_name + ".pt"))

from a2c_ppo_acktr.utils import get_render_func

def look():
    args.load_dir = args.log_dir
    args.det = True
    env = make_vec_envs(args.env_name, args.seed + 1000, 1, None, None,
                        args.add_timestep, device='cpu', allow_early_resets=False)

    render_func = get_render_func(env)

    # We need to use the same statistics for normalization as used in training
    actor_critic, ob_rms = torch.load(os.path.join(args.load_dir, args.env_name + ".pt"))
    recurrent_hidden_states = torch.zeros(1, actor_critic.recurrent_hidden_state_size)
    masks = torch.zeros(1, 1)

    obs = env.reset()
    while True:
        with torch.no_grad():
            value, action, _, recurrent_hidden_states = actor_critic.act(
                obs, recurrent_hidden_states, masks, deterministic=args.det)

        # Obser reward and next obs
        obs, reward, done, _ = env.step(action)

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
    look()


    # e = Point2DWallEnv("-", render_size=84)
#     e = ImageEnv(Point2DWallEnv(wall_shape="u", render_size=84))
#     for i in range(10):
#         e.reset()
#         for j in range(50):
#             e.step(np.random.rand(2))
#             e.render()
#             im = e.get_image()