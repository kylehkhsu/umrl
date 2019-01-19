import argparse
import datetime
import os
import numpy as np

def get_args():
    parser = argparse.ArgumentParser(description='RL')
    now = datetime.datetime.now()

    # RL optimizer
    parser.add_argument('--algo', default='a2c',
                        help='algorithm to use: a2c | ppo | acktr')
    parser.add_argument('--lr', type=float, default=7e-4,
                        help='learning rate (default: 7e-4)')
    parser.add_argument('--eps', type=float, default=1e-5,
                        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument('--alpha', type=float, default=0.99,
                        help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument('--use-gae', action='store_true', default=False,
                        help='use generalized advantage estimation')
    parser.add_argument('--tau', type=float, default=0.95,
                        help='gae parameter (default: 0.95)')
    parser.add_argument('--entropy-coef', type=float, default=0.01,
                        help='entropy term coefficient (default: 0.01)')
    parser.add_argument('--value-loss-coef', type=float, default=0.5,
                        help='value loss coefficient (default: 0.5)')
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help='max norm of gradients (default: 0.5)')
    parser.add_argument('--recurrent-policy', action='store_true', default=False,
                        help='use a recurrent policy')
    parser.add_argument('--use-linear-lr-decay', action='store_true', default=False,
                        help='use a linear schedule on the learning rate')
    parser.add_argument('--use-linear-clip-decay', action='store_true', default=False,
                        help='use a linear schedule on the ppo clipping parameter')
    parser.add_argument('--ppo-epoch', type=int, default=4,
                        help='number of ppo epochs (default: 4)')
    parser.add_argument('--clip-param', type=float, default=0.2,
                        help='ppo clip parameter (default: 0.2)')

    # policy initialization
    parser.add_argument('--init-gain', type=float, default=np.sqrt(2),
                        help='gain for orthogonal weight matrix initialization')

    # environment, reward
    parser.add_argument('--env-name', default='PongNoFrameskip-v4',
                        help='environment to train on (default: PongNoFrameskip-v4)')
    parser.add_argument('--interface', type=str, default='contextual',
                        help='contextual or rl2 or maml')
    parser.add_argument('--rewarder', type=str, default='supervised',
                        help='supervised or unsupervised')

    # specific to args.rewarder == 'unsupervised'
    parser.add_argument('--clusterer', type=str, default='vae',
                        help='mog or dp-mog or diayn or vae')
    parser.add_argument('--cumulative-reward', type=bool, default=False)
    parser.add_argument('--reward', type=str, default='s|z')
    parser.add_argument('--conditional_coef', type=float, default=1)
    parser.add_argument('--rewarder-fit-period', type=int, default=10)
    parser.add_argument('--subsample-num', type=int, default=1024)
    parser.add_argument('--subsample-strategy', type=str, default='last-random',
                        help='last-random or random')
    parser.add_argument('--subsample-last-per-fit', type=int, default=100)

    # specific to parser.add_argument('--clusterer == 'vae'
    parser.add_argument('--vae_beta', type=float, default=0.5)
    parser.add_argument('--vae_lr', type=float, default=5e-4)
    parser.add_argument('--vae_hidden_size', type=int, default=1024)
    parser.add_argument('--vae_latent_size', type=int, default=16)
    parser.add_argument('--vae_layers', type=int, default=10)
    parser.add_argument('--vae_plot', action='store_false')
    parser.add_argument('--vae_normalize', action='store_false')
    parser.add_argument('--vae_max_fit_epoch', type=int, default=1000,
                        help='first fitting is fixed to be 1000 epochs, this controls subsequent fittings')
    parser.add_argument('--vae_weights', type=str, default=None)
    parser.add_argument('--vae_load', action='store_true')
    parser.add_argument('--vae_batches', type=int, default=8)

    # specific to args.clusterer == 'mog' or args.clusterer == 'dp-mog'
    parser.add_argument('--num_components', type=int, default=25)

    # supervised valuation or args.rewarder == 'supervised'
    parser.add_argument('--dense-coef', type=float, default=1,
                        help='coefficient on dense reward term')
    parser.add_argument('--success-coef', type=float, default=10,
                        help='coefficient on goal-reaching reward term')
    parser.add_argument('--tasks', type=str, default='single',
                        help='task number keyword')
    parser.add_argument('--task-type', type=str, default='goal',
                        help='task type keyword')

    # compute, seed
    parser.add_argument('--device', type=str, default='cpu',
                        help='cpu or cuda:{int}')
    parser.add_argument('--cuda-deterministic', action='store_true', default=False,
                        help="sets flags for determinism when using CUDA (potentially slow!)")
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')

    # steps, processes
    parser.add_argument('--num-mini-batch', type=int, default=32,
                        help='number of batches for ppo (default: 32)')
    parser.add_argument('--num-processes', type=int, default=16,
                        help='how many training CPU processes to use (default: 16)')
    parser.add_argument('--trial-length', type=int, default=1,
                        help='number of episodes in a trial')
    parser.add_argument('episode-length', type=int, default=30,
                        help='number of time steps in an episode')
    parser.add_argument('--trials-per-update', type=int, default=100,
                        help='number of trials before policy updates')
    parser.add_argument('--num-updates', type=int, default=100,
                        help='number of policy updates to run for')

    # logging, saving, visualization
    parser.add_argument('--save-period', type=int, default=10,
                        help='save period, one save per n updates')
    parser.add_argument('--vis-period', type=int, default=10,
                        help='vis period, one log per n updates')
    parser.add_argument('--experiment-name', type=str,
                        default=f"{now.year}{now.month:02d}{now.day}/"
                                f"{now.hour}:{now.minute}:{now.second}:{now.microsecond:06d}",
                        help='experiment name')
    parser.add_argument('--log_dir_root', type=str, default='./output')
    parser.add_argument('--look', action='store_true',
                        help='make videos and meta-evaluate')
    parser.add_argument('--plot', action='store_true',
                        help='plot data and make html')

    args = parser.parse_args()

    # calculated, not set
    args.trials_per_process_per_update = args.trials_per_update // args.num_processes
    args.num_steps = args.episode_length * args.trial_length * args.trials_per_process_per_update
    args.log_dir = os.path.join(args.log_dir_root, args.experiment_name)

    # unused
    # parser.add_argument('--log-period', type=int, default=10,
    #                     help='log period, one log per n updates (default: 10)')
    # parser.add_argument('--eval-period', type=int, default=None,
    #                     help='eval period, one eval per n updates (default: None)')
    # parser.add_argument('--num-env-steps', type=int, default=10e6,
    #                     help='number of environment steps to train (default: 10e6)')
    # parser.add_argument('--save-dir', default='./trained_models/',
    #                     help='directory to save agent logs (default: ./trained_models/)')
    # parser.add_argument('--add-timestep', action='store_true', default=False,
    #                     help='add timestep to observations')

    return args
