import argparse
import datetime
import os
import numpy as np

def get_args():
    parser = argparse.ArgumentParser(description='Reinforcement learning with '
        'Model-Agnostic Meta-Learning (MAML)')
    now = datetime.datetime.now()

    # Environment, reward
    parser.add_argument('--env-name', type=str,
        help='name of the environment')
    parser.add_argument('--rewarder', type=str, default='supervised',
                        help='supervised or unsupervised')
    parser.add_argument('--episode-length', type=int, default=30,
                        help='number of time steps in an episode')

    # specific to args.rewarder == 'unsupervised'
    parser.add_argument('--clusterer', type=str, default='vae',
                        help='mog or dp-mog or diayn or vae')
    parser.add_argument('--cumulative-reward', action='store_true', default=False)
    parser.add_argument('--reward', type=str, default='s_given_z')
    parser.add_argument('--conditional-coef', type=float, default=1)
    parser.add_argument('--rewarder-fit-period', type=int, default=10)
    parser.add_argument('--subsample-num', type=int, default=1024)
    parser.add_argument('--subsample-strategy', type=str, default='last-random',
                        help='last-random or random')

    # specific to parser.add_argument('--clusterer == 'vae'
    parser.add_argument('--vae-beta', type=float, default=0.5)
    parser.add_argument('--vae-lr', type=float, default=5e-4)
    parser.add_argument('--vae-hidden-size', type=int, default=1024)
    parser.add_argument('--vae-latent-size', type=int, default=16)
    parser.add_argument('--vae-layers', type=int, default=10)
    parser.add_argument('--vae-plot', default=False, action='store_true')
    parser.add_argument('--vae-normalize', default=False, action='store_true')
    parser.add_argument('--vae-max-fit-epoch', type=int, default=1000,
                        help='first fitting is fixed to be 1000 epochs, this controls subsequent fittings')
    parser.add_argument('--vae-weights', type=str, default='')
    parser.add_argument('--vae-load', default=False, action='store_true')
    parser.add_argument('--vae-batches', type=int, default=8)
    parser.add_argument('--vae-marginal-samples', type=int, default=128)

    parser.add_argument('--gamma', type=float, default=0.95,
        help='value of the discount factor gamma')
    parser.add_argument('--tau', type=float, default=1.0,
        help='value of the discount factor for GAE')
    parser.add_argument('--first-order', action='store_true',
        help='use the first-order approximation of MAML')

    # Policy network (relu activation function)
    parser.add_argument('--hidden-size', type=int, default=256,
        help='number of hidden units per layer')
    parser.add_argument('--num-layers', type=int, default=2,
        help='number of hidden layers')
    parser.add_argument('--bias-transformation-size', type=int, default=16)
    parser.add_argument('--init-gain', type=float, default=1)

    # Task-specific
    parser.add_argument('--fast-batch-size', type=int, default=20,
        help='batch size for each individual task')
    parser.add_argument('--fast-lr', type=float, default=0.5,
        help='learning rate for the 1-step gradient update of MAML')
    parser.add_argument('--fast-batch-size-val', type=int, default=20,
                        help='batch size for each individual task')
    parser.add_argument('--fast-lr-val-after-one', type=float, default=0.25,
                        help='we use fast-lr for first adapt')
    parser.add_argument('--num-adapt-val', type=int, default=1)

    # Optimization
    parser.add_argument('--num-batches', type=int, default=200,
        help='number of batches')
    parser.add_argument('--meta-batch-size', type=int, default=40,
        help='number of tasks per batch')
    parser.add_argument('--max-kl', type=float, default=1e-2,
        help='maximum value for the KL constraint in TRPO')
    parser.add_argument('--cg-iters', type=int, default=10,
        help='number of iterations of conjugate gradient')
    parser.add_argument('--cg-damping', type=float, default=1e-5,
        help='damping in conjugate gradient')
    parser.add_argument('--ls-max-steps', type=int, default=15,
        help='maximum number of iterations for line search')
    parser.add_argument('--ls-backtrack-ratio', type=float, default=0.8,
        help='maximum number of iterations for line search')
    parser.add_argument('--entropy-coef', type=float, default=0.001)
    parser.add_argument('--entropy-coef-val', type=float, default=0.001)

    # compute, seed
    parser.add_argument('--device', type=str, default='cpu',
                        help='cpu or cuda:{int}')
    parser.add_argument('--num-processes', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')

    # supervised validation or args.rewarder == 'supervised'
    parser.add_argument('--dense-coef', type=float, default=1,
                        help='coefficient on dense reward term')
    parser.add_argument('--success-coef', type=float, default=10,
                        help='coefficient on goal-reaching reward term')
    # parser.add_argument('--tasks', type=str, default='single',
    #                     help='task number keyword')
    # parser.add_argument('--task-type', type=str, default='goal',
    #                     help='task type keyword')

    # logging, saving, visualization, validation
    parser.add_argument('--save-period', type=int, default=10,
                        help='save period, one save per n updates')
    parser.add_argument('--vis-period', type=int, default=10,
                        help='vis period, one log per n updates')
    parser.add_argument('--val-period', type=int, default=10)
    parser.add_argument('--experiment-name', type=str,
                        default=f"{now.year:04d}{now.month:02d}{now.day:02d}-"
                                f"{now.hour:02d}:{now.minute:02d}:{now.second:02d}:{now.microsecond:06d}",
                        help='experiment name')
    parser.add_argument('--log-dir-root', type=str, default='./output/maml')
    parser.add_argument('--look', default=False, action='store_true',
                        help='make videos and meta-evaluate')
    parser.add_argument('--plot', default=False, action='store_true',
                        help='plot data and make html')

    args = parser.parse_args()

    args.log_dir = os.path.join(args.log_dir_root, args.experiment_name)

    return args