import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
from a2c_ppo_acktr.arguments import get_args
import pickle
import seaborn
from collections import defaultdict
import os
import math
import ipdb
import re
import glob
from itertools import chain

from sklearn.exceptions import NotFittedError

from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap

from pyhtmlwriter.Element import Element
from pyhtmlwriter.TableRow import TableRow
from pyhtmlwriter.Table import Table
from pyhtmlwriter.TableWriter import TableWriter


def setup_axes(ax, limit=10, walls=True):
    ax.set_xlim(left=-limit, right=limit)
    ax.set_ylim(bottom=-limit, top=limit)

    if walls:
        plot_background(ax)


def plot_background(ax):
    # walls
    ax.add_patch(patches.Rectangle((-5.5, -10), width=1, height=8.5, color='black'))
    ax.add_patch(patches.Rectangle((-5.5, 1.5), width=1, height=6, color='black'))
    ax.add_patch(patches.Rectangle((-0.5, -5.5), width=7, height=1, color='black'))


def plot_components(ax, model):
    i_component_and_weight = zip(range(len(model.weights_)), model.weights_)

    for (i, weight) in sorted(i_component_and_weight, key=lambda x: x[1]):
        v, w = np.linalg.eigh(model.covariances_[i])
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = patches.Ellipse(model.means_[i], v[0], v[1], 180. + angle, color='red')
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.35 * (weight ** 0.25))
        ax.add_artist(ell)


def add_time(trajectories):
    assert trajectories.ndim == 3
    time = np.arange(trajectories.shape[1])
    time = np.tile(time, [trajectories.shape[0], 1])
    time = np.expand_dims(time, axis=2)
    trajectories = np.concatenate([trajectories, time], axis=2)
    return trajectories


def plot_per_fitting_iteration(history):
    trajectories_all, component_ids_all, models = history['trajectories'], history['component_ids'], history['models']

    num_plots = len(trajectories_all)
    num_cols = 5
    num_rows = math.ceil(num_plots / num_cols)

    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, sharex='all', sharey='all',
                             figsize=[10, 2 * num_rows])
    axes = axes.reshape([-1])

    for i_fit in range(len(trajectories_all)):
        ax = axes[i_fit]
        setup_axes(ax)
        if i_fit > 0 and len(models) > 0:
            plot_components(ax, models[i_fit-1])

        trajectories = torch.stack(trajectories_all[i_fit]).numpy()
        trajectories = add_time(trajectories)
        states = trajectories.reshape([-1, trajectories.shape[-1]])
        ax.scatter(states[:, 0], states[:, 1], s=0.25**2, c=states[:, 2], marker='o')

        if len(models) > 0:
            # Evaluate an existing colormap from 0.5 (midpoint) to 1 (upper end)
            cmap = plt.get_cmap('Greys')
            colors = cmap(np.linspace(0.5, 1, cmap.N // 2))

            # Create a new colormap from those colors
            cmap_upper = LinearSegmentedColormap.from_list('Upper Half', colors)

            component_ids = np.array(component_ids_all[i_fit])
            component_ids_unique = np.unique(component_ids)
            for indices in [np.argwhere(component_ids == component_id) for component_id in component_ids_unique]:
                mean_trajectory = np.mean(trajectories[indices, :, :], axis=0, dtype=np.float64).squeeze(axis=0)
                ax.scatter(mean_trajectory[:, 0], mean_trajectory[:, 1], c=mean_trajectory[:, 2],
                           s=1**2, cmap=cmap_upper, marker='o')

        ax.set_title('fitting iteration {}'.format(i_fit))


def plot_previous_states_per_fitting_iteration(history):
    trajectories_all, models = history['trajectories'], history['models']
    max_trajectories = 10000

    num_plots = len(trajectories_all)
    num_cols = 5
    num_rows = math.ceil(num_plots / num_cols)

    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, sharex='all', sharey='all',
                             figsize=[10, 2 * num_rows])
    axes = axes.reshape([-1])

    for i_fit in range(len(trajectories_all)):
        ax = axes[i_fit]
        setup_axes(ax)

        if len(models) > 0:
            plot_components(ax, models[i_fit])

        trajectories = list(chain(*trajectories_all[:i_fit + 1]))
        if len(trajectories) > max_trajectories:
            indices = np.random.choice(len(trajectories), size=max_trajectories, replace=True)
            trajectories = [trajectories[index] for index in indices]
        states = torch.cat(trajectories).numpy()

        ax.scatter(states[:, 0], states[:, 1], s=0.1 ** 2, c='black', marker='o')

        ax.set_title('fitting iteration {}'.format(i_fit))


def plot_and_save(log_dir, sub_dir='plt'):
    save_dir = os.path.join(log_dir, sub_dir)
    os.makedirs(save_dir, exist_ok=True)

    history_filename = os.path.join(log_dir, 'history.pkl')
    history = pickle.load(open(history_filename, 'rb'))

    if 'point2d' in log_dir:
        plot_per_fitting_iteration(history)
        plt.savefig(os.path.join(save_dir, 'trajectories_per_iteration'))

        plot_previous_states_per_fitting_iteration(history)
        plt.savefig(os.path.join(save_dir, 'all_states'))
    elif 'half-cheetah' in log_dir or 'ant' in log_dir:
        pass
    else:
        raise ValueError

    make_html(log_dir)

    plt.close('all')
    return


def make_html(root_dir, sub_dir='plt', extension='.png'):
    contents = os.listdir(os.path.join(root_dir, sub_dir))
    regexp = re.compile('(.*){}'.format(extension), flags=re.ASCII)

    media = []

    for filename in contents:
        match = regexp.search(filename)
        if match:
            title = match[1]
            media.append((filename, title))

    media.extend(get_vae_media(root_dir))

    table = Table()
    for (filename, title) in media:
        row = TableRow()

        e = Element()
        e.addTxt(title)
        row.addElement(e)

        e = Element()
        e.addImg(img_path=filename, width=1000)
        row.addElement(e)

        table.addRow(row)
    tw = TableWriter(table, outputdir=os.path.join(root_dir, sub_dir), rowsPerPage=100)
    tw.write()


def get_vae_media(root_dir, sub_dir='plt'):
    filenames = []
    vae_dirs = glob.glob(os.path.join(root_dir, sub_dir, 'vae*'))
    for vae_dir in vae_dirs:
        vae_dir = os.path.relpath(vae_dir, os.path.join(root_dir, sub_dir))
        files = os.listdir(os.path.join(root_dir, sub_dir, vae_dir))
        if len(files) == 0:
            continue
        latest_file = max(files, key=lambda x: os.path.getctime(os.path.join(root_dir, sub_dir, vae_dir, x)))
        filenames.append(os.path.join(vae_dir, latest_file))

    media = []
    regexp = re.compile('vae_*(\d+)/*')
    for filename in filenames:
        match = regexp.search(filename)
        if match:
            title = int(match[1])
            media.append((filename, title))

    media = sorted(media, key=lambda x: x[1])

    return media


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-dir', default='./output/point2d/20190108/context_dp-mog_T30_K10_lambda0.8_ent0.1_gamma0.99')
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()
    if args.test:
        get_vae(args.log_dir)
    else:
        plot_and_save(args.log_dir)



# class Args:
#     def __init__(self):
#         pass
#
#
# def get_make_vis_dir(history):
#     vis_dir = os.path.join(history.args.log_dir, 'vis')
#     os.makedirs(vis_dir, exist_ok=True)
#     return vis_dir
#
#
# def make_gif(i_clustering, history, skip_if_exists=True):
#     vis_dir = get_make_vis_dir(history)
#     filename = os.path.join(vis_dir, 'iteration_{}.{}'.format(i_clustering, history.args.vis_type))
#     if skip_if_exists and os.path.isfile(filename):
#         print('{} exists; skipping.'.format(filename))
#         return
#
#     iteration = history.all[i_clustering]
#     generative_model = iteration['generative_model']
#     standardizer = iteration['standardizer']
#     episodes = iteration['episodes']
#     T = history.args.episode_length
#
#     def sort_episodes_by_task(episodes):
#         tasks = defaultdict(list)
#         for episode in episodes:
#             task_id = episode[1]
#             tasks[task_id].append(episode[0])
#         for key, value in tasks.items():
#             tasks[key] = torch.stack(value, dim=0)
#         return tasks
#
#     tasks = sort_episodes_by_task(episodes)
#
#     if history.args.num_components:
#         max_components = history.args.num_components
#     else:
#         max_components = 50
#
#     fig, axes = plt.subplots(nrows=math.ceil(max_components / 10), ncols=10, sharex='all', sharey='all',
#                              figsize=[50, 5 * math.ceil(max_components / 10)])
#     scats = []
#     axes = axes.reshape([-1])
#
#     for i_plot, i_component in enumerate(sorted(tasks.keys())):
#         ax = axes[i_plot]
#         limit = 10 + 0.5
#         ax.set_xlim(left=-limit, right=limit)
#         ax.set_ylim(bottom=-limit, top=limit)
#         scats.append(ax.scatter(None, None))
#
#         # walls
#         ax.add_patch(patches.Rectangle((-5.5, -10), width=1, height=8.5, color='black'))
#         ax.add_patch(patches.Rectangle((-5.5, 1.5), width=1, height=6, color='black'))
#         ax.add_patch(patches.Rectangle((-0.5, -5.5), width=7, height=1, color='black'))
#
#         if history.args.vis_type == 'png':  # else gif, handled in update()
#             states = tasks[i_component].numpy()
#             states = states.reshape([-1, states.shape[-1]])
#             ax.scatter(states[:, 0], states[:, 1], s=2**2, c='black', marker='o')
#         if generative_model:
#             if generative_model.__class__.__name__ == 'Discriminator':
#                 ax.set_title('skill {}'.format(i_component))
#             else:   # EM
#                 try:
#                     generative_model._check_is_fitted()
#                     fitted = True
#                 except NotFittedError:
#                     fitted = False
#
#                 if fitted:
#                     means = generative_model.means_
#                     covs = generative_model.covariances_
#                     if history.args.standardize_data:
#                         means = standardizer.inverse_transform(means)
#
#                     # component ellipse
#                     v, w = np.linalg.eigh(covs[i_component])
#                     v = 2. * np.sqrt(2.) * np.sqrt(v)
#                     u = w[0] / np.linalg.norm(w[0])
#                     angle = np.arctan(u[1] / u[0])
#                     angle = 180. * angle / np.pi  # convert to degrees
#                     ell = patches.Ellipse(means[i_component], v[0], v[1], 180. + angle, color='green')
#                     ell.set_clip_box(ax.bbox)
#                     ell.set_alpha(0.3)
#                     ax.add_artist(ell)
#
#                     if history.args.cluster_on == 'trajectory_embedding':
#                         ax.scatter(means[i_component, 0], means[i_component, 1],
#                                    edgecolors='g', s=30 ** 2, facecolors='none', linewidths=6)
#                     ax.set_title('(x, y): {}'.format(np.round(means[i_component], decimals=3)))
#         else:
#             ax.set_title('unfitted')
#
#         # mean trajectory
#         mean_trajectory = np.mean(tasks[i_component].numpy(), axis=0)
#         ax.plot(mean_trajectory[:, 0], mean_trajectory[:, 1], 'c-', linewidth=8)
#
#     def update(t):
#         for i_plot, i_component in enumerate(sorted(tasks.keys())):
#             ax = axes[i_plot]
#             scat = scats[i_plot]
#             label = 'timestep {0}'.format(t)
#             # Update the line and the axes (with a new xlabel). Return a tuple of
#             # "artists" that have to be redrawn for this frame.
#             num_trajectories = tasks[i_component][:, t:t + 1].shape[0]
#             num_traj_plot = min(10, num_trajectories)
#             data = tasks[i_component][-num_traj_plot:, t:t + 1, :2].reshape([-1, 2])
#             scat.set_offsets(data)
#             # scat.set_offsets(trajectories[0, :t, :2])
#             ax.set_xlabel(label)
#         return scats, axes
#     if history.args.vis_type == 'gif':
#         anim = FuncAnimation(fig, update, frames=T, interval=100)
#         anim.save(filename, writer='imagemagick', fps=10)
#     elif history.args.vis_type == 'png':
#         fig.savefig(filename)
#     plt.close('all')
#     return
#
#
# def make_html(history):
#     vis_dir = get_make_vis_dir(history)
#     contents = os.listdir(vis_dir)
#     names = list(filter(lambda content: content[-4:] in ['.gif', '.png'] and content.find('_') != -1, contents))
#
#     iters = [int(name[name.find('_')+1 : name.find('.')]) for name in names]
#     names = [os.path.join('vis', name) for name in names]
#
#     names_and_iters = sorted(zip(names, iters), key=lambda pair: pair[1], reverse=True)
#
#     table = Table()
#
#     for name, i in names_and_iters:
#         row = TableRow(rno=i)
#
#         e = Element()
#         e.addTxt('iteration {}'.format(i))
#         row.addElement(e)
#
#         e = Element()
#         e.addImg(name, width=1000)
#         row.addElement(e)
#
#         table.addRow(row)
#
#     tw = TableWriter(table, history.args.log_dir, rowsPerPage=min(max(1, len(names_and_iters)), 1000))
#     tw.write()
#
#
# def visualize(history, which='last'):
#     if len(history.all) == 0:
#         return
#     make_html(history)
#     if which == 'last':
#         iteration = len(history.all) - 1
#         make_gif(iteration, history, skip_if_exists=False)
#         make_html(history)
#     elif which == 'all':
#         num_iterations = len(history.all)
#         for i in range(num_iterations - 1, -1, -1):
#             make_gif(i, history, skip_if_exists=True)
#             make_html(history)

# def main():
#     args = Args()
#     # args.log_dir = '/home/kylehsu/experiments/umrl/output/point2d/20181231/entropy0.03_gaussian_components25'
#     args.log_dir = '/home/kylehsu/experiments/umrl/output/point2d/20190101/disc_entropy0.03_components100_length100'
#     history_shell = History(args)
#     history = pickle.load(open(history_shell.filename, 'rb'))
#     visualize(history, which='all')
#
#
# if __name__ == '__main__':
#     main()


