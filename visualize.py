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
    args = parser.parse_args()
    plot_and_save(args.log_dir)

