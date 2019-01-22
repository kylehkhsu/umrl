import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
import pickle
import os
import ipdb
import re
import glob
from utils.map import Map
import json
import ast
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

def plot_tasks(args, history, iteration):
    episodes_pre, episodes_post = history['pre_update'][iteration], history['post_update'][iteration]
    num_tasks = max(args.meta_batch_size, 10)

    fig, axes = plt.subplots(nrows=2, ncols=num_tasks, sharex='all', sharey='all',
                             figsize=[num_tasks * 2, 4])

    cmap = plt.get_cmap('Greys')
    colors = cmap(np.linspace(0.5, 1, cmap.N // 2))
    cmap_upper = LinearSegmentedColormap.from_list('Upper Half', colors)

    for i_row, task_episodes in enumerate([episodes_pre, episodes_post]):
        task_episodes = task_episodes[-args.meta_batch_size:]     # last policy update
        task_episodes = task_episodes[:num_tasks]
        
        for i_col, episodes in enumerate(task_episodes):
            ax = axes[i_row, i_col]
            setup_axes(ax, limit=10, walls=True)

            trajs = episodes.observations.cpu().transpose(0, 1).numpy()
            trajs = add_time(trajs)
            states = trajs.reshape([-1, trajs.shape[-1]])
            ax.scatter(states[:, 0], states[:, 1], c=states[:, 2], s=1**2, marker='o')

            mean_states = np.mean(trajs, axis=0)
            ax.scatter(mean_states[:, 0], mean_states[:, 1], c=mean_states[:, 2], s=1**2,
                       cmap=cmap_upper, marker='o')


def plot_tasks_all(args, history, save_dir, skip_if_exists=True):
    num_fitting_iterations = len(history['pre_update'])
    for i_fit in range(num_fitting_iterations - 1, -1, -1):
        filepath = os.path.join(save_dir, f"tasks_{i_fit}.png")
        if os.path.isfile(filepath):
            continue

        plot_tasks(args, history, iteration=i_fit)
        plt.savefig(filepath)
        plt.close('all')

def plot_coverage_all(args, history, save_dir):
    num_fitting_iterations = len(history['pre_update'])
    num_cols = 5
    num_rows = int(np.ceil(num_fitting_iterations / num_cols))
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, sharex='all', sharey='all',
                             figsize=[20, 4 * num_rows])
    axes = axes.reshape([-1])
    for i_fit in range(num_fitting_iterations):
        ax = axes[i_fit]
        setup_axes(ax, limit=10, walls=True)

        episodes = history['post_update'][i_fit]
        trajs = [episode_batch.observations.cpu().transpose(0, 1) for episode_batch in episodes]
        trajs = torch.cat(trajs).numpy()
        trajs = add_time(trajs)
        states = trajs.reshape([-1, trajs.shape[-1]])
        ax.scatter(states[:, 0], states[:, 1], c=states[:, 2], s=1**2, marker='o')
        ax.set_title(f"fitting iteration {i_fit-1}")

    plt.savefig(os.path.join(save_dir, 'trajectories_per_fitting_iteration'))
    plt.close('all')


def plot_point2d_val(episodes):
    num_tasks = len(episodes.keys())
    num_iters = len(list(episodes.values())[0])

    fig, axes = plt.subplots(nrows=num_iters, ncols=num_tasks, sharex='all', sharey='row',
                             figsize=[3 * num_tasks, 3 * num_iters])

    for i_col, (task_name, task_episodes) in enumerate(episodes.items()):
        regexp = re.compile('goal.*(\[.*([-+]?[0-9]*\.?[0-9]+).*\])')
        match = regexp.search(task_name)
        task_name = f"goal: {match[1]}"
        goal = ast.literal_eval(match[1])

        for i_row, episode in enumerate(task_episodes):
            ax = axes[i_row, i_col]
            setup_axes(ax, limit=10, walls=True)

            if i_row == 0:
                ax.set_title(task_name)
            if i_col == 0:
                ax.set_ylabel(f"iteration {i_row}")

            trajs = episode.observations.cpu().transpose(0, 1).numpy()
            trajs = add_time(trajs)
            states = trajs.reshape([-1, trajs.shape[-1]])
            ax.scatter(states[:, 0], states[:, 1], c=states[:, 2], s=1 ** 2, marker='o')
            goal_artist = plt.Circle(goal, radius=2, color='g', alpha=0.5)
            ax.add_artist(goal_artist)


def plot_val_all(args, log_dir, plt_dir, val_dir, skip_if_exists=True):
    contents = os.listdir(val_dir)
    regexp = re.compile('(val_(\d+)).pkl', flags=re.ASCII)
    to_plot = []
    for content in contents:
        match = regexp.search(content)
        if match:
            plt_name = match[1]
            sortby = int(match[2])
            to_plot.append((content, plt_name, sortby))
    to_plot = sorted(to_plot, key=lambda x: x[2], reverse=True)

    for (pickle_name, plt_name, _) in to_plot:
        plt_path = os.path.join(plt_dir, f'{plt_name}.png')
        if os.path.isfile(plt_path) and skip_if_exists:
            continue

        pickle_path = os.path.join(val_dir, pickle_name)
        episodes = pickle.load(open(pickle_path, 'rb'))
        if 'Point2D' in args.env_name:
            plot_point2d_val(episodes)
            plt.savefig(plt_path)
            plt.close('all')


def plot_supervised(args, log_dir, plt_dir):
    os.makedirs(plt_dir, exist_ok=True)
    val_dir = os.path.join(log_dir, 'val')
    plot_val_all(args, log_dir, plt_dir, val_dir)


def plot_unsupervised(args, log_dir, plt_dir):
    history_filename = os.path.join(log_dir, 'history.pkl')
    history = pickle.load(open(history_filename, 'rb'))
    args = Map(json.load(open(os.path.join(log_dir, 'params.json'), 'r')))

    if 'point2d' in log_dir:
        plot_tasks_all(args, history, plt_dir, skip_if_exists=True)
        plot_coverage_all(args, history, plt_dir)
    elif 'half-cheetah' in log_dir or 'ant' in log_dir:
        pass
    else:
        raise ValueError

    plt.close('all')
    return


def main(log_dir):
    args = Map(json.load(open(os.path.join(log_dir, 'params.json'), 'r')))
    plt_dir = os.path.join(log_dir, 'plt')
    os.makedirs(plt_dir, exist_ok=True)

    try:
        plot_unsupervised(args, log_dir, plt_dir)
    except FileNotFoundError:
        pass

    plot_supervised(args, log_dir, plt_dir)

    make_html(log_dir)


def make_html(root_dir, sub_dir='plt', extension='.png'):
    contents = filter(lambda x: x[-4:] == extension, os.listdir(os.path.join(root_dir, sub_dir)))
    media = []

    regexp = re.compile(f'(.*(\d+)){extension}', flags=re.ASCII)

    # trajectories_per_fitting_iteration
    regexp_any = re.compile(f'(.*){extension}', flags=re.ASCII)

    for filename in contents:
        match = regexp.search(filename)
        if match:
            title = match[1]
            sortby = int(match[2])
            media.append((filename, title, sortby))
        else:
            match = regexp_any.search(filename)
            title = match[1]
            sortby = float('inf')
            media.append((filename, title, sortby))
    media = sorted(media, key=lambda x: x[2], reverse=True)

    media.extend(get_vae_media(root_dir))

    table = Table()
    for (filename, title, _) in media:
        row = TableRow()

        e = Element()
        e.addTxt(title)
        row.addElement(e)

        e = Element()
        if 'tasks' in title:
            width = 2000
        elif 'vae' in title:
            width = 1000
        elif 'val' in title:
            width = 1000
        else:
            width = 1000
        e.addImg(img_path=filename, width=width)
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
    regexp = re.compile('(vae_*(\d+))/*')
    for filename in filenames:
        match = regexp.search(filename)
        if match:
            title = match[1]
            sortby = match[2]
            media.append((filename, title, sortby))

    media = sorted(media, key=lambda x: x[2], reverse=True)

    return media


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-dir', default='')
    parser.add_argument('--log-dir-root', default='')
    args = parser.parse_args()
    if args.log_dir != '':
        main(args.log_dir)
    elif args.log_dir_root != '':
        contents = os.listdir(args.log_dir_root)
        for content in contents:
            main(os.path.join(args.log_dir_root, content))

