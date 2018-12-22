import matplotlib.pyplot as plt
import numpy as np
import torch
from a2c_ppo_acktr.arguments import get_args
from rewarder import History
import pickle
import seaborn
from collections import defaultdict
import os
import math
import ipdb

from matplotlib.animation import FuncAnimation

from pyhtmlwriter.Element import Element
from pyhtmlwriter.TableRow import TableRow
from pyhtmlwriter.Table import Table
from pyhtmlwriter.TableWriter import TableWriter


class Args:
    def __init__(self):
        pass


def make_gif(iteration, skip_if_exists=True):
    i = iteration

    gif_name = os.path.join(gif_dir, 'iteration{}.gif'.format(i))
    if skip_if_exists and os.path.isfile(gif_name):
        print('{} exists; skipping.'.format(gif_name))
        return gif_name

    iteration = history.all[i]
    generative_model = iteration['generative_model']
    standardizer = iteration['standardizer']
    episodes = iteration['episodes']
    means = generative_model.means_
    T = history.args.episode_length
    if history.args.standardize_embeddings:
        means = standardizer.inverse_transform(means)

    def sort_episodes_by_task(episodes):
        tasks = defaultdict(list)
        for episode in episodes:
            task_id = episode[1]
            tasks[task_id].append(episode[0])
        for key, value in tasks.items():
            tasks[key] = torch.stack(value, dim=0)
        return tasks

    tasks = sort_episodes_by_task(episodes)

    if history.args.max_components:
        max_components = history.args.max_components
    else:
        max_components = 50

    fig, axes = plt.subplots(nrows=math.ceil(max_components // 10), ncols=10, sharex=True, sharey=True,
                             figsize=[50, 5 * math.ceil(max_components // 10)])
    scats = []
    axes = axes.reshape([-1])

    for i_plot, i_component in enumerate(tasks.keys()):
        ax = axes[i_plot]
        limit = 10 + 1
        ax.set_xlim(left=-limit, right=limit)
        ax.set_ylim(bottom=-limit, top=limit)
        ax.scatter(means[i_component, 0], means[i_component, 1],
                   edgecolors='g', s=30 ** 2, facecolors='none', linewidths=6)
        scats.append(ax.scatter(None, None))
        ax.set_title('(x, y, speed): {}'.format(np.round(means[i_component], decimals=3)))

    def update(t):
        for i_plot, i_component in enumerate(tasks.keys()):
            ax = axes[i_plot]
            scat = scats[i_plot]
            label = 'timestep {0}'.format(t)
            # Update the line and the axes (with a new xlabel). Return a tuple of
            # "artists" that have to be redrawn for this frame.
            data = tasks[i_component][:, t:t + 1, :2].reshape([-1, 2])
            scat.set_offsets(data)
            # scat.set_offsets(trajectories[0, :t, :2])
            ax.set_xlabel(label)
        return scats, axes

    anim = FuncAnimation(fig, update, frames=T, interval=100)
    anim.save(gif_name, writer='imagemagick', fps=10)
    return gif_name


def make_html(gif_names):
    table = Table()

    for i, gif_name in enumerate(gif_names):
        row = TableRow(rno=i)

        e = Element()
        e.addTxt('iteration {}'.format(i))
        row.addElement(e)

        e = Element()
        e.addImg(gif_name, width=1000)
        row.addElement(e)

        table.addRow(row)

    tw = TableWriter(table, args.log_dir, rowsPerPage=min(num_iterations, 100))
    tw.write()


if __name__ == '__main__':
    args = Args()
    args.log_dir = '/home/kylehsu/experiments/umrl/output/point2d/20181220/z_given_w_bigger_keep'
    gif_dir = os.path.join(args.log_dir, 'vis')
    os.makedirs(gif_dir, exist_ok=True)
    history_shell = History(args)
    history = pickle.load(open(history_shell.filename, 'rb'))

    gif_names = []
    num_iterations = len(history.all)
    for i in range(num_iterations - 1, -1, -1):
        gif_names.append(make_gif(i, True))

    make_html(gif_names)