import matplotlib.pyplot as plt
import matplotlib.patches as patches
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


def get_make_vis_dir(args):
    vis_dir = os.path.join(args.log_dir, 'vis')
    os.makedirs(vis_dir, exist_ok=True)
    return vis_dir


def make_gif(iteration, args, history, skip_if_exists=True):
    i = iteration

    vis_dir = get_make_vis_dir(args)
    gif_name = os.path.join(vis_dir, 'iteration_{}.gif'.format(i))
    png_name = os.path.join(vis_dir, 'iteration_{}.png'.format(i))
    if skip_if_exists and os.path.isfile(gif_name):
        print('{} exists; skipping.'.format(gif_name))
        return

    iteration = history.all[i]
    generative_model = iteration['generative_model']
    standardizer = iteration['standardizer']
    episodes = iteration['episodes']
    T = history.args.episode_length
    if not generative_model.__class__.__name__ == 'Discriminator':
        means = generative_model.means_
        covs = generative_model.covariances_
        if history.args.standardize_data:
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

    fig, axes = plt.subplots(nrows=math.ceil(max_components / 10), ncols=10, sharex=True, sharey=True,
                             figsize=[50, 5 * math.ceil(max_components / 10)])
    scats = []
    axes = axes.reshape([-1])

    for i_plot, i_component in enumerate(sorted(tasks.keys())):
        ax = axes[i_plot]
        limit = 10 + 0.5
        ax.set_xlim(left=-limit, right=limit)
        ax.set_ylim(bottom=-limit, top=limit)
        scats.append(ax.scatter(None, None))

        # walls
        ax.add_patch(patches.Rectangle((-5.5, -10), width=1, height=8.5, color='black'))
        ax.add_patch(patches.Rectangle((-5.5, 1.5), width=1, height=6, color='black'))
        ax.add_patch(patches.Rectangle((-0.5, -5.5), width=7, height=1, color='black'))

        if not generative_model.__class__.__name__ == 'Discriminator':
            # component ellipse
            v, w = np.linalg.eigh(covs[i_component])
            v = 2. * np.sqrt(2.) * np.sqrt(v)
            u = w[0] / np.linalg.norm(w[0])
            angle = np.arctan(u[1] / u[0])
            angle = 180. * angle / np.pi  # convert to degrees
            ell = patches.Ellipse(means[i_component], v[0], v[1], 180. + angle, color='green')
            ell.set_clip_box(ax.bbox)
            ell.set_alpha(0.3)
            ax.add_artist(ell)

            # # mean
            # ax.scatter(means[i_component, 0], means[i_component, 1],
            #            edgecolors='g', s=30 ** 2, facecolors='none', linewidths=6)
            ax.set_title('(x, y, speed): {}'.format(np.round(means[i_component], decimals=3)))
        else:
            mean_trajectory = np.mean(tasks[i_component].numpy(), axis=0)
            ax.plot(mean_trajectory[:, 0], mean_trajectory[:, 1], 'g-')

    def update(t):
        for i_plot, i_component in enumerate(sorted(tasks.keys())):
            ax = axes[i_plot]
            scat = scats[i_plot]
            label = 'timestep {0}'.format(t)
            # Update the line and the axes (with a new xlabel). Return a tuple of
            # "artists" that have to be redrawn for this frame.
            num_trajectories = tasks[i_component][:, t:t + 1].shape[0]
            num_traj_plot = min(10, num_trajectories)
            data = tasks[i_component][-num_traj_plot:, t:t + 1, :2].reshape([-1, 2])
            scat.set_offsets(data)
            # scat.set_offsets(trajectories[0, :t, :2])
            ax.set_xlabel(label)
        return scats, axes
    if history.args.vis_type == 'gif':
        anim = FuncAnimation(fig, update, frames=T, interval=100)
        anim.save(gif_name, writer='imagemagick', fps=10)
    elif history.args.vis_type == 'png':
        fig.savefig(png_name)
    plt.close('all')
    return


def make_html(args):
    vis_dir = get_make_vis_dir(args)
    contents = os.listdir(vis_dir)
    names = list(filter(lambda content: content[-4:] in ['.gif', '.png'] and content.find('_') != -1, contents))

    iters = [int(name[name.find('_')+1 : name.find('.')]) for name in names]
    names = [os.path.join('vis', name) for name in names]

    names_and_iters = sorted(zip(names, iters), key=lambda pair: pair[1], reverse=True)

    table = Table()

    for name, i in names_and_iters:
        row = TableRow(rno=i)

        e = Element()
        e.addTxt('iteration {}'.format(i))
        row.addElement(e)

        e = Element()
        e.addImg(name, width=1000)
        row.addElement(e)

        table.addRow(row)

    tw = TableWriter(table, args.log_dir, rowsPerPage=min(max(1, len(names_and_iters)), 100))
    tw.write()


def visualize(args, history, just_one=True):
    if len(history.all) == 0:
        return
    make_html(args)
    if just_one:
        iteration = len(history.all) - 1
        make_gif(iteration, args, history, skip_if_exists=True)
        make_html(args)
    else:
        num_iterations = len(history.all)
        for i in range(num_iterations - 1, -1, -1):
            make_gif(i, args, history, skip_if_exists=True)
            make_html(args)


if __name__ == '__main__':
    args = Args()
    # args.log_dir = '/home/kylehsu/experiments/umrl/output/point2d/20181231/entropy0.03_gaussian_components25'
    args.log_dir = '/home/kylehsu/experiments/umrl/output/point2d/20190101/disc_entropy0.03_components100_length100'
    history_shell = History(args)
    history = pickle.load(open(history_shell.filename, 'rb'))
    visualize(args, history, just_one=True)


