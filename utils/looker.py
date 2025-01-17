import sys
sys.path.append(".")
from utils.map import Map
from utils.misc import load_model
from env_interface import RL2EnvInterface, ContextualEnvInterface
import os
import torch
import imageio
import numpy as np
import json
import re
import ipdb
from multiworld.envs.pygame.point2d import Point2DEnv, Point2DTrajectoryEnv
from multiworld.envs.mujoco.classic_mujoco.half_cheetah import HalfCheetahEnv

from collections import defaultdict
from pyhtmlwriter.Element import Element
from pyhtmlwriter.TableRow import TableRow
from pyhtmlwriter.Table import Table
from pyhtmlwriter.TableWriter import TableWriter

class Looker:
    def __init__(self, log_dir, sub_dir='vis'):

        self.args = Map(json.load(open(os.path.join(log_dir, 'params.json'), 'r')))

        # set seeds
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed_all(self.args.seed)
        np.random.seed(self.args.seed)

        if self.args.interface is None:
            print('args.interface not set, default rl2')
            self.args.interface = 'rl2'
        if self.args.interface == 'contextual':
            self.envs = ContextualEnvInterface(self.args, mode='val')
        elif self.args.interface == 'rl2':
            self.envs = RL2EnvInterface(self.args, mode='val')
        self.tasks = self.envs.rewarder.get_assess_tasks()
        self.sub_dir = sub_dir
        os.makedirs(os.path.join(self.args.log_dir, self.sub_dir), exist_ok=True)

    def look(self, iteration=-1):
        # actor_critic, obs_rms = torch.load(os.path.join(self.args.log_dir, self.args.env_name + ".pt"))
        actor_critic, obs_rms = load_model(self.args.log_dir, iteration=iteration)

        episode_returns = [0 for i in range(self.args.trial_length)]
        episode_final_reward = [0 for i in range(self.args.trial_length)]
        i_episode = 0

        def _truncate_task_name(task):
            if len(task) > 4:
                task = '{}-truncated'.format(task[:3])
            return task

        for task in self.tasks:
            recurrent_hidden_state = torch.zeros(1, actor_critic.recurrent_hidden_state_size)
            mask = torch.zeros(1, 1)

            obs = self.envs.reset()
            self.envs.set_task_one(task)    # must come after reset since reset samples a task
            video = []
            video.extend(self.envs.envs.get_images())

            assert i_episode == 0

            for t in range(self.args.episode_length * self.args.trial_length):
                with torch.no_grad():
                    value, action, _, recurrent_hidden_state = actor_critic.act(
                        obs, recurrent_hidden_state, mask, deterministic=True
                    )
                obs, reward, done, infos = self.envs.step(action)
                if self.args.interface == 'rl2':
                    masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done['trial']])
                    episode_returns[i_episode] += reward.sum().item()
                    if all(done['episode']):
                        episode_final_reward[i_episode] += reward.sum().item()
                        i_episode = (i_episode + 1) % self.args.trial_length
                elif self.args.interface == 'contextual':
                    masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
                video.extend(self.envs.envs.get_images())

            filename = 'iteration_{}-task_{}.mp4'.format(iteration, _truncate_task_name(task))
            imageio.mimwrite(os.path.join(self.args.log_dir, self.sub_dir, filename), video)
        self.make_html(self.args.log_dir, sub_dir='vis', extension='.mp4')

        return_avg = np.sum(episode_returns) / len(self.tasks)
        return return_avg, episode_returns, episode_final_reward

    def look_all(self, sub_dir='ckpt'):
        contents = os.listdir(os.path.join(self.args.log_dir, sub_dir))
        regexp = re.compile('iteration_*(\d+).pt', flags=re.ASCII)
        iterations = []
        for content in contents:
            match = regexp.search(content)
            if match:
                iterations.append(int(match[1]))
        for iteration in sorted(iterations, reverse=True):
            self.look(iteration)

    def make_html(self, root_dir, sub_dir='vis', extension='.mp4'):
        contents = os.listdir(os.path.join(root_dir, sub_dir))
        regexp = re.compile('iteration_*(\d+)-*(task_*(?s).*){}'.format(extension), flags=re.ASCII)
        iter_to_media = defaultdict(list)
        for filename in contents:
            match = regexp.search(filename)
            if match:
                iter = int(match[1])
                task_info = match[2]
                iter_to_media[iter].append((filename, task_info))

        table = Table()
        for iter in sorted(iter_to_media.keys(), reverse=True):
            row = TableRow(rno=iter)

            e = Element()
            e.addTxt('iteration {}'.format(iter))
            row.addElement(e)

            for (filename, task_info) in sorted(iter_to_media[iter], key=lambda x: x[1]):
                e = Element()
                e.addTxt(task_info)
                e.addVideo(filename)
                row.addElement(e)

            table.addRow(row)
        tw = TableWriter(table, outputdir=os.path.join(root_dir, sub_dir), rowsPerPage=3)
        tw.write()

def look_in_sub_dirs(root_dir):
    contents = os.listdir(root_dir)
    for content in contents:
        looker = Looker(log_dir=os.path.join(root_dir, content))
        looker.look_all()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-dir', default='')
    parser.add_argument('--log-dir-root', default='')
    parser.add_argument('--iteration', type=int, default=-1)
    args = parser.parse_args()
    if args.log_dir != '':
        looker = Looker(log_dir=args.log_dir)
        if args.iteration != -1:
            looker.look(iteration=args.iteration)
        else:
            looker.look_all()
    elif args.log_dir_root != '':
        look_in_sub_dirs(root_dir=args.log_dir_root)




    # looker.look(iteration=999)
    # look_in_sub_dirs('/home/kylehsu/experiments/umrl/output/half-cheetah/20190119')
