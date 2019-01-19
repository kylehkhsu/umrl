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

class Looker:
    def __init__(self, log_dir, sub_dir='vis'):
        self.args = Map(json.load(open(os.path.join(log_dir, 'params.json'), 'r')))
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
        from pyvirtualdisplay import Display
        display = Display(visible=False, size=(256, 256))
        display.start()
        _ = self.envs.envs.get_images()

    def look(self, iteration=-1):
        # actor_critic, obs_rms = torch.load(os.path.join(self.args.log_dir, self.args.env_name + ".pt"))
        actor_critic, obs_rms = load_model(self.args.log_dir, iteration=iteration)
        for task in self.tasks:
            recurrent_hidden_state = torch.zeros(1, actor_critic.recurrent_hidden_state_size)
            mask = torch.zeros(1, 1)

            obs = self.envs.reset()
            self.envs.set_task_one(task)    # must come after reset since reset samples a task
            video = []
            video.extend(self.envs.envs.get_images())
            for t in range(self.args.episode_length * self.args.trial_length):
                with torch.no_grad():
                    value, action, _, recurrent_hidden_state = actor_critic.act(
                        obs, recurrent_hidden_state, mask, deterministic=True
                    )
                obs, rew, done, infos = self.envs.step(action)
                if self.args.interface == 'rl2':
                    masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done['trial']])
                elif self.args.interface == 'contextual':
                    masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
                video.extend(self.envs.envs.get_images())

            filename = 'iteration_{}-task_{}.mp4'.format(iteration, task)
            imageio.mimwrite(os.path.join(self.args.log_dir, self.sub_dir, filename), video)
        self.make_html(self.args.log_dir, sub_dir='vis', extension='.mp4')

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

    def __del__(self):
        self.envs.envs.close()

    def make_html(self, root_dir, sub_dir='vis', extension='.mp4'):
        contents = os.listdir(os.path.join(root_dir, sub_dir))
        regexp = re.compile('iteration_*(\d+)-*(task_*.+){}'.format(extension), flags=re.ASCII)
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
                e.addVideo(os.path.join(sub_dir, filename))
                row.addElement(e)

            table.addRow(row)
        tw = TableWriter(table, outputdir=root_dir, rowsPerPage=len(iter_to_media))
        tw.write()


if __name__ == '__main__':
    # looker = Looker(log_dir='./output/debug/half-cheetah/20190106/rl2_tasks-direction-two_run4')
    looker = Looker(log_dir='./output/debug/point2d/20190107/context_tasks-four_run2')
    looker.look_all()
    # looker.look(iteration=999)
