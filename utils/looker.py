import sys
sys.path.append(".")
from utils.map import Map
from utils.misc import make_html, load_model
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
            self.envs.task_current[0] = task
            video = []
            video.extend(self.envs.envs.get_images())
            for t in range(self.args.episode_length * self.args.trial_length):
                with torch.no_grad():
                    value, action, _, recurrent_hidden_state = actor_critic.act(
                        obs, recurrent_hidden_state, mask, deterministic=True
                    )
                obs, rew, done, infos = self.envs.step(action)
                masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done['trial']])
                video.extend(self.envs.envs.get_images())

            filename = 'iteration_{}-task_{}.mp4'.format(iteration, task)
            imageio.mimwrite(os.path.join(self.args.log_dir, self.sub_dir, filename), video)
        make_html(self.args.log_dir, sub_dir='vis', extension='.mp4')

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


if __name__ == '__main__':
    looker = Looker(log_dir='./output/debug/point2d/20190107/rl2_tasks-four_run0')
    looker.look_all()
    # looker.look(iteration=100)
