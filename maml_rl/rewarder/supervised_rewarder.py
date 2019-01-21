import torch
import numpy as np


class SupervisedRewarder:
    def __init__(self, args, **kwargs):
        self.args = args
        self.tasks = None
        self.goals = None

    def reset(self):
        pass

    def append(self, *args, **kwargs):
        pass

    def save_episodes(self, *args, **kwargs):
        pass

    def sample_tasks(self, *args):
        if self.args.env_name == 'Point2DWalls-corner-v0':
            goals = np.array([[-10, -10],
                              [-8, -8],
                              [-8, -5],
                              [-8, 0],
                              [-8, 5],
                              [-10, 10],
                              [0, 0],
                              [10, 0],
                              [8, 8]]).astype(np.float32)
            tasks = [dict(goal=goal) for goal in goals]
        else:
            raise NotImplementedError

        return tasks

    def set_tasks(self, tasks):
        self.tasks = tasks
        self.goals = torch.stack([torch.from_numpy(task['goal']) for task in tasks])

    def fit(self):
        raise RuntimeError

    def calculate_reward(self, obs, actions):
        if self.args.env_name == 'Point2DWalls-corner-v0':
            distance = torch.norm(self.goals - obs.cpu(), dim=-1)
            dense_reward = -distance
            success_reward = (distance < 2).float()
            reward = self.args.dense_coef * dense_reward + self.args.success_coef * success_reward
        else:
            raise NotImplementedError

        return reward, None
