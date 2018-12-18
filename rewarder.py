import numpy as np

class Rewarder(object):

    def __init__(self,
                 num_processes,
                 observation_shape):
        self.num_processes = num_processes
        self.observation_shape = observation_shape
        self.trajectories = []
        self.fit_counter = 0
        self.encoding_function = lambda x: x
        self.current_trajectory = [None for i in range(num_processes)]
        self.task_embedding = [np.zeros(shape=observation_shape) for i in range(num_processes)]

    def encode_trajectory(self, trajectory):
        return self.encoding_function(trajectory)

    def fit_generative_model(self):
        pass

    def skew_generative_model(self):
        pass

    def _insert_one(self, observation, i_process):
        if self.current_trajectory[i_process] is None:
            self.current_trajectory[i_process] = observation[i_process]
        else:
            np.append(self.current_trajectory[i_process], observation[i_process], axis=1)

    def _sample_task_one(self):
        position = np.random.uniform(low=-5, high=5, size=2)
        speed = np.random.uniform(low=-2, high=2, size=2)
        return np.concatenate((position, speed))

    def _reset_one(self, i_process):
        self.current_trajectory[i_process] = None
        self.task_embedding[i_process] = self._sample_task_one()

    def reset(self):
        for i in range(self.num_processes):
            self._reset_one(i)

    def step(self, observation, done):
        self._insert(observation)
        for i, done_ in enumerate(done):
            if done_:
                self._save_trajectory(i)
                self._reset_one(i)

    def _insert(self, observation):
        for i in range(self.num_processes):
            self._insert_one()

        assert observation.shape == (self.num_processes, self.observation_shape)

    def _reward(self, observation):
        embedding = np.mean(self.current_trajectory, axis=2)
        d = np.linalg.norm(embedding - self.task_embedding, axis=1)
        assert d.shape == (self.num_processes,)
        return -d

    def _save_trajectory(self, i_process):
        self.trajectories.append(self.current_trajectory[i_process])

if __name__ == '__main__':
    import ipdb
    rewarder = Rewarder(num_processes=5,
                        observation_shape=(4,))
    ipdb.set_trace()
    rewarder.reset()
    ipdb.set_trace()

