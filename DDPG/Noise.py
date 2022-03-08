import numpy as np
import copy


class OU_Noise1:
    def __init__(self, action_size, mu=0., theta=0.15, sigma=0.2):
        self.action_size = action_size
        self.mu = np.ones(self.action_size) * mu
        self.theta = theta
        self.sigma = sigma
        self.state = None
        self.reset()

    def reset(self):
        self.state = copy.copy(self.mu)

    def add_noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state


class OU_Noise2:
    def __init__(self, action_size,  mu=0., sigma=0.15, theta=0.2, dt=1e-2, x0=None):
        self.action_size = action_size
        self.mu = np.ones(self.action_size) * mu
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def add_noise(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x

        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)
