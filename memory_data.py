import numpy as np
import random
from collections import deque
from dataclasses import dataclass
import torch


@ dataclass
class Sample:
    state: np.ndarray
    action: int or np.ndarray
    reward: float
    next_state: np.ndarray
    done: bool


class SamplesMemory:
    def __init__(self, max_size, device):
        self.max_size = max_size
        self.device = device
        self.memory_buffer = deque(maxlen=max_size)  # maxlen ensure that samples num won't exceed

    def add_sample(self, state, action, reward, next_state, done):
        sample = Sample(state, action, reward, next_state, done)
        self.memory_buffer.append(sample)

    def get_batch(self, batch_size, continuous_action=False):
        batch = random.sample(self.memory_buffer, batch_size)
        f = lambda x, my_type: torch.tensor(np.vstack(x), device=self.device, dtype=my_type)

        state_batch = f([sample.state for sample in batch], torch.float)
        action_batch = f([sample.action for sample in batch], torch.float) if continuous_action \
            else f([sample.action for sample in batch], torch.long)
        reward_batch = f([sample.reward for sample in batch], torch.float)
        next_state_batch = f([sample.next_state for sample in batch], torch.float)
        done_batch = f([sample.done for sample in batch], torch.float)

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch


