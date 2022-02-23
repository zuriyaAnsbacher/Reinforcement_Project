import numpy as np
from itertools import product
import random
import torch


def set_seed(env, seed):
    np.random.seed(seed)
    env.seed(seed)
    # env.action_space.seed(seed)  # todo: check (kfir+sharon code)
    torch.manual_seed(seed)
    random.seed(seed)
    # cudnn.deterministic = True # todo: check (kfir+sharon code)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def quantize_space(actions_range, bins):
    discrete_actions = product(*[np.linspace(start, end, num=num_of_splits) for
                                 (start, end), num_of_splits in zip(actions_range, bins)])
    return list(discrete_actions)
