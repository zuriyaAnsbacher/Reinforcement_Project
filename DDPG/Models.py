import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import initialize_layer


class Actor(nn.Module):  # todo: check weight_init
    def __init__(self, input_size, output_size, hidd1_size, hidd2_size, weight_init=3e-3):
        super(Actor, self).__init__()

        # todo: check activation function?
        self.fc1 = nn.Sequential(nn.Linear(input_size, hidd1_size), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(hidd1_size, hidd2_size), nn.ReLU())
        self.fc3 = nn.Linear(hidd2_size, output_size)

        self.weight_init = weight_init
        initialize_layer(self.fc3, self.weight_init)  # todo?

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = F.tanh(self.fc3(x))
        return x


class Critic(nn.Module):  # todo: check weight_init
    def __init__(self, input_size, action_size, hidd1_size, hidd2_size, weight_init=3e-3):
        super(Critic, self).__init__()

        self.fc1 = nn.Sequential(nn.Linear(input_size, hidd1_size), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(hidd1_size + action_size, hidd2_size), nn.ReLU())
        self.fc3 = nn.Linear(hidd2_size, 1)

        self.weight_init = weight_init
        initialize_layer(self.fc3, self.weight_init)

    def forward(self, state, action):
        x = self.fc1(state)
        x = torch.cat((x, action), dim=1)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


