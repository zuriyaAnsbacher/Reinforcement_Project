import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, input_size, output_size, hidd1_size, hidd2_size):
        super(DQN, self).__init__()

        self.fc1 = nn.Sequential(nn.Linear(input_size, hidd1_size), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(hidd1_size, hidd2_size), nn.ReLU())
        self.fc3 = nn.Linear(hidd2_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class DuelingDQN(nn.Module):
    def __init__(self, input_size, output_size, linear_hid_size, adv_hid_size, val_hid_size):
        super(DuelingDQN, self).__init__()

        # common linear layer
        self.linear1 = nn.Linear(input_size, linear_hid_size)

        # 2 linear layers for advantage calculation
        self.linear_adv_1 = nn.Linear(linear_hid_size, adv_hid_size)
        self.linear_adv_2 = nn.Linear(adv_hid_size, output_size)

        # 2 linear layers for value calculation
        self.linear_val_1 = nn.Linear(linear_hid_size, val_hid_size)
        self.linear_val_2 = nn.Linear(val_hid_size, 1)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        adv = self.linear_adv_2(F.relu(self.linear_adv_1(x)))
        val = self.linear_val_2(F.relu(self.linear_val_1(x)))

        return val + (adv - adv.mean())
