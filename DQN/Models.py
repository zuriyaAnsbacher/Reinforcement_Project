import torch.nn as nn


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
