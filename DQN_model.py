import torch as th
import torch.nn as nn
import numpy as np

from avalanche.models.base_model import BaseModel

class DQNModel(nn.Module, BaseModel):
    def __init__(self, num_actions):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, padding='same')
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding='same')
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding='same')
        self.flatten = nn.Flatten()
        self.conv_out_size = self.__get_conv_out()

        self.linear1 = nn.Linear(self.conv_out_size, 512)
        self.linear2 = nn.Linear(512, num_actions)

        self.activation = nn.ReLU()

    def __get_conv_out(self):
        output = th.zeros(1, 4, 84, 84)
        output = self.conv1(output)
        output = self.conv2(output)
        output = self.conv3(output)

        return int(np.prod(output.size()))

    def forward(self, x):
        x = x.float()
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.flatten(x)
        x = self.activation(self.linear1(x))
        x = self.linear2(x)

        return x

    def get_features(self, x):
        x = x.float()
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.flatten(x)

        return x