import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
from pettingzoo.atari import joust_v3
import random
random.seed(981)

class Net(nn.Module):
    def __init__(self,obs_size,action_size):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(obs_size, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64*7*7, 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, action_size)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = self.bn1(x)
        x = nn.functional.relu(self.conv2(x))
        x = self.bn2(x)
        x = nn.functional.relu(self.conv3(x))
        x = self.bn3(x)
        x = x.view(x.size(0), -1) 
        x = nn.functional.relu(self.fc1(x))
        x = self.bn4(x)
        x = self.fc2(x)
        return nn.functional.softmax(x, dim=1)


IMAGE_SIZE = 84
ROWS = IMAGE_SIZE
COLS = IMAGE_SIZE
REM_STEP = 4

state_size = (REM_STEP+2, ROWS, COLS)
net = Net(state_size[0],18)
net.load_state_dict(torch.load('net_lr01'))
net.eval()
play_random_games(net)