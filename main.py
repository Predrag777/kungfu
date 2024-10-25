import cv2
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributions as distributions
from torch.distributions import Categorical
import gymnasium as gym
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box

from DEEP_Q_LEARNING.main import learning_rate, discount_factor


class Network(nn.Module):
    def __init__(self, action_size):
        super(Network, self).__init__()
        self.conv1=torch.nn.Conv2d(in_channels=4, out_channels=32, kernel_size=(3,3), stride=2)
        self.conv2=torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), stride=2)
        self.conv3=torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), stride=2)
        #Flattening
        self.flatten=torch.nn.Flatten()
        #FCL
        self.fc1=torch.nn.Linear(512, 128)
        #Output
        self.fc2a=torch.nn.Linear(128, action_size)#FullConnected Action
        self.fc2s=torch.nn.Linear(128, 1)#FullConncected State

    def forward(self, state):
        x=self.conv1(state)
        x=F.relu(x)
        x=self.conv2(x)
        x=F.relu(x)
        x=self.conv3(x)
        x = F.relu(x)

        x=self.flatten(x)

        x=self.fc1(x)

        action_value=self.fc2a(x)
        state_value=self.fc2s(x)

        return action_value, state_value


learning_rate=1e-4
discount_factor=0.99
number_environments=10



class Agent():
    def __init__(self, action_size):
        self.device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.action_size=action_size
        self.network=Network(action_size).to(self.device)
        self.optimizer=torch.optim.Adam(self.network.parameters(), lr=learning_rate)

    def act(self, state):
        if state.ndim==3:
            state=[state]#Make sure it is in the batch format
        state=torch.tensor(state, dtype=torch.float32, device=self.device)
        action_values,_=self.network.forward(state)
        policy=F.softmax(action_values, dim=-1) #Convert action values into probabilities
        return np.array([np.random.choice(len(p), p=p) for p in policy.detach().cpu().numpy()])