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