"""
Created by: louisenaud on 5/24/22 at 3:33 PM for banana_pickup.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, layers, seed=199):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            layers (tuple or list): Size of each input sample for each layer
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, layers[0], bias=False)
        self.fc2 = nn.Linear(layers[0], layers[1], bias=False)
        self.fc3 = nn.Linear(layers[1], action_size, bias=False)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
