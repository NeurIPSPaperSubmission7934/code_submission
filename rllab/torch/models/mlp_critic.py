# Copyright (c) 2018 Copyright holder of the paper Generative Adversarial Model Learning
# submitted to NeurIPS 2019 for review
# All rights reserved.

# modified file from rlrl repo

import torch.nn as nn
import torch.nn.functional as F

"""
critic V(s)
"""
class StateValue(nn.Module):
    def __init__(self, input_dim, hidden_size=(128, 128), activation='tanh'):
        super().__init__()
        if activation == 'tanh':
            self.activation = F.tanh
        elif activation == 'relu':
            self.activation = F.relu
        elif activation == 'sigmoid':
            self.activation = F.sigmoid

        self.affine_layers = nn.ModuleList()
        last_dim = input_dim
        for nh in hidden_size:
            self.affine_layers.append(nn.Linear(last_dim, nh))
            last_dim = nh

        self.value_head = nn.Linear(last_dim, 1)
        self.value_head.weight.data.mul_(0.1)
        self.value_head.bias.data.mul_(0.0)

    def forward(self, x):
        """
        :param x: tensor with dim (batchsize, state_dim + action_dim)
        :return: the predicted value of the critic
        """
        for affine in self.affine_layers:
            x = self.activation(affine(x))

        value = self.value_head(x)
        return value
