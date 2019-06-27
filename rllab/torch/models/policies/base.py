# Copyright (c) 2018 Copyright holder of the paper Generative Adversarial Model Learning
# submitted to NeurIPS 2019 for review
# All rights reserved.

from abc import abstractmethod
from typing import Union

import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import torch

class Policy(object):
    @abstractmethod
    def select_action(self, obs: Union[np.ndarray, Variable], t: int) -> \
            (int, dict):
        """
        Applies the policy for a given observation and time step
        :param obs: an observation vector of the environment
        :param t: the current time step
        :return: the chosen action of the policy as well as an info dictionary
        """

    def terminate(self):
        """
        Clean up operation
        """
        pass

class PytorchPolicy(Policy, nn.Module):
    pass

class ActionReplayPolicy(PytorchPolicy):

    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.normalized_input = [False]*state_dim
        self.normalized_output = [False]*action_dim

    def select_action(self, obs, t):
        try:
            action = self.replay_actions[t, :]
        except IndexError:
            action = self.replay_actions[-1, :]

        return action, dict()

    def set_param_values(self, params):
        self.replay_actions = params

class PGSupportingPolicy(PytorchPolicy):
    def __init__(self, init_io_args, is_disc_action):
        super().__init__()

        self.is_disc_action = is_disc_action
        self.input_dim, self.output_dim = \
            self._init_input_output_dims(*init_io_args)

    @abstractmethod
    def _init_input_output_dims(self, *args):
        raise NotImplementedError

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError

    @abstractmethod
    def select_action(self, obs: Union[np.ndarray, Variable], t: int) -> \
            (int, dict):
        raise NotImplementedError

    @abstractmethod
    def get_kl(self, x):
        raise NotImplementedError

    @abstractmethod
    def get_log_prob(self, x, actions):
        raise NotImplementedError

    @abstractmethod
    def get_fim(self, x):
        raise NotImplementedError
