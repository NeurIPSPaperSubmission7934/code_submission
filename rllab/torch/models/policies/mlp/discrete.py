# Copyright (c) 2018 Copyright holder of the paper Generative Adversarial Model Learning
# submitted to NeurIPS 2019 for review
# All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from rllab.torch.models.policies.base import PGSupportingPolicy


class DiscreteMLP(PGSupportingPolicy):
    def __init__(self, state_dim, action_num, subaction_dims=None, hidden_sizes=None,
                 embedding=None, activation=F.elu):
        """
        :param state_dim: Input dimension for each element in sequence
        :param action_num: Output dimension for embedding
        :param subaction_dims: (optional) factorization of the action into subactions
        :param hidden_sizes: List, Size of hidden layers
        :param embedding: Embedding module of type nn.Module
        :param activation: PyTorch functional activation (e.g. F.elu)
        """

        super().__init__([state_dim, action_num], is_disc_action=True)

        #TODO: Use action space definitions to create policies that conform the environment requierement
        if subaction_dims is None:  # hacked
            subaction_dims = [action_num]

        self.subaction_dims = subaction_dims

        if hidden_sizes is None:
            hidden_sizes = [128, 128]

        self.embedding = embedding

        self.activation = activation

        self.affine_layers = nn.ModuleList()
        last_dim = state_dim
        for hidden_size in hidden_sizes:
            self.affine_layers.append(nn.Linear(last_dim, hidden_size))
            last_dim = hidden_size

        self.action_output = nn.Linear(last_dim, action_num)

    def _init_input_output_dims(self, state_dim, action_num):
        return state_dim, action_num

    def forward(self, x, embedding_features=None):
        """
        :param x: State of Shape Batch x Features
        :param embedding_features: Either None or Batch x Sequence x Features
        :return: Action probability in a discrete space and additional embedding
         information if available
        """
        embedding_info = None
        if self.embedding is not None:
            assert embedding_features is not None, "Can't run forward pass: " \
                                                   "Missing additional_features"
            embedding_output = self.embedding(embedding_features)

            # Some feature representations return additional information about
            # the embeddings (e.g. PointNet)
            if type(embedding_output) == tuple:
                x = torch.cat((x, embedding_output[0]), 1)
                embedding_info = embedding_output[0]
            else:
                x = torch.cat((x, embedding_output), 1)

        for affine in self.affine_layers:
            x = self.activation(affine(x))

        action_prob = F.softmax(self.action_output(x),1)
        return action_prob, embedding_info

    def select_action(self, x, t):
        action_prob, _ = self.forward(x)
        action = torch.multinomial(action_prob, 1)
        # FIX: this does not work multinomial needs arguments
        #action = action_prob.multinomial()
        return int(action.data.cpu().numpy()[0, 0]), dict(action_prob=action_prob)

    def get_kl(self, x):
        action_prob1, _ = self.forward(x)
        action_prob0 = Variable(action_prob1.data)
        kl = action_prob0 * (torch.log(action_prob0) - torch.log(action_prob1))
        return kl.sum(1, keepdim=True)

    def get_log_prob(self, x, actions):
        action_prob, _ = self.forward(x)
        action_prob = torch.log(action_prob)
        return action_prob.gather(1, actions.unsqueeze(1).long())

    def get_fim(self, x):
        action_prob, _ = self.forward(x)
        M = action_prob.pow(-1).view(-1).data
        return M, action_prob, {}

