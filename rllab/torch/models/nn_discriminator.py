# Copyright (c) 2018 Copyright holder of the paper Generative Adversarial Model Learning
# submitted to NeurIPS 2019 for review
# All rights reserved.


import torch
import torch.nn as nn
import torch.nn.functional as F

"""
simple NN discriminator for a action pair input
modified from multi agent gail repo
"""
class NNDiscriminator(nn.Module):
    def __init__(self, input_dim, hidden_sizes=[32, 32], hidden_activation=F.leaky_relu, output_activation=torch.sigmoid):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_sizes = hidden_sizes

        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

        self.hidden_layers = nn.ModuleList()
        last_dim = input_dim
        for hidden_size in hidden_sizes:
            self.hidden_layers.append(nn.Linear(last_dim, hidden_size))
            last_dim = hidden_size

        self.output_layer = nn.Linear(last_dim, 1)
        self.output_layer.weight.data.mul_(0.1)
        self.output_layer.bias.data.mul_(0.0)

    def forward(self, x):
        """
        Forward pass of discriminator

        :param x: Either combined observation and action or only core and indicator observations
        :return: Prediction whether the data is coming from a policy or expert
        """
        for hidden_layer in self.hidden_layers:
            x = self.hidden_activation(hidden_layer(x))

        output = self.output_activation(self.output_layer(x))
        return output

    def forward_logits(self, x):
        """
         Forward pass of discriminator but return the logits instead of the probabilities of the prediction

         :param x: Either combined observation and action or only core and indicator observations
         :return: Prediction whether the data is coming from a policy or expert
         """
        for hidden_layer in self.hidden_layers:
            x = self.hidden_activation(hidden_layer(x))

        return self.output_layer(x)

    def surrogate_reward(self, x):
        with torch.no_grad():
            # TODO: do we need to make this numerical more stable?
            # DONE we clip prediction at certain value. s.t. we don't get numerical instabilities
            prediction = torch.clamp(self(x), max=0.99)
            reward = -torch.log(1.0 - prediction)
            return reward
        # with torch.no_grad():
        #     self.eval() # set the neural network to eval mode, which is needed for dropout, which is don't use here
        #     score = self.forward_score(x)
        #     # TODO: make this numerical more stable
        #     reward = F.logsigmoid(score)
        #     return reward

    def compute_reward(self, x):
        with torch.no_grad():
            self.eval()
            prediction = self(x)
            return prediction