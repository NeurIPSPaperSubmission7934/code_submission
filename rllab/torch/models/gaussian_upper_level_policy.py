# Copyright (c) 2018 Copyright holder of the paper Generative Adversarial Model Learning
# submitted to NeurIPS 2019 for review
# All rights reserved.

import torch
from torch.distributions.multivariate_normal import MultivariateNormal

"""
policy which generates parameters for a lower level policy / environment
"""
class GaussianUpperLevelPolicy(object):

    def __init__(self, mean, covariance):
        self.mean = mean
        self.covariance = covariance
        self.multivariate_normal = MultivariateNormal(self.mean, covariance_matrix=self.covariance)

    def sample(self):
        return self.multivariate_normal.sample()

    def sample_n(self, n=1):
        return self.multivariate_normal.sample((n,))

    def set_parameters(self, mean=None, covariance=None):
        if mean is not None:
            self.mean = mean
        if covariance is not None:
            self.covariance = covariance
        self.multivariate_normal = MultivariateNormal(self.mean, covariance_matrix=self.covariance)

    def get_parameters(self):
        return [self.mean, self.covariance]




