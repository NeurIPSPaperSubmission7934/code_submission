# Copyright (c) 2018 Copyright holder of the paper Generative Adversarial Model Learning
# submitted to NeurIPS 2019 for review
# All rights reserved.

import numpy as np
import torch


class Optimizer(object):
    def __init__(self, policy, use_gpu=False):
        self.networks = self._init_networks(policy.input_dim, policy.output_dim)

        networks = self.networks.copy()
        networks['policy'] = policy
        self.optimizers = self._init_optimizers(networks)

        self.use_gpu = use_gpu
        if self.use_gpu:
            self.networks = {k: v.cuda() for k, v in self.networks.items()}

    @classmethod
    def _init_networks(cls, obs_dim, action_dim):
        raise NotImplementedError

    def process_batch(self, policy, batch, update_policy_args):
        states, actions, rewards, masks = unpack_batch(batch)
        if self.use_gpu:
            states, actions, rewards, masks = map(
                lambda x: x.cuda(), [states, actions, rewards, masks])
        policy = self.update_networks(
            policy, actions, masks, rewards, states,
            batch["num_episodes"], *update_policy_args)
        return policy

    def update_networks(self, policy,
                        actions, masks, rewards, states, num_episodes,
                        *args, **step_kwargs):
        raise NotImplementedError

    @staticmethod
    def _init_optimizers(networks, lr_rates=None):
        return init_optimizers(networks, lr_rates=lr_rates)


def init_optimizers(networks, lr_rates=None):
    args = {key: [network] for key, network in networks.items()}
    if lr_rates is not None:
        for key in args.keys():
            args[key].append(lr_rates[key])

    optimizers = {key: init_optimizer(*args[key])
                  for key in networks.keys()}
    return optimizers


def unpack_batch(batch):
    states = torch.from_numpy(np.array(batch["states"], dtype=np.float32))
    rewards = torch.from_numpy(np.array(batch["rewards"], dtype=np.float32))
    masks = torch.from_numpy(np.array(batch["masks"], dtype=np.float32))
    actions = torch.from_numpy(np.array(batch["actions"]))
    return states, actions, rewards, masks


def init_optimizer(network, lr_rate=0.01):
    return torch.optim.Adam(network.parameters(), lr=lr_rate)

