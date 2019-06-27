# Copyright (c) 2018 Copyright holder of the paper Generative Adversarial Model Learning
# submitted to NeurIPS 2019 for review
# All rights reserved.

import torch
from torch.autograd import Variable

from rllab.torch.algos.base import Optimizer
from rllab.torch.algos.reinforce.returns import compute_returns
import rllab.misc.logger as logger
from rllab.torch.utils import torch as torch_utils

def init_optimizers(networks, lr_rates=None, optimizers=None):
    args = {key: [network] for key, network in networks.items()}
    if lr_rates is not None:
        for key in args.keys():
            args[key].append(lr_rates[key])
    if optimizers is not None:
        for key in args.keys():
            args[key].append(optimizers[key])

    optimizers = {key: init_optimizer(*args[key])
                  for key in networks.keys()}
    return optimizers

def init_optimizer(network, lr_rate=0.01, optimizer=torch.optim.Adam):
    return optimizer(network.parameters(), lr=lr_rate)

class Reinforce(Optimizer):
    def __init__(self, policy, discount=0.99, lr_rate=0.01, optimizer=None, use_gpu=False):
        self.networks = self._init_networks(policy.input_dim, policy.output_dim)

        networks = self.networks.copy()
        networks['policy'] = policy
        if optimizer is not None:
            optimizers = {"policy": optimizer}
        self.optimizers = self._init_optimizers(networks, {"policy": lr_rate}, optimizers)

        self.use_gpu = use_gpu
        if self.use_gpu:
            self.networks = {k: v.cuda() for k, v in self.networks.items()}
        self.discount = discount

    def _init_networks(self, obs_dim, action_dim):
        return {}

    @staticmethod
    def _init_optimizers(networks, lr_rates=None, optimizers=None):
        return init_optimizers(networks, lr_rates=lr_rates, optimizers=optimizers)

    def update_networks(self, policy,
                        actions, masks, rewards, states, num_episodes,
                        *args, **step_kwargs):
        returns = compute_returns(rewards, masks, discount=self.discount, returns4traj=True)

        logger.record_tabular("avg_surr_reward", torch.mean(rewards).detach().numpy())
        logger.record_tabular("max_surr_return", torch.max(returns).detach().numpy())
        logger.record_tabular("min_surr_return", torch.min(returns).detach().numpy())

        self.step(policy, self.optimizers["policy"],
                  states, actions, returns, num_episodes)
        return policy

    @staticmethod
    def step(policy_net, optimizer_policy, states, actions, advantages, traj_num):
        """update policy, objective function is average trajectory return"""
        log_probs = policy_net.get_log_prob(Variable(states), Variable(actions))
        print("old log probs", log_probs)
        policy_loss = -(log_probs * Variable(advantages)).sum() / traj_num

        # flat_grad = torch_utils.compute_flat_grad(policy_loss, policy_net.parameters(), create_graph=True).detach().numpy()
        # logger.log("gradient:" + str(flat_grad))
        #
        # # check what would be the outcome if we just add gradient to the current parameters
        # prev_params = torch_utils.get_flat_params_from(policy_net).detach().numpy()
        # logger.log("old_parameters" + str(prev_params))
        # logger.log("new_parameters handcoded plus" + str(prev_params + flat_grad * 0.01))
        # logger.log("new_parameters handcoded minus" + str(prev_params - flat_grad * 0.01))

        prev_params = torch_utils.get_flat_params_from(policy_net).detach().numpy()
        logger.log("old_parameters" + str(prev_params))

        optimizer_policy.zero_grad()
        policy_loss.backward()
        logger.record_tabular("policy_loss before", policy_loss.item())
        # for param in policy_net.parameters():
        #     logger.log("parameter_grad:" + str(param.grad))

        optimizer_policy.step()

        new_params = torch_utils.get_flat_params_from(policy_net).detach().numpy()
        logger.log("old_parameters" + str(new_params))

        # calculate new loss
        log_probs = policy_net.get_log_prob(Variable(states), Variable(actions))
        print("new log probs",log_probs)
        policy_loss = -(log_probs * Variable(advantages)).sum() / traj_num
        logger.record_tabular("policy_loss after", policy_loss.item())


