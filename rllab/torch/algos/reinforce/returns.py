# Copyright (c) 2018 Copyright holder of the paper Generative Adversarial Model Learning
# submitted to NeurIPS 2019 for review
# All rights reserved.

import torch


def compute_returns(rewards, masks, discount, returns4traj=True):
    """
    Using rewards to compute returns.
    :param rewards: rewards agent obtains per step. Size: batch_size x 1
    :param masks: indicate the final step of each trajectory. 0 for last step and 1 for others. Size: batch_size x 1
    :param discount: discount coefficient
    :param returns4traj: bool, decide whether the outputs are returns from each time step on or returns for whole
    trajectory
    :return: returns of each step in whole batch. Size: batch_size x 1
    """

    returns = torch.zeros(rewards.size(0), 1).type_as(rewards)
    prev_return = 0
    
    for i in reversed(range(rewards.size(0))):
        returns[i] = rewards[i] + discount * prev_return * masks[i]
        prev_return = returns[i, 0]
    
    if returns4traj:
        traj_start = 0
        for i in range(len(returns)):
            returns[i] = returns[traj_start]
            if masks[i] == 0:
                traj_start = i + 1
    
    return returns


def returns_minus_base(returns, b, masks):
    """
    Subtract optimal baseline from returns
    :param returns: returns from each step on or for whole trajectory. Size: batch_size x 1
    :param b: Optimal baseline per time step for every trajectory. Size: length_time_horizon
    :param masks: indicate the final step of each trajectory. 0 for last step and 1 for others. Size: batch_size x 1
    :return: Returns after subtraction
    """

    returns_sub_base = returns.new(returns.size(0), 1).fill_(0)

    traj_start = 0
    for i in range(len(returns_sub_base)):
        returns_sub_base[i] = returns[i] - b[i - traj_start]
        if masks[i] == 0:
            traj_start = i + 1

    return returns_sub_base


def returns_minus_param_base(returns, b, masks):
    """
    Subtract parameter dependent optimal baseline from returns
    :param returns: returns from each step on or for whole trajectory. Size: batch_size x 1
    :param b: parameter dependent optimal baseline per time step for every trajectory. Size: T x #param
    :param masks: indicate the final step of each trajectory. 0 for last step and 1 for others. Size: batch_size x 1
    :return: Returns after subtraction. Size: #param x batch_size
    """

    returns_sub_base = returns.new(returns.size(0), b.size(1)).fill_(0)

    traj_start = 0
    for i in range(len(returns_sub_base)):
        returns_sub_base[i, :] = returns[i] - b[i - traj_start, :]
        if masks[i] == 0:
            traj_start = i + 1

    return returns_sub_base.t()
