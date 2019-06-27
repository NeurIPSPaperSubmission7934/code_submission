# Copyright (c) 2018 Copyright holder of the paper Generative Adversarial Model Learning
# submitted to NeurIPS 2019 for review
# All rights reserved.

def gae(rewards, masks, values, discount, gae_lambda, use_gpu=False):
    if use_gpu:
        rewards, masks, values = rewards.cpu(), masks.cpu(), values.cpu()

    tensor_type = type(rewards)
    returns = tensor_type(rewards.size(0), 1)
    deltas = tensor_type(rewards.size(0), 1)
    advantages = tensor_type(rewards.size(0), 1)

    prev_return = 0
    prev_value = 0
    prev_advantage = 0
    for i in reversed(range(rewards.size(0))):
        returns[i] = rewards[i] + discount * prev_return * masks[i]
        deltas[i] = rewards[i] + discount * prev_value * masks[i] - values[i]
        advantages[i] = deltas[i] + discount * gae_lambda * prev_advantage * masks[i]

        prev_return = returns[i, 0]
        prev_value = values[i, 0]
        prev_advantage = advantages[i, 0]
    advantages = (advantages - advantages.mean()) / advantages.std()

    if use_gpu:
        advantages, returns = advantages.cuda(), returns.cuda()
    return advantages, returns