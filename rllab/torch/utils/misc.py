# Copyright (c) 2018 Copyright holder of the paper Generative Adversarial Model Learning
# submitted to NeurIPS 2019 for review
# All rights reserved.


import torch
import numpy as np

def create_torch_var_from_paths(data_paths):
    observations = torch.from_numpy(data_paths["observations"]).float()
    actions = torch.from_numpy(data_paths["actions"]).float()
    next_observations = torch.from_numpy(data_paths["next_observations"]).float()
    return observations, actions, next_observations

def batch_diagonal(input):
    # code taken from: https://github.com/pytorch/pytorch/issues/12160
    # batches a stack of vectors (batch x N) -> a stack of diagonal matrices (batch x N x N)
    # works in  2D -> 3D, should also work in higher dimensions
    # make a zero matrix, which duplicates the last dim of input
    dims = [input.size(i) for i in torch.arange(input.dim())]
    dims.append(dims[-1])
    output = torch.zeros(dims)
    # stride across the first dimensions, add one to get the diagonal of the last dimension
    strides = [output.stride(i) for i in torch.arange(input.dim() - 1 )]
    strides.append(output.size(-1) + 1)
    # stride and copy the imput to the diagonal
    output.as_strided(input.size(), strides ).copy_(input)
    return output

def paths_to_fitting_env_normalized_torch_input(env, processed_paths):
    normalized_env_input_obs = env.normalized_input_obs
    normalized_env_input_obs_idx = [i for i, x in enumerate(normalized_env_input_obs) if x]
    obs = processed_paths["observations"].copy()
    obs[:, normalized_env_input_obs_idx] = processed_paths["normalized_observations"][:, normalized_env_input_obs_idx]

    normalized_env_input_action = env.normalized_input_a
    normalized_env_input_action_idx = [i for i, x in enumerate(normalized_env_input_action) if x]
    actions = processed_paths["actions"].copy()
    actions[:, normalized_env_input_action_idx] = processed_paths["unscaled_actions"][:, normalized_env_input_action_idx]

    input = np.concatenate([obs, actions], axis=1)
    return torch.from_numpy(input).float()

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")