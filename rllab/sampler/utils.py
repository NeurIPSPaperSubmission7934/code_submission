import numpy as np
from rllab.misc import tensor_utils
import time
import torch
from rllab.torch.models.base import TorchModel

def rollout(env, agent, max_path_length=np.inf, animated=False, speedup=1,
            always_return_paths=False):
    observations = []
    next_observations = []
    actions = []
    rewards = []
    agent_infos = []
    env_infos = []
    o = env.reset()
    agent.reset()
    path_length = 0
    if animated:
        env.render()
    while path_length < max_path_length:
        a, agent_info = agent.get_action(o)
        try:
            if agent.scale_action:  # check if action needs to be scaled
                if isinstance(env.action_space, Box):
                    # rescale the action
                    lb, ub = env.action_space.bounds
                    a = lb + (a + 1.) * 0.5 * (ub - lb)
                    a = np.clip(a, lb, ub)
        except AttributeError:
            pass
        next_o, r, d, env_info = env.step(a)
        observations.append(env.observation_space.flatten(o))
        next_observations.append(env.observation_space.flatten(next_o))
        rewards.append(r)
        actions.append(env.action_space.flatten(a))
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        if animated:
            env.render()
            timestep = 0.05
            time.sleep(timestep / speedup)
            print(o, r, a, next_o, d)
        path_length += 1
        if d:
            break
        o = next_o
    if animated and not always_return_paths:
        return

    return dict(
        observations=tensor_utils.stack_tensor_list(observations),
        actions=tensor_utils.stack_tensor_list(actions),
        rewards=tensor_utils.stack_tensor_list(rewards),
        next_observations=tensor_utils.stack_tensor_list(next_observations),
        agent_infos=tensor_utils.stack_tensor_dict_list(agent_infos),
        env_infos=tensor_utils.stack_tensor_dict_list(env_infos),
    )

from rllab.spaces.box import Box

def rollout_torch(env, agent, max_path_length=np.inf, animated=False, speedup=1,
            always_return_paths=False, extra_clip=False, terminate_only_max_path=False):
    observations = []
    next_observations = []
    normalized_observations=[]
    normalized_next_observations = []
    unscaled_actions = []
    actions = []
    rewards = []
    agent_infos = []
    env_infos = []
    mask=[]
    o = env.reset()
    try:
        agent.reset()
    except AttributeError:
        pass
    path_length = 0
    t=0

    def handle_obs(o):
        # get list with bools if output of env is normalized
        if isinstance(env, TorchModel):
            normalized_obs = env.normalized_output
        else:
            normalized_obs = [False] * len(o)

        unnormalized_idx = [i for i, x in enumerate(normalized_obs) if not x]
        normalized_idx = [i for i, x in enumerate(normalized_obs) if x]

        lb, ub = env.observation_space.bounds
        # normalize the unnormalized idx
        normalized_unnormalized_val = (2 * (o[unnormalized_idx] - lb[unnormalized_idx]) / (
                ub[unnormalized_idx] - lb[unnormalized_idx])) - 1
        normalized_unnormalized_val = np.clip(normalized_unnormalized_val, -1, 1)
        # unnormalize the normalized idx
        unnormalized_normalized_val = lb[normalized_idx] + (o[normalized_idx] + 1.) * 0.5 * (
                ub[normalized_idx] - lb[normalized_idx])
        unnormalized_normalized_val = np.clip(unnormalized_normalized_val, lb[normalized_idx], ub[normalized_idx])

        # put everything together
        normalized_obs = np.zeros(o.shape)
        normalized_obs[normalized_idx] = o[normalized_idx]
        normalized_obs[unnormalized_idx] = normalized_unnormalized_val
        unnormalized_obs = np.zeros(o.shape)
        unnormalized_obs[unnormalized_idx] = o[unnormalized_idx]
        unnormalized_obs[normalized_idx] = unnormalized_normalized_val
        # do extra clipping since original values could be out of bounds
        if extra_clip:
            normalized_obs = np.clip(normalized_obs, -1, 1)
            unnormalized_obs = np.clip(unnormalized_obs, lb, ub)

        # TODO: build own function for this
        # select the right observations for the agent
        normalized_policy_input = agent.normalized_input
        normalized_policy_input_idx = [i for i, x in enumerate(normalized_policy_input) if x]
        unnormalized_policy_input_idx = [i for i, x in enumerate(normalized_policy_input) if not x]

        policy_input = np.zeros(o.shape)
        policy_input[normalized_policy_input_idx] = normalized_obs[normalized_policy_input_idx]
        policy_input[unnormalized_policy_input_idx] = unnormalized_obs[unnormalized_policy_input_idx]
        agent_obs_torch_var = (torch.from_numpy(policy_input.astype(np.float32))).unsqueeze(0)

        # select the right observations for the env
        if isinstance(env, TorchModel):
            normalized_env_input = env.normalized_input_obs
        else:
            normalized_env_input = [False] * len(o)
        normalized_env_input_idx = [i for i, x in enumerate(normalized_env_input) if x]
        unnormalized_env_input_idx = [i for i, x in enumerate(normalized_env_input) if not x]
        env_input = np.zeros(o.shape)
        env_input[normalized_env_input_idx] = normalized_obs[normalized_env_input_idx]
        env_input[unnormalized_env_input_idx] = unnormalized_obs[unnormalized_env_input_idx]
        env_obs_torch_var = (torch.from_numpy(env_input.astype(np.float32)))

        return normalized_obs, unnormalized_obs, agent_obs_torch_var, env_obs_torch_var

    def handle_action(a):
        normalized_a = agent.normalized_output
        # scale only the normalized action outputs
        unnormalized_idx = [i for i, x in enumerate(normalized_a) if not x]
        normalized_idx = [i for i, x in enumerate(normalized_a) if x]

        lb, ub = env.action_space.bounds

        # normalize the unnormalized idx
        normalized_unnormalized_val = (2 * (a[unnormalized_idx] - lb[unnormalized_idx]) / (
                ub[unnormalized_idx] - lb[unnormalized_idx])) - 1
        normalized_unnormalized_val = np.clip(normalized_unnormalized_val, -1, 1)
        # unnormalize the normalized idx
        unnormalized_normalized_val = lb[normalized_idx] + (a[normalized_idx] + 1.) * 0.5 * (
                ub[normalized_idx] - lb[normalized_idx])
        unnormalized_normalized_val = np.clip(unnormalized_normalized_val, lb[normalized_idx], ub[normalized_idx])

        # put everything together
        normalized_a = np.zeros(a.shape)
        normalized_a[normalized_idx] = a[normalized_idx]
        normalized_a[unnormalized_idx] = normalized_unnormalized_val
        unnormalized_a = np.zeros(a.shape)
        unnormalized_a[unnormalized_idx] = a[unnormalized_idx]
        unnormalized_a[normalized_idx] = unnormalized_normalized_val

        # do extra clipping since original values could be out of bounds
        if extra_clip:
            normalized_a = np.clip(normalized_a, -1, 1)
            unnormalized_a = np.clip(unnormalized_a, lb, ub)

        unscaled_a = normalized_a
        action = unnormalized_a

        # select the right actions for the env
        if isinstance(env, TorchModel):
            normalized_env_input = env.normalized_input_a
        else:
            normalized_env_input = [False] * len(a)
        normalized_env_input_idx = [i for i, x in enumerate(normalized_env_input) if x]
        unnormalized_env_input_idx = [i for i, x in enumerate(normalized_env_input) if not x]
        env_input = np.zeros(a.shape)
        env_input[normalized_env_input_idx] = normalized_a[normalized_env_input_idx]
        env_input[unnormalized_env_input_idx] = unnormalized_a[unnormalized_env_input_idx]
        env_a_np_var = env_input

        return action, unscaled_a, env_a_np_var

    if animated:
        env.render()
    while path_length < max_path_length:
        # TODO: it might be the case that the env is not giving a numpy array
        normalized_o, o, agent_obs_torch, env_obs_torch = handle_obs(o)
        a, agent_info = agent.select_action(agent_obs_torch, t)
        #print(a, agent_obs_torch)
        a, unscaled_a, env_a_torch = handle_action(a)
        if isinstance(env, TorchModel):
            #print(env_a_torch, env_obs_torch, o)
            #print(a, unscaled_a, env_a_torch)
            next_orig_o, r, d, env_info = env.step(env_a_torch, env_obs_torch, o)
        else:
            next_orig_o, r, d, env_info = env.step(a)
        normalized_next_o, next_o, _, _ = handle_obs(next_orig_o)
        observations.append(env.observation_space.flatten(o))
        normalized_observations.append(env.observation_space.flatten(normalized_o))
        next_observations.append(next_o)
        normalized_next_observations.append(normalized_next_o)
        rewards.append(r)
        actions.append(env.action_space.flatten(a))
        unscaled_actions.append(env.action_space.flatten(unscaled_a))
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        if animated:
            env.render()
            timestep = 0.05
            time.sleep(timestep / speedup)
            print(o, r, a, next_o)
        path_length += 1
        if d and not terminate_only_max_path:
            mask.append(0)
            break
        elif path_length == max_path_length:
            mask.append(0) # add termination when we reached max time
            break
        elif not d:
            mask.append(1)
        else:
            mask.append(0)
        o = next_orig_o
        t += 1
    if animated:
        try:
            env.close()
        except AttributeError:
            pass
    if animated and not always_return_paths:
        return

    return dict(
        observations=tensor_utils.stack_tensor_list(observations),
        actions=tensor_utils.stack_tensor_list(actions),
        rewards=tensor_utils.stack_tensor_list(rewards),
        next_observations=tensor_utils.stack_tensor_list(next_observations),
        agent_infos=tensor_utils.stack_tensor_dict_list(agent_infos),
        env_infos=tensor_utils.stack_tensor_dict_list(env_infos),
        mask=tensor_utils.stack_tensor_list(mask),
        normalized_observations=tensor_utils.stack_tensor_list(normalized_observations),
        normalized_next_observations=tensor_utils.stack_tensor_list(normalized_next_observations),
        unscaled_actions=tensor_utils.stack_tensor_list(unscaled_actions),
    )