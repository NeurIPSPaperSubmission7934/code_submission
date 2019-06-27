# Copyright (c) 2018 Copyright holder of the paper Generative Adversarial Model Learning
# submitted to NeurIPS 2019 for review
# All rights reserved.

import torch
import numpy as np
from rllab.sampler.base import Sampler
from rllab.sampler import parallel_sampler
from rllab.misc import special
from rllab.misc import tensor_utils
from rllab.torch.utils.misc import create_torch_var_from_paths
from rllab.torch.models.policies.base import ActionReplayPolicy

class TrajSampler(Sampler):
    def __init__(self, policy, env, n_traj, max_path_length, discount=1, useImitationEnv=False, useImitationPolicy=False, terminate_only_max_path=False):
        """
        :type algo: BatchPolopt
        """
        #TODO: change whole sampler s.t. useImitationEnv and useImitationPolicy is not needed anymore
        self.policy = policy
        self.env = env
        self.n_traj = n_traj
        self.max_path_length=max_path_length
        self.scope = None
        self.imitationEnv = None
        self.imitationPolicy = None
        self.useImitationEnv = useImitationEnv
        self.useImitationPolicy = useImitationPolicy
        self.discount = discount
        if useImitationEnv:
            self.imitationEnv = env
        if useImitationPolicy:
            self.imitationPolicy = policy
        self.terminate_only_max_path = terminate_only_max_path

    def start_worker(self, use_furuta_controller=False):
        parallel_sampler.populate_task(self.env, self.policy, scope=self.scope, imitationPolicy=self.imitationPolicy, imitationEnv=self.imitationEnv,
                                       use_furuta_controller=use_furuta_controller)

    def shutdown_worker(self):
        parallel_sampler.terminate_task(scope=self.scope)

    def obtain_samples(self, itr, env_params=None):
        try:
            cur_params = self.policy.get_param_values()
        except AttributeError:
            cur_params = None
        paths = parallel_sampler.sample_paths(
            policy_params=cur_params,
            max_samples=self.n_traj,
            max_path_length=self.max_path_length,
            scope=self.scope,
            useImitationEnv=self.useImitationEnv,
            useImitationPolicy=self.useImitationPolicy,
            count_traj=True,
            terminate_only_max_path=self.terminate_only_max_path,
            env_params=env_params
        )
        # truncate the paths if we collected more than self.n_traj
        return paths[:self.n_traj]

    def process_samples(self, itr, paths):
        returns = []

        for idx, path in enumerate(paths):
            path["returns"] = special.discount_cumsum(path["rewards"]*path["mask"], self.discount)
            returns.append(path["returns"])

        observations = tensor_utils.concat_tensor_list([path["observations"] for path in paths])
        next_observations = tensor_utils.concat_tensor_list([path["next_observations"] for path in paths])
        actions = tensor_utils.concat_tensor_list([path["actions"] for path in paths])
        rewards = tensor_utils.concat_tensor_list([path["rewards"] for path in paths])
        returns = tensor_utils.concat_tensor_list([path["returns"] for path in paths])
        env_infos = tensor_utils.concat_tensor_dict_list([path["env_infos"] for path in paths])
        agent_infos = tensor_utils.concat_tensor_dict_list([path["agent_infos"] for path in paths])
        timesteps = tensor_utils.concat_tensor_list([np.arange(len(path["observations"])) for path in paths])
        normalized_observations = tensor_utils.concat_tensor_list([path["normalized_observations"] for path in paths])
        normalized_next_observations = tensor_utils.concat_tensor_list([path["normalized_next_observations"] for path in paths])
        unscaled_actions = tensor_utils.concat_tensor_list([path["unscaled_actions"] for path in paths])
        masks = tensor_utils.concat_tensor_list([path["mask"] for path in paths])

        samples_data = dict(
            observations=observations,
            next_observations=next_observations,
            actions=actions,
            rewards=rewards,
            returns=returns,
            env_infos=env_infos,
            agent_infos=agent_infos,
            paths=paths,
            timesteps=timesteps,
            normalized_observations=normalized_observations,
            normalized_next_observations=normalized_next_observations,
            unscaled_actions=unscaled_actions,
            masks=masks,
        )
        return samples_data

    def calc_avg_traj_length(self, processed_paths):
        return np.mean([np.argmax(path["mask"] == 0) for path in processed_paths["paths"]])

    def calc_avg_undiscounted_return(self,processed_paths):
        return np.mean([sum(path["rewards"]*path["mask"]) for path in processed_paths["paths"]])

    def calc_avg_discounted_return(self, processed_paths):
        return np.mean([path["returns"][0] for path in processed_paths["paths"]])


class FixedActionTrajSampler(TrajSampler):
    def __init__(self, action_sequences, env, n_traj, discount=1):
        self.action_sequences = action_sequences
        self.env = env
        self.n_traj = n_traj
        self.discount = discount
        self.scope = None
        self.policy = ActionReplayPolicy(env.observation_space.flat_dim, env.action_space.flat_dim)

    def start_worker(self):
        parallel_sampler.populate_task(self.env, self.policy, scope=self.scope)

    def shutdown_worker(self):
        parallel_sampler.terminate_task(scope=self.scope)

    def obtain_samples(self, itr, env_params=None):
        all_paths = []
        for action_sequence in self.action_sequences:
            paths = parallel_sampler.sample_paths(
                policy_params=action_sequence,
                max_samples=self.n_traj,
                max_path_length=action_sequence.shape[0],
                scope=self.scope,
                count_traj=True,
                terminate_only_max_path=True,
                env_params=env_params
            )
            # truncate the paths if we collected more than self.n_traj
            all_paths += paths[:self.n_traj]
        return all_paths

""" returns the loglikelihood of the given paths under the given model"""
def calc_loglikelihood_traj(processed_paths, model):
    # calculate the loglikelihood of all paths
    observations, actions, new_observations = create_torch_var_from_paths(processed_paths)
    inputs = torch.cat([observations, actions], dim=1)
    loglike = model.get_log_prob(inputs, new_observations)
    return loglike



def create_timestep_dict_from_path(path, max_path_length, modeltype, max_expert_obs=None, min_expert_obs=None):
    data_ts_dictionary = {}
    # create a processed_expert_data dictionary for each timestep as key with only data of the given timestep
    datapoint_per_ts = []
    for t in range(max_path_length):
        # get indices for the timestep
        indices = np.argwhere(path["timesteps"] == t)
        timestep_dict = {}
        datapoint_per_ts.append(indices.shape[0])
        for key, value in path.items():
            if key == "paths" or key == "agent_infos" or key == "env_infos":
                continue
            elif key == "observations" or key == "next_observations":
                if modeltype == "CartPole":
                    # TODO: for now we just hack the sin and cos for rotational states into this discriminator, but
                    # it would make sense to put this to another place such that it can be used also by the environment
                    costheta = np.cos(value[indices][:, :, [2]])
                    sintheta = np.sin(value[indices][:, :, [2]])

                    if max_expert_obs is None and min_expert_obs is None:
                        normalized_states = value[indices][:,:,[0, 1, 3]]
                    else:
                        normalizer = 1 / (max_expert_obs - min_expert_obs)
                        normalized_states = 2 * (value[indices][:,:,[0, 1, 3]] - min_expert_obs[[0, 1, 3]]) * normalizer[[0, 1, 3]] - 1

                    # we use squeeze since the indices selection gives us (#batchsize, 1, dim_action)
                    timestep_dict[key] = np.concatenate((normalized_states, costheta, sintheta), axis=2).squeeze(1)
                elif modeltype =='Furuta':
                    costheta = np.cos(value[indices][:, :, [0]])
                    sintheta = np.sin(value[indices][:, :, [0]])
                    cosalpha = np.cos(value[indices][:, :, [1]])
                    sinalpha = np.sin(value[indices][:, :, [1]])

                    if max_expert_obs is None and min_expert_obs is None:
                        normalized_states = value[indices][:,:,[2, 3]]
                    else:
                        normalizer = 1 / (max_expert_obs - min_expert_obs)
                        normalized_states = 2 * (value[indices][:,:,[2, 3]] - min_expert_obs[[2, 3]]) * normalizer[[2, 3]] - 1

                    # we use squeeze since the indices selection gives us (#batchsize, 1, dim_action)
                    timestep_dict[key] = np.concatenate((normalized_states, costheta, sintheta, cosalpha, sinalpha),
                                                        axis=2).squeeze(1)
            else:
                # we use squeeze since the indices selection gives us (#batchsize, 1, dim_value)
                timestep_dict[key] = value[indices].squeeze(1)
        data_ts_dictionary[t] = timestep_dict

    return data_ts_dictionary, datapoint_per_ts

def calc_avg_displacement_timesteps(processed_expert_paths, processed_paths, max_path_length, modeltype, mode='avg', normalize=False):

    if normalize:
        # used for nomalization
        expert_obs = processed_expert_paths["observations"]
        max_expert_obs = np.max(expert_obs, axis=0)
        min_expert_obs = np.min(expert_obs, axis=0)
        # put first all trajectories in timestep-based dict and then calc the displacement for each timestep individually
        expert_data_timestep_dictionary, expert_dp_per_ts = create_timestep_dict_from_path(processed_expert_paths,
                                                                                           max_path_length, modeltype,
                                                                                           max_expert_obs, min_expert_obs)
        generated_data_timestep_dictionary, generated_dp_per_ts = create_timestep_dict_from_path(processed_paths,
                                                                                                 max_path_length,
                                                                                                 modeltype,
                                                                                                 max_expert_obs,
                                                                                                 min_expert_obs)
    else:
        expert_data_timestep_dictionary, expert_dp_per_ts = create_timestep_dict_from_path(processed_expert_paths,
                                                                                           max_path_length, modeltype)
        generated_data_timestep_dictionary, generated_dp_per_ts = create_timestep_dict_from_path(processed_paths,
                                                                                                 max_path_length,
                                                                                                 modeltype)
    displacements = np.zeros(max_path_length)

    for t in range(max_path_length):
        # do now the broadcasting trick
        states_expert = np.expand_dims(expert_data_timestep_dictionary[t]["observations"], axis=1)
        states_generated = np.expand_dims(generated_data_timestep_dictionary[t]["observations"], axis=0)

        if mode == 'avg':
            difference = np.sum(np.sqrt(np.sum(np.square(states_expert-states_generated), axis=2))) / (expert_dp_per_ts[t]*generated_dp_per_ts[t])
        elif mode == 'min':
            difference = np.sum(np.min(np.sqrt(np.sum(np.square(states_expert-states_generated), axis=2)), axis=0)) / (expert_dp_per_ts[t]*generated_dp_per_ts[t])
        else:
            raise NotImplementedError()
        displacements[t] = difference
    
    return displacements

def calc_leaving_boundaries_rate(processed_paths):
    # count number of trajectories that are terminating earlier
    terminate_earlier = 0
    for path in processed_paths["paths"]:
        terminate_step = np.argmax(path["mask"] == 0)
        if terminate_step < len(path["mask"])-2:
            terminate_earlier += 1

    return terminate_earlier / len(processed_paths["paths"])

def calc_success_rate(processed_paths):
    # count number of trajectories where we are able to balance the pole for at least 200 timesteps
    success = 0

    def search_sequence_numpy(arr, seq):
        """ Find sequence in an array using NumPy only.

        Parameters
        ----------
        arr    : input 1D array
        seq    : input 1D array

        Output
        ------
        Output : 1D Array of indices in the input array that satisfy the
        matching of input sequence in the input array.
        In case of no match, an empty list is returned.
        """

        # Store sizes of input array and sequence
        Na, Nseq = arr.size, seq.size

        # Range of sequence
        r_seq = np.arange(Nseq)

        # Create a 2D array of sliding indices across the entire length of input array.
        # Match up with the input sequence & get the matching starting indices.
        M = (arr[np.arange(Na - Nseq + 1)[:, None] + r_seq] == seq).all(1)

        # Get the range of those indices as final output
        if M.any() > 0:
            return np.where(np.convolve(M, np.ones((Nseq), dtype=int)) > 0)[0]
        else:
            return []  # No match found

    for path in processed_paths["paths"]:
        bool_array = path["rewards"] > 0.9
        indices = search_sequence_numpy(bool_array, np.array([True]*200))
        terminate_step = np.argmax(path["mask"] == 0)
        if len(indices) > 0 and not terminate_step < len(path["mask"])-2:
            success += 1

    return success / len(processed_paths["paths"])

def calc_success_rate_furuta(processed_paths):
    # count number of trajectories where we are able to balance the pole for at least 5 seconds and where the swingup
    # has to be done in 8 seconds
    success = 0

    def search_sequence_numpy(arr, seq):
        """ Find sequence in an array using NumPy only.

        Parameters
        ----------
        arr    : input 1D array
        seq    : input 1D array

        Output
        ------
        Output : 1D Array of indices in the input array that satisfy the
        matching of input sequence in the input array.
        In case of no match, an empty list is returned.
        """

        # Store sizes of input array and sequence
        Na, Nseq = arr.size, seq.size

        # Range of sequence
        r_seq = np.arange(Nseq)

        # Create a 2D array of sliding indices across the entire length of input array.
        # Match up with the input sequence & get the matching starting indices.
        M = (arr[np.arange(Na - Nseq + 1)[:, None] + r_seq] == seq).all(1)

        # Get the range of those indices as final output
        if M.any() > 0:
            return np.where(np.convolve(M, np.ones((Nseq), dtype=int)) > 0)[0]
        else:
            return []  # No match found

    for path in processed_paths["paths"]:
        bool_array = np.cos(path["observations"][:,1]) > 0.9
        indices = search_sequence_numpy(bool_array, np.array([True]*500))
        if len(indices) > 0:
            success += 1

    return success / len(processed_paths["paths"])
