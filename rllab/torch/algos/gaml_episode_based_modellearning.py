# Copyright (c) 2018 Copyright holder of the paper Generative Adversarial Model Learning
# submitted to NeurIPS 2019 for review
# All rights reserved.

# inspired from multi agent gail repo

import rllab.misc.logger as logger
from rllab.algos.base import RLAlgorithm
import numpy as np
import torch
from rllab.misc.overrides import overrides
from rllab.torch.metrics.eval_metrics import TrajSampler
from rllab.torch.algos.reinforce.returns import compute_returns
from sklearn.metrics import accuracy_score
import sys
from tqdm import tqdm

"""
class which is used for GAML training of the dynamics model
"""
class GAMLEpisodeBasedModelLearning(RLAlgorithm):

    def __init__(self, policy, expert_data, model, discriminator, upperLevelPolicy, n_itr=500, n_traj=1, n_samples=5, max_path_length=500,
                 discount=0.99, current_itr=0, use_timesteps=False, use_state_diff_in_discriminator=False, discriminator_updates_per_itr=1):

        self.policy = policy
        # TODO: for now we don't use expert data since we don't train the discriminator
        self.expert_data = expert_data
        self.upperLevelPolicy = upperLevelPolicy
        self.model = model
        self.imitationModel = self.model
        self.discriminator = discriminator
        self.n_itr = n_itr
        self.current_itr = current_itr
        self.n_traj = n_traj
        self.max_path_length = max_path_length
        self.n_samples = n_samples
        self.discount = discount
        self.use_timesteps = use_timesteps
        self.use_state_diff_in_discriminator = use_state_diff_in_discriminator
        self.label_smoothing = True
        self.discriminator_updates_per_itr = discriminator_updates_per_itr
        if not hasattr(self.discriminator, 'update_not_needed'):
            self.discrim_optimizer = torch.optim.RMSprop(discriminator.parameters())
        # use our model as env
        self.sampler = TrajSampler(self.policy, self.model, self.n_traj, self.max_path_length, self.discount, useImitationEnv=False, useImitationPolicy=False)

    def start_worker(self):
        self.sampler.start_worker()

    def shutdown_worker(self):
        self.sampler.shutdown_worker()

    def create_torch_var_from_paths_for_discrim(self, data_paths, are_from_learned):
        # normalize the input to similar form as used in the model
        # TODO: add different normalization for discriminator?
        # TODO: maybe we can also do without copy()
        normalize_input_obs = self.model.normalized_input_obs
        normalize_input_a = self.model.normalized_input_a
        observations_np = data_paths["observations"].copy()[:,[0,1,3,4,5]]
        # normalized_input_obs_idx = [i for i, x in enumerate(normalize_input_obs) if x]
        # observations_np[:, normalized_input_obs_idx] = data_paths["normalized_observations"][:,
        #                                                normalized_input_obs_idx]
        actions_np = data_paths["actions"].copy()
        normalized_input_a_idx = [i for i, x in enumerate(normalize_input_a) if x]
        actions_np[:, normalize_input_a] = data_paths["unscaled_actions"][:, normalized_input_a_idx]

        if self.use_state_diff_in_discriminator:
            # we need to check if the data_paths are from the true env or the learned env, because they might be from
            # different scales --> we want to have the same scale as the output of the learned env in the end
            if are_from_learned:
                state_diff = data_paths["env_infos"]["obs_diff"]
            else:
                # rescale them to the output of the learned env by normalizing them to -1 and 1
                state_diff = data_paths["env_infos"]["obs_diff"].copy()
                normalize_output_state_diff = self.model.normalized_output_state_diff
                normalize_output_state_diff_idx = [i for i, x in enumerate(normalize_output_state_diff) if x]
                lb, ub = self.model.observation_space.bounds
                lb = lb[normalize_output_state_diff_idx]
                ub = ub[normalize_output_state_diff_idx]
                if len(normalize_output_state_diff_idx) > 0:
                    state_diff[normalize_output_state_diff_idx] = (2 * (
                                state_diff[normalize_output_state_diff_idx] - lb) / (
                                                                           ub - lb)) - 1
                    state_diff[normalize_output_state_diff_idx] = np.clip(state_diff[normalize_output_state_diff_idx],
                                                                          -1, 1)

            torch_input_batch = torch.cat([torch.from_numpy(observations_np).float(),
                                           torch.from_numpy(actions_np).float(),
                                           torch.from_numpy(state_diff).float()], dim=1)

            if self.use_timesteps:
                # reshape of timesteps is needed since it is only a 1D vector
                # no normalization needed since we have a simple discriminator
                timesteps = torch.from_numpy(data_paths["timesteps"]).view(-1, 1).float()
                print(timesteps)
                torch_input_batch = torch.cat([torch_input_batch, timesteps], dim=1)
        else:
            next_observations_np = data_paths["next_observations"].copy()[:,[0,1,3,4,5]]
            # next_observations_np[:, normalized_input_obs_idx] = data_paths["normalized_next_observations"][:,
            #                                                     normalized_input_obs_idx]

            torch_input_batch = torch.cat([torch.from_numpy(observations_np).float(),
                                           torch.from_numpy(actions_np).float(),
                                           torch.from_numpy(next_observations_np).float()], dim=1)
            if self.use_timesteps:
                # reshape of timesteps is needed since it is only a 1D vector
                # no normalization needed since we have a simple discriminator
                timesteps = torch.from_numpy(data_paths["timesteps"]/self.max_path_length).view(-1, 1).float()
                torch_input_batch = torch.cat([torch_input_batch, timesteps], dim=1)

        return torch_input_batch

    def create_torch_var_from_paths(self, expert_data):
        normalize_input_obs = self.imitationModel.normalized_input_obs
        normalize_input_a = self.imitationModel.normalized_input_a
        expert_observations_np = expert_data["observations"]
        normalized_input_obs_idx = [i for i, x in enumerate(normalize_input_obs) if x]
        expert_observations_np[:, normalized_input_obs_idx] = expert_data["normalized_observations"][:,
                                                              normalized_input_obs_idx]
        expert_actions_np = expert_data["actions"]
        normalized_input_a_idx = [i for i, x in enumerate(normalize_input_a) if x]
        expert_actions_np[:, normalize_input_a] = expert_data["unscaled_actions"][:, normalized_input_a_idx]
        torch_input_batch = torch.cat([torch.from_numpy(expert_observations_np).float(),
                                       torch.from_numpy(expert_actions_np).float()], dim=1)
        try:
            if self.imitationModel.pred_diff:
                # we assume that they are all unnormalized, since they come directly from the expert env
                expert_obs_diff_np = expert_data["env_infos"]["obs_diff"]
                # normalize them now as needed
                normalize_output_state_diff = self.imitationModel.normalized_output_state_diff
                lb, ub = self.imitationModel._wrapped_env.observation_space.bounds
                # select only the one we need to normalize
                normalized_idx = [i for i, x in enumerate(normalize_output_state_diff) if x]
                lb = lb[normalized_idx]
                ub = ub[normalized_idx]
                expert_obs_diff_np[:, normalized_idx] = (2 * (expert_obs_diff_np[:, normalized_idx] - lb) / (
                        ub - lb)) - 1
                expert_obs_diff_np[:, normalized_idx] = np.clip(expert_obs_diff_np[:, normalized_idx], -1, 1)
                torch_output_batch = torch.from_numpy(expert_obs_diff_np).float()
        except AttributeError:
            raise NotImplementedError("We cannot deal with envs with only next state predictions yet")

        return torch_input_batch, torch_output_batch

    def train(self):
        self.start_worker()

        expert_data = self.expert_data
        processsed_expert_data = self.sampler.process_samples(0, expert_data)

        self.expert_data_timestep_dictionary = {}
        # create a processed_expert_data dictionary for each timestep as key with only data of the given timestep
        expert_data_point_per_timestep = []
        for t in range(self.max_path_length):
            # get indices for the timestep
            indices = np.argwhere(processsed_expert_data["timesteps"] == t)
            timestep_dict = {}
            expert_data_point_per_timestep.append(indices.shape[0])
            for key, value in processsed_expert_data.items():
                if key == "paths" or key == "agent_infos" or key == "env_infos":
                    continue
                elif key == "observations" or key == "next_observations":
                    # TODO: for now we just hack the sin and cos for rotational states into this discriminator, but
                    # it would make sense to put this to another place such that it can be used also by the environment
                    costheta = np.cos(value[indices][:, :, [2]])
                    sintheta = np.sin(value[indices][:, :, [2]])
                    # we use squeeze since the indices selection gives us (#batchsize, 1, dim_action)
                    timestep_dict[key] = np.concatenate((value[indices], costheta, sintheta), axis=2).squeeze(1)
                else:
                    # we use squeeze since the indices selection gives us (#batchsize, 1, dim_value)
                    timestep_dict[key] = value[indices].squeeze(1)
            self.expert_data_timestep_dictionary[t] = timestep_dict


        with tqdm(total=self.n_itr, file=sys.stdout) as pbar:
            for itr in range(self.current_itr, self.n_itr+1):
        #for itr in tnrange(self.n_itr, desc="MLE training"):
                with logger.prefix('itr #%d | ' % itr):

                    # sample n parameters
                    model_parameters = self.upperLevelPolicy.sample_n(self.n_samples)
                    rewards=torch.zeros([self.n_samples])
                    real_rewards = torch.zeros([self.n_samples])
                    # list for collecting all the generated data and merging them into 1 large batch s.t. we can perform a large discriminator update
                    all_generated_paths = []

                    avgTrajLength = 0
                    # evaluate the sampled parameters by doing rollouts and using the discriminator
                    for model_param, i in zip(model_parameters, range(self.n_samples)):
                        generated_paths = self.sampler.obtain_samples(itr, env_params=model_param)
                        generated_data = self.sampler.process_samples(itr, generated_paths)
                        if hasattr(self.discriminator, 'custom_discriminator'):
                            generated_rewards = self.discriminator.surrogate_reward(generated_data)
                        else:
                            # TODO: hack for cos and sin
                            costheta = np.cos(generated_data["observations"][:,[2]])
                            sintheta = np.sin(generated_data["observations"][:,[2]])
                            generated_data["observations"] = np.concatenate((generated_data["observations"], costheta, sintheta), axis=1)
                            costheta = np.cos(generated_data["next_observations"][:,[2]])
                            sintheta = np.sin(generated_data["next_observations"][:,[2]])
                            generated_data["next_observations"] = np.concatenate((generated_data["next_observations"], costheta, sintheta), axis=1)

                            # compute surrogate reward using the discriminator
                            discrim_input = self.create_torch_var_from_paths_for_discrim(generated_data, are_from_learned=True)
                            # input for the discriminator
                            generated_rewards = self.discriminator.surrogate_reward(discrim_input)
                        masks = torch.from_numpy(generated_data["masks"]).float()
                        returns = compute_returns(generated_rewards, masks, discount=self.discount, returns4traj=True)
                        # now get only 1 value per trajectory and not per state-action pair and take mean over traj
                        reward = torch.mean(returns[np.where(generated_data["masks"] == 0)])
                        rewards[i] = reward
                        # compute rewardof the task performance
                        real_returns = compute_returns(torch.from_numpy(generated_data["rewards"]).float(), masks, discount=self.discount, returns4traj=True)
                        real_reward = torch.mean(real_returns[np.where(generated_data["masks"] == 0)])
                        real_rewards[i] = real_reward
                        all_generated_paths = all_generated_paths + generated_paths
                        avgTrajLength += np.mean([np.argmax(path["mask"] == 0) for path in generated_data["paths"]])

                    logger.log("surrogate discounted_surrogate_returns:"+ str(rewards))
                    logger.log("real discounted_returns:" + str(real_rewards))
                    logger.log("model_parameters:" + str(model_parameters))

                    logger.record_tabular("avg_discounted_surrogate_returns", np.mean(np.array(rewards)))
                    logger.record_tabular("avg_discounted_real_returns", np.mean(np.array(real_rewards)))
                    logger.record_tabular("avg_traj_length", avgTrajLength/self.n_samples)

                    # do weighted maximum likelihood to compute new parameters of the gaussian upperLevelPolicy
                    # weighted mean
                    mean = torch.sum(rewards.reshape(-1, 1)*model_parameters, dim=0)/rewards.sum()

                    # weighted covariance
                    theta_minus_mats = torch.bmm((model_parameters - mean).unsqueeze(2), (model_parameters - mean).unsqueeze(2).transpose(1, 2))
                    covariance = torch.sum(rewards.reshape(-1, 1, 1) * theta_minus_mats, dim=0)/rewards.sum()
                    if itr > 0: # skip update in itr 0 since, we don't have any trained discriminator yet
                        self.upperLevelPolicy.set_parameters(mean=mean, covariance=covariance)

                    logger.log("mean:" + str(mean))
                    logger.log("covariance:" + str(covariance))

                    logger.record_tabular("mean", mean)
                    logger.record_tabular("covariance", covariance)

                    ############ do now the discriminator update
                    if not hasattr(self.discriminator, 'update_not_needed'):
                        all_generated_data = self.sampler.process_samples(itr, all_generated_paths)

                        generated_data_timestep_dictionary = {}
                        generated_data_point_per_timestep = []
                        # create a processed_expert_data dictionary for each timestep as key with only data of the given timestep
                        for t in range(self.max_path_length):
                            # get indices for the timestep
                            indices = np.argwhere(all_generated_data["timesteps"] == t)
                            generated_data_point_per_timestep.append(indices.shape[0])
                            timestep_dict = {}
                            for key, value in all_generated_data.items():
                                if key == "paths" or key == "agent_infos":
                                    continue
                                elif key == "env_infos":
                                    # here we need to go one layer deeper since it is a dict
                                    dictionary = {}
                                    for key_dictionary, value_dictionary in value.items():
                                        dictionary[key_dictionary] = value_dictionary[indices].squeeze(1)
                                    timestep_dict[key] = dictionary
                                elif key == "observations" or key == "next_observations":
                                    # TODO: for now we just hack the sin and cos for rotational states into this discriminator, but
                                    # it would make sense to put this to another place such that it can be used also by the environment
                                    costheta = np.cos(value[indices][:, :, [2]])
                                    sintheta = np.sin(value[indices][:, :, [2]])
                                    # we use squeeze since the indices selection gives us (#batchsize, 1, dim_action)
                                    timestep_dict[key] = np.concatenate((value[indices], costheta, sintheta), axis=2).squeeze(1)
                                else:
                                    # we use squeeze since the indices selection gives us (#batchsize, 1, dim_value)
                                    timestep_dict[key] = value[indices].squeeze(1)
                            generated_data_timestep_dictionary[t] = timestep_dict

                        self.batch_optimize_discriminator(itr,
                                                          self.expert_data_timestep_dictionary,
                                                          generated_data_timestep_dictionary,
                                                          expert_data_point_per_timestep,
                                                          generated_data_point_per_timestep, self.max_path_length)

                    logger.dump_tabular(with_prefix=False)

                    pbar.set_description('iteration: %d' % (1 + itr))
                    pbar.update(1)
                    ############ check early stopping

                    # variant 2 upper level policy converged --> check if covariance matrix entries are all smaller than 1e-6
                    if torch.max(covariance) < 1e-6:
                        pbar.close()
                        print("upper level policy converged...")
                        print("saving results")
                        logger.log("upper level policy converged...")
                        params, torch_params = self.get_itr_snapshot(itr, generated_data)
                        params["algo"] = self
                        logger.log("saving best parameters")
                        logger.save_itr_params(self.n_itr, params, torch_params)
                        logger.log("saved")
                        break

                    params, torch_params = self.get_itr_snapshot(itr, generated_data)
                    params["algo"] = self
                    logger.save_itr_params(itr, params, torch_params)
                    logger.log("saved")

    def batch_optimize_discriminator(self, itr, expert_data_per_timestep_dict, generated_data_per_timestep_dict,
                                     numDataPointsExpert_per_ts, numDataPointsGenerated_per_ts, current_max_path_length):

        discriminator = self.discriminator
        discrim_optimizer = self.discrim_optimizer

        if hasattr(discriminator, 'update_not_needed'):
            return

        generated_inputs_all_ts = []
        expert_inputs_all_ts = []
        for t in range(current_max_path_length):
            expert_data = expert_data_per_timestep_dict[t]
            generated_data = generated_data_per_timestep_dict[t]
            numDataPointsExpert = numDataPointsExpert_per_ts[t]
            numDataPointsGenerated = numDataPointsGenerated_per_ts[t]
            if numDataPointsGenerated < 10:
                # don't train discriminator if we have less than 10 samples
                continue
            expert_inputs = self.create_torch_var_from_paths_for_discrim(expert_data, are_from_learned=False)
            generated_inputs = self.create_torch_var_from_paths_for_discrim(generated_data, are_from_learned=True)
            # sub-sample inputs, s.t. we have a even number between expert and generated transitions
            if numDataPointsGenerated > numDataPointsExpert:
                generated_inputs = generated_inputs[0:numDataPointsExpert]
            else:
                idx = np.random.permutation(
                    numDataPointsExpert)  # permute the idx, s.t. we don't get the same one every time
                expert_inputs = expert_inputs[idx[0:numDataPointsGenerated]]
            generated_inputs_all_ts.append(generated_inputs)
            expert_inputs_all_ts.append(expert_inputs)

        # now merge them all together
        expert_inputs = torch.cat(expert_inputs_all_ts)
        generated_inputs = torch.cat(generated_inputs_all_ts)

        # calculate loss
        # TODO: should we work with minibatches or take the whole batch for training the classifier
        discrim_optimizer.zero_grad()
        # TODO: we might want to change to BCEWithLogitsLoss to be more numerical stable, in case we work with a sigmoid
        loss_fn = torch.nn.BCEWithLogitsLoss()

        expert_label = torch.ones((expert_inputs.shape[0], 1))  # expert labels are 1
        generated_label = torch.zeros((generated_inputs.shape[0], 1))  # generated trajectory labels are 0

        if self.label_smoothing:
            # try out one-sided label smoothing
            expert_label *= 0.9

        for update_step in range(self.discriminator_updates_per_itr):
            expert_pred_score = discriminator.forward_logits(expert_inputs)
            generated_pred_score = discriminator.forward_logits(generated_inputs)
            expert_pred = discriminator.forward(expert_inputs)
            generated_pred = discriminator.forward(generated_inputs)

            discrim_loss = loss_fn(expert_pred_score, expert_label) + loss_fn(generated_pred_score, generated_label)
            discrim_loss.backward()
            # if update_step == 0:
            #     logger.record_tabular('loss_before_train', discrim_loss.item())
            #     logger.record_tabular('avg_pred_expert_before_train', torch.mean(expert_pred).detach().numpy())
            #     logger.record_tabular('avg_pred_generated_before_train', torch.mean(generated_pred).detach().numpy())
            discrim_optimizer.step()

        ### computation for losses after training
        expert_label = torch.ones((expert_inputs.shape[0], 1))  # expert labels are 1
        generated_label = torch.zeros((generated_inputs.shape[0], 1))  # generated trajectory labels are 0
        expert_pred_score = discriminator.forward_logits(expert_inputs)
        generated_pred_score = discriminator.forward_logits(generated_inputs)
        expert_pred = discriminator.forward(expert_inputs)
        generated_pred = discriminator.forward(generated_inputs)
        discrim_loss = loss_fn(expert_pred_score, expert_label) + loss_fn(generated_pred_score, generated_label)
        # logger.record_tabular('loss_after_train', discrim_loss.item())
        # logger.record_tabular('avg_pred_expert_after_train', torch.mean(expert_pred).detach().numpy())
        # logger.record_tabular('avg_pred_generated_after_train', torch.mean(generated_pred).detach().numpy())

        all_output = torch.cat([torch.round(generated_pred), torch.round(expert_pred)]).data.numpy()
        all_labels = torch.cat([generated_label, expert_label]).data.numpy()

        # calculate accuracy using sklearn
        accuracy = accuracy_score(all_output, all_labels)
        logger.log("accuracy:" +str(accuracy))
        logger.record_tabular("accuracy", accuracy)

    @overrides
    def get_itr_snapshot(self, itr, samples_data):
        return dict(
            itr=itr,
            policy=self.policy,
            imitationEnv=self.model,
            upperLevelPolicy=self.upperLevelPolicy,
        ),  dict(imitationEnv=self.model,)




