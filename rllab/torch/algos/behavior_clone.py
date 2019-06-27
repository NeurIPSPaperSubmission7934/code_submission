# Copyright (c) 2018 Copyright holder of the paper Generative Adversarial Model Learning
# submitted to NeurIPS 2019 for review
# All rights reserved.

import torch
from rllab.algos.base import Algorithm
from rllab.misc.overrides import overrides
import rllab.misc.logger as logger
import numpy as np
from rllab.torch.utils import torch as torch_utils
from rllab.dynamic_models.cartpole_model import CartPoleModel
import scipy
from tqdm import tqdm
import sys

"""
class which is used for behavior cloning to imitate a expert policy or a environment
"""
class BehaviorCloning(Algorithm):

    def __init__(self, expert_data, imitation_model, n_itr ,mini_batchsize=1000, weight_decay=0, mode="imitate_env", optim=torch.optim.Adam):
        self.imitationModel = imitation_model
        self.expert_data = expert_data
        if optim is not None:
            self.optimizer = optim(imitation_model.parameters(), weight_decay=weight_decay)
        else:
            self.optimizer = None
        self.mode = mode
        self.mini_batchsize = mini_batchsize
        self.n_itr = n_itr
        self.l2_reg = weight_decay

    def create_torch_var_from_paths(self, expert_data):
        if self.mode == "imitate_env":
            normalize_input_obs = self.imitationModel.normalized_input_obs
            normalize_input_a = self.imitationModel.normalized_input_a
            expert_observations_np = expert_data["observations"]
            normalized_input_obs_idx = [i for i, x in enumerate(normalize_input_obs) if x]
            expert_observations_np[:, normalized_input_obs_idx] = expert_data["normalized_observations"][:, normalized_input_obs_idx]
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
                    lb , ub = self.imitationModel._wrapped_env.observation_space.bounds
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
        elif self.mode == "imitate_policy":
            normalize_input = self.imitationModel.normalized_input
            normalize_output = self.imitationModel.normalized_output
            normalized_input_idx = [i for i, x in enumerate(normalize_input) if x]
            normalized_output_idx = [i for i, x in enumerate(normalize_output) if x]
            expert_observations_np = expert_data["observations"]
            expert_observations_np[normalized_input_idx] = expert_data["normalized_observations"][normalized_input_idx]
            expert_actions_np = expert_data["actions"]
            expert_actions_np[normalized_output_idx] = expert_data["unscaled_actions"][normalized_output_idx]
            torch_input_batch = torch.from_numpy(expert_observations_np).float()
            torch_output_batch = torch.from_numpy(expert_actions_np).float()
        else:
            raise ValueError("invalid mode")

        return torch_input_batch, torch_output_batch

    def train(self):
        if self.optimizer is not None:
            self._train_SGD()
        else:
            self._train_BGFS()

    def _train_SGD(self):

        # TODO: we need to get here the right observations, actions and next_observations for the model
        # expert_observations, expert_actions, expert_next_observations = create_torch_var_from_paths(self.expert_data)
        # now train imitation policy using collect batch of expert_data with MLE on log prob since we have a Gaussian
        # TODO: do we train mean and variance? or only mean

        torch_input_batch, torch_output_batch = self.create_torch_var_from_paths(self.expert_data)

        # split data randomly into training and validation set, let's go with 70 - 30 split
        numTotalSamples = torch_input_batch.size(0)
        trainingSize = int(numTotalSamples*0.7)
        randomIndices = np.random.permutation(np.arange(numTotalSamples))
        trainingIndices = randomIndices[:trainingSize]
        validationIndices = randomIndices[trainingSize:]
        validation_input_batch = torch_input_batch[validationIndices]
        validation_output_batch = torch_output_batch[validationIndices]
        torch_input_batch = torch_input_batch[trainingIndices]
        torch_output_batch = torch_output_batch[trainingIndices]

        best_loss = np.inf
        losses = np.array([best_loss] * 25)
        with tqdm(total=self.n_itr, file=sys.stdout) as pbar:
            for epoch in range(self.n_itr+1):
                with logger.prefix('epoch #%d | ' % epoch):
                    # split into mini batches for training
                    total_batchsize = torch_input_batch.size(0)

                    logger.record_tabular('Iteration', epoch)
                    indices = np.random.permutation(np.arange(total_batchsize))
                    if isinstance(self.imitationModel, CartPoleModel):
                        logger.record_tabular("theta", str(self.imitationModel.theta.detach().numpy()))
                        logger.record_tabular("std", str(self.imitationModel.std.detach().numpy()))
                    # go through the whole batch
                    for k in range(int(total_batchsize/self.mini_batchsize)):
                        idx = indices[self.mini_batchsize*k:self.mini_batchsize*(k+1)]
                        # TODO: how about numerical stability?

                        log_prob = self.imitationModel.get_log_prob(torch_input_batch[idx, :], torch_output_batch[idx, :])

                        # note that L2 regularization is in weight decay of optimizer
                        loss = -torch.mean(log_prob) # negative since we want to minimize and not maximize
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()

                    # calculate the loss on the whole batch
                    log_prob = self.imitationModel.get_log_prob(validation_input_batch, validation_output_batch)
                    loss = -torch.mean(log_prob)
                    # Note: here we add L2 regularization to the loss to log the proper loss
                    # weight decay
                    for param in self.imitationModel.parameters():
                        loss += param.pow(2).sum() * self.l2_reg
                    logger.record_tabular("loss", loss.item())

                    # check if loss has decreased in the last 25 itr on the validation set, if not stop training
                    # and return the best found parameters
                    losses[1:] = losses[0:-1]
                    losses[0] = loss

                    if epoch == 0:
                        best_loss = np.min(losses)
                        best_flat_parameters = torch_utils.get_flat_params_from(self.imitationModel).detach().numpy()
                        logger.record_tabular("current_best_loss", best_loss)
                    elif np.min(losses) <= best_loss and not (np.mean(losses) == best_loss): #second condition prevents same error in whole losses
                        # set best loss to new one if smaller or keep it
                        best_loss = np.min(losses)
                        best_flat_parameters = torch_utils.get_flat_params_from(self.imitationModel).detach().numpy()
                        logger.record_tabular("current_best_loss", best_loss)
                    else:
                        pbar.close()
                        print("best loss did not decrease in last 25 steps")
                        print("saving best result...")
                        logger.log("best loss did not decrease in last 25 steps")
                        torch_utils.set_flat_params_to(self.imitationModel, torch_utils.torch.from_numpy(best_flat_parameters))
                        logger.log("SGD converged")
                        logger.log("saving best result...")
                        params, torch_params = self.get_itr_snapshot(epoch)
                        if not params is None:
                            params["algo"] = self
                        logger.save_itr_params(self.n_itr, params, torch_params)
                        logger.log("saved")
                        break

                    pbar.set_description('epoch: %d' % (1 + epoch))
                    pbar.update(1)

                # save result
                logger.log("saving snapshot...")
                params, torch_params = self.get_itr_snapshot(epoch)
                if not params is None:
                    params["algo"] = self
                logger.save_itr_params(epoch, params, torch_params)
                logger.log("saved")
                logger.dump_tabular(with_prefix=False)

    def _train_BGFS(self):
        if not isinstance(self.imitationModel, CartPoleModel):
            raise NotImplementedError("train BGFS can be only called with CartPoleModel")
        expert_observations = torch.from_numpy(self.expert_data["observations"]).float()
        expert_actions = torch.from_numpy(self.expert_data["actions"]).float()
        expert_obs_diff = torch.from_numpy(self.expert_data["env_infos"]["obs_diff"]).float()
        # now train imitation policy using collect batch of expert_data with MLE on log prob since we have a Gaussian
        # TODO: do we train mean and variance? or only mean

        if self.mode == "imitate_env":
            input = torch.cat([expert_observations, expert_actions], dim=1)
            output = expert_obs_diff
        else:
            return ValueError("invalid mode")

        imitation_model = self.imitationModel
        total_batchsize = input.size(0)

        def get_negative_likelihood_loss(flat_params):
            torch_utils.set_flat_params_to(imitation_model, torch_utils.torch.from_numpy(flat_params))
            for param in imitation_model.parameters():
                if param.grad is not None:
                    param.grad.data.fill_(0)

            indices = np.random.permutation(np.arange(total_batchsize))

            loss = - torch.mean(imitation_model.get_log_prob(input[indices[:self.mini_batchsize]], output[indices[:self.mini_batchsize]]))

            # weight decay
            for param in imitation_model.parameters():
                loss += param.pow(2).sum() * self.l2_reg
            loss.backward()

            # FIX: removed [0] since, mean reduces already it to an int (new functionality of new torch version?
            return loss.detach().numpy(), \
                   torch_utils.get_flat_grad_from(
                       imitation_model.parameters()).detach().numpy(). \
                       astype(np.float64)

        curr_itr = 0

        def callback_fun(flat_params):
            nonlocal curr_itr
            torch_utils.set_flat_params_to(imitation_model, torch_utils.torch.from_numpy(flat_params))
            # calculate the loss of the whole batch
            loss = - torch.mean(imitation_model.get_log_prob(input, output))
            # weight decay
            for param in imitation_model.parameters():
                loss += param.pow(2).sum() * self.l2_reg
            loss.backward()
            if isinstance(self.imitationModel, CartPoleModel):
                logger.record_tabular("theta", str(self.imitationModel.theta.detach().numpy()))
                logger.record_tabular("std", str(self.imitationModel.std.detach().numpy()))
            logger.record_tabular('Iteration', curr_itr)
            logger.record_tabular("loss", loss.item())
            logger.dump_tabular(with_prefix=False)
            curr_itr += 1

        x0 = torch_utils.get_flat_params_from(self.imitationModel).detach().numpy()
        # only allow positive variables since we know the masses and variance cannot be negative
        bounds = [(0, np.inf) for _ in x0]

        flat_params, _, opt_info = scipy.optimize.fmin_l_bfgs_b(
            get_negative_likelihood_loss,
            x0, maxiter=self.n_itr, bounds=bounds, callback=callback_fun)
        logger.log(str(opt_info))
        torch_utils.set_flat_params_to(self.imitationModel, torch.from_numpy(flat_params))

        # save result
        logger.log("saving snapshot...")
        params, torch_params = self.get_itr_snapshot(0)
        params["algo"] = self
        logger.save_itr_params(self.n_itr, params, torch_params)
        logger.log("saved")

    @overrides
    def get_itr_snapshot(self, itr):
        if itr == 0:
            return dict(
                itr=itr,
                expert_data=self.expert_data,
                imitationModel=self.imitationModel,
            ), dict(imitationModel=self.imitationModel)
        else:
            return None, {'imitationModel': self.imitationModel}