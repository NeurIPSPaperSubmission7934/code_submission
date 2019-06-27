# Copyright (c) 2018 Copyright holder of the paper Generative Adversarial Model Learning
# submitted to NeurIPS 2019 for review
# All rights reserved.

import rllab.misc.logger as logger
from rllab.torch.sampler.torchSampler import TorchBaseSampler
from rllab.sampler import parallel_sampler
from rllab.algos.base import RLAlgorithm
from rllab.torch.algos.trpo import TRPO
from rllab.torch.utils.misc import create_torch_var_from_paths
import torch
from rllab.misc.overrides import overrides

class BatchSampler(TorchBaseSampler):
    def __init__(self, algo):
        """
        :type algo: BatchPolopt
        """
        self.algo = algo

    def start_worker(self):
        parallel_sampler.populate_task(self.algo.env, self.algo.policy, scope=self.algo.scope)

    def shutdown_worker(self):
        parallel_sampler.terminate_task(scope=self.algo.scope)

    def obtain_samples(self, itr):
        paths = parallel_sampler.sample_paths(
            policy_params=None,
            max_samples=self.algo.batch_size,
            max_path_length=self.algo.max_path_length,
            scope=self.algo.scope,
        )
        return paths


"""
class which is used for GAIL training to imitate a expert policy
"""
class TRPOTraining(RLAlgorithm):

    def __init__(self, env, policy, n_itr=500, batch_size=5000, max_path_length=500, discount=0.99, current_itr=0, max_kl=0.01, scope=None,):
        self.env = env
        self.policy = policy
        self.n_itr = n_itr
        self.current_itr = current_itr
        self.max_path_length = max_path_length
        self.batch_size = batch_size
        self.discount = discount
        sampler_cls = BatchSampler
        self.sampler = sampler_cls(self)
        self.scope = scope
        self.trpo_optimizer = TRPO(self.policy, max_kl=max_kl, discount=discount)

        # TODO: add parameter to init
        self.store_paths = False

    def start_worker(self):
        self.sampler.start_worker()

    def shutdown_worker(self):
        self.sampler.shutdown_worker()


    def train(self):
        self.start_worker()

        for itr in range(self.current_itr, self.n_itr):
            with logger.prefix('itr #%d | ' % itr):
                # TODO: do we use a new rollout on expert data in each itr? for now we can do so but at some point we only have a fixed dataset
                generated_paths = self.sampler.obtain_samples(itr)
                generated_data = self.sampler.process_samples(itr, generated_paths)
                self.log_diagnostics(generated_paths)

                self.optimize_policy(itr, generated_data)

                logger.log("saving snapshot...")
                params = self.get_itr_snapshot(itr, generated_data)
                self.current_itr = itr + 1
                params["algo"] = self
                if self.store_paths:
                    params["paths"] = generated_data["paths"]
                logger.save_itr_params(itr, params)
                logger.log("saved")

                logger.dump_tabular(with_prefix=False)

        self.shutdown_worker()

    def log_diagnostics(self, paths):
        self.env.log_diagnostics(paths)

    def optimize_policy(self, itr, samples_data):
        observations, actions, next_observations = create_torch_var_from_paths(samples_data)
        rewards = torch.from_numpy(samples_data["rewards"]).float()
        masks = torch.from_numpy(samples_data["masks"]).float()
        self.trpo_optimizer.update_networks(self.policy, actions, masks, rewards, observations, 1)


    @overrides
    def get_itr_snapshot(self, itr, samples_data):
        return dict(
            itr=itr,
            policy=self.policy,
            env=self.env,
        )
