# Copyright (c) 2018 Copyright holder of the paper Generative Adversarial Model Learning
# submitted to NeurIPS 2019 for review
# All rights reserved.


import numpy as np
from rllab.sampler.base import Sampler
from rllab.misc import special
from rllab.misc import tensor_utils
import rllab.misc.logger as logger
import torch

class GAILSampler(Sampler):
    def __init__(self, algo):
        """
        :type algo: BatchPolopt
        """
        self.algo = algo

    def process_samples(self, itr, paths):
        returns = []

        for idx, path in enumerate(paths):
            path["returns"] = special.discount_cumsum(path["rewards"], self.algo.discount)
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

        average_discounted_return = \
            np.mean([path["returns"][0] for path in paths])

        undiscounted_returns = [sum(path["rewards"]) for path in paths]

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

        assert not np.any(np.isnan(observations)), "observations are nan"
        assert not np.any(np.isnan(next_observations)), "next observations are nan"
        assert not np.any(np.isnan(actions)), "actions are nan"

        logger.record_tabular('Iteration', itr)
        logger.record_tabular('AverageDiscountedReturn',
                              average_discounted_return)
        logger.record_tabular('AverageReturn', np.mean(undiscounted_returns))
        logger.record_tabular('NumTrajs', len(paths))
        logger.record_tabular('StdReturn', np.std(undiscounted_returns))
        logger.record_tabular('MaxReturn', np.max(undiscounted_returns))
        logger.record_tabular('MinReturn', np.min(undiscounted_returns))
        logger.record_tabular('AverageTrajLength', np.mean([np.argmax(path["mask"] == 0) for path in paths]))
        if hasattr(self.algo, 'model'):
            observations = torch.from_numpy(observations).float()
            actions = torch.from_numpy(actions).float()
            x = torch.cat([observations, actions], dim=1)
            next_observations = torch.from_numpy(next_observations).float()
            ent = np.mean(self.algo.model.get_entropy(x, next_observations).detach().numpy())
            logger.record_tabular('Entropy', ent)
        if hasattr(self.algo, 'imitationPolicy'):
            x = torch.from_numpy(observations).float()
            actions = torch.from_numpy(actions).float()
            ent = np.mean(self.algo.imitationPolicy.get_entropy(x, actions).detach().numpy())
            logger.record_tabular('Entropy', ent)
        return samples_data
