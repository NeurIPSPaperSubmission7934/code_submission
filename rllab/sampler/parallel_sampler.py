# Copyright (c) 2018 Copyright holder of the paper Generative Adversarial Model Learning
# submitted to NeurIPS 2019 for review
# All rights reserved.

# The following class is modified from rllab
# (https://github.com/rll/rllab)
# licensed under the MIT license

from rllab.sampler.utils import rollout, rollout_torch
from rllab.sampler.stateful_pool import singleton_pool, SharedGlobal
from rllab.misc import ext
from rllab.misc import logger
from rllab.misc import tensor_utils
import pickle
import numpy as np
from rllab.torch.models.policies.base import PytorchPolicy
import dill
#from rllab.controllers.FurutaExperimentController import FurutaExperimentController

def _worker_init(G, id):
    if singleton_pool.n_parallel > 1:
        import os
        os.environ['THEANO_FLAGS'] = 'device=cpu'
        os.environ['CUDA_VISIBLE_DEVICES'] = ""
    G.worker_id = id


def initialize(n_parallel):
    singleton_pool.initialize(n_parallel)
    singleton_pool.run_each(_worker_init, [(id,) for id in range(singleton_pool.n_parallel)])


def _get_scoped_G(G, scope):
    if scope is None:
        return G
    if not hasattr(G, "scopes"):
        G.scopes = dict()
    if scope not in G.scopes:
        G.scopes[scope] = SharedGlobal()
        G.scopes[scope].worker_id = G.worker_id
    return G.scopes[scope]


def _worker_populate_task(G, env, policy, scope=None, imitationPolicy=None, imitationEnv=None, use_furuta_controller=False):
    G = _get_scoped_G(G, scope)
    G.env = pickle.loads(env)
    if use_furuta_controller:
        G.policy = FurutaExperimentController()
    else:
        G.policy = dill.loads(policy)
    # if not imitationPolicy is None:
    G.imitationPolicy = pickle.loads(imitationPolicy)
    # if not imitationEnv is None:
    G.imitationEnv = pickle.loads(imitationEnv)


def _worker_terminate_task(G, scope=None):
    G = _get_scoped_G(G, scope)
    if getattr(G, "env", None):
        G.env.terminate()
        G.env = None
    if getattr(G, "policy", None):
        G.policy.terminate()
        G.policy = None


def populate_task(env, policy, scope=None, imitationPolicy=None, imitationEnv=None, use_furuta_controller=False):
    logger.log("Populating workers...")
    if singleton_pool.n_parallel > 1:

        if use_furuta_controller:
            singleton_pool.run_each(
                _worker_populate_task,
                [(
                 pickle.dumps(env), None, scope, pickle.dumps(imitationPolicy), pickle.dumps(imitationEnv), use_furuta_controller)] * singleton_pool.n_parallel
            )
        else:
            singleton_pool.run_each(
                _worker_populate_task,
                [(
                 pickle.dumps(env), dill.dumps(policy), scope, pickle.dumps(imitationPolicy), pickle.dumps(imitationEnv))] * singleton_pool.n_parallel
            )
        # if not imitationPolicy is None:
        #     singleton_pool.run_each(
        #         _worker_populate_task,
        #         [(pickle.dumps(env), pickle.dumps(policy), scope, pickle.dumps(imitationPolicy))] * singleton_pool.n_parallel
        #     )
        # else:
        #     singleton_pool.run_each(
        #         _worker_populate_task,
        #         [(pickle.dumps(env), pickle.dumps(policy), scope)] * singleton_pool.n_parallel
        #     )
    else:
        # avoid unnecessary copying
        G = _get_scoped_G(singleton_pool.G, scope)
        G.env = env
        G.policy = policy
        if not imitationPolicy is None:
            G.imitationPolicy = imitationPolicy
        if not imitationEnv is None:
            G.imitationEnv = imitationEnv
    logger.log("Populated")


def terminate_task(scope=None):
    singleton_pool.run_each(
        _worker_terminate_task,
        [(scope,)] * singleton_pool.n_parallel
    )


def _worker_set_seed(_, seed):
    logger.log("Setting seed to %d" % seed)
    ext.set_seed(seed)


def set_seed(seed):
    singleton_pool.run_each(
        _worker_set_seed,
        [(seed,) for i in range(singleton_pool.n_parallel)]
    )


def _worker_set_policy_params(G, params, scope=None):
    G = _get_scoped_G(G, scope)
    G.policy.set_param_values(params)

def _worker_set_env_params(G,params,scope=None):
    G = _get_scoped_G(G, scope)
    G.env.set_param_values(params)

def _worker_collect_one_path(G, max_path_length, scope=None, useImitationPolicy=False, useImitationEnv=False, count_traj=False, terminate_only_max_path=False):
    G = _get_scoped_G(G, scope)

    if useImitationEnv:
        env = G.imitationEnv
    else:
        env = G.env

    if useImitationPolicy:
        policy = G.imitationPolicy
    else:
        policy = G.policy

    if isinstance(policy, PytorchPolicy):
        rollout_func = rollout_torch
    else:
        rollout_func = rollout

    if terminate_only_max_path:
        path = rollout_func(env, policy, max_path_length=max_path_length, terminate_only_max_path=terminate_only_max_path)
    else:
        path = rollout_func(env, policy, max_path_length)
    if not count_traj:
        return path, len(path["rewards"])
    else:
        return path, 1

def sample_paths(
        policy_params,
        max_samples,
        max_path_length=np.inf,
        env_params=None,
        scope=None,
        useImitationPolicy=False,
        useImitationEnv=False,
        count_traj=False,
        terminate_only_max_path=False):
    """
    :param policy_params: parameters for the policy. This will be updated on each worker process
    :param max_samples: desired maximum number of samples to be collected. The actual number of collected samples
    might be greater since all trajectories will be rolled out either until termination or until max_path_length is
    reached
    :param max_path_length: horizon / maximum length of a single trajectory
    :param count_traj: if true then max_samples is the desired maximum number of trajectories to be collected.
    :return: a list of collected paths
    """
    if not useImitationPolicy and not policy_params is None:
        singleton_pool.run_each(
            _worker_set_policy_params,
            [(policy_params, scope)] * singleton_pool.n_parallel
        )
    if env_params is not None:
        singleton_pool.run_each(
            _worker_set_env_params,
            [(env_params, scope)] * singleton_pool.n_parallel
        )
    return singleton_pool.run_collect(
        _worker_collect_one_path,
        threshold=max_samples,
        args=(max_path_length, scope, useImitationPolicy, useImitationEnv, count_traj, terminate_only_max_path),
        show_prog_bar=True
    )

def truncate_paths(paths, max_samples):
    """
    Truncate the list of paths so that the total number of samples is exactly equal to max_samples. This is done by
    removing extra paths at the end of the list, and make the last path shorter if necessary
    :param paths: a list of paths
    :param max_samples: the absolute maximum number of samples
    :return: a list of paths, truncated so that the number of samples adds up to max-samples
    """
    # chop samples collected by extra paths
    # make a copy
    paths = list(paths)
    total_n_samples = sum(len(path["rewards"]) for path in paths)
    while len(paths) > 0 and total_n_samples - len(paths[-1]["rewards"]) >= max_samples:
        total_n_samples -= len(paths.pop(-1)["rewards"])
    if len(paths) > 0:
        last_path = paths.pop(-1)
        truncated_last_path = dict()
        truncated_len = len(last_path["rewards"]) - (total_n_samples - max_samples)
        for k, v in last_path.items():
            if k in ["observations", "actions", "rewards"]:
                truncated_last_path[k] = tensor_utils.truncate_tensor_list(v, truncated_len)
            elif k in ["env_infos", "agent_infos"]:
                truncated_last_path[k] = tensor_utils.truncate_tensor_dict(v, truncated_len)
            else:
                raise NotImplementedError
        paths.append(truncated_last_path)
    return paths
