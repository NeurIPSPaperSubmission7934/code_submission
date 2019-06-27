# Copyright (c) 2018 Copyright holder of the paper Generative Adversarial Model Learning
# submitted to NeurIPS 2019 for review
# All rights reserved.

from rllab.misc.instrument import run_experiment_custom
from rllab.torch.algos.behavior_clone import BehaviorCloning
from rllab.dynamic_models.cartpole_model import CartPoleModel
from rllab.torch.metrics.eval_metrics import TrajSampler
import numpy as np
import joblib
import argparse
import torch
import os
from rllab.torch.utils.misc import str2bool
from pathlib import Path

def run_task(v):

    # sampler only needed to process expert paths
    sampler = TrajSampler(None, None, 0, 0, useImitationPolicy=False, useImitationEnv=False)
    sampler.start_worker()

    file_name = v["expert_data_path"]
    expert_paths = joblib.load(file_name)
    expert_data = sampler.process_samples(0, expert_paths)

    imitation_env = CartPoleModel(initRandom=True, init_std=np.sqrt([v["var_x"], v["var_theta"]]))

    if v["optimizer"] == "Adam":
        optimizer = torch.optim.Adam
    elif v["optimizer"] == "RMSProp":
        optimizer = torch.optim.RMSprop
    elif v["optimizer"] == "BGFS":
        optimizer = None
    else:
        raise NotImplementedError("Wrong optimizer")

    algo = BehaviorCloning(
                expert_data=expert_data,
                imitation_model=imitation_env,
                n_itr=v["n_itr"],
                weight_decay=v["L2_reg"],
                mini_batchsize=v["batchsize"],
                optim=optimizer,
                )

    algo.train()

def run_MLE_training(dataset_num_traj, seed):
    mainDir = Path(__file__).parents[1]

    # parameters reported in supplement
    n_itr = 2500
    var_x = 0.01
    var_theta = 0.01
    expert_data_path = str(Path.joinpath(mainDir, "datasets/expert_paths_n_traj_100_max_path_500_selected_" + str(dataset_num_traj) + ".pkl"))
    batchsize = 100
    L2_reg = 1e-5
    optimizer='Adam'

    resultsPath = str(Path.joinpath(mainDir, "results/"))

    if not os.path.exists(resultsPath):
        os.makedirs(resultsPath)

    run_experiment_custom(
        run_task,
        # Number of parallel workers for sampling
        n_parallel=1,
        # Only keep the snapshot parameters for the last iteration
        snapshot_mode="gap",
        snapshot_gap=25,
        # Specifies the seed for the experiment. If this is not provided, a random seed
        # will be used
        seed=seed,
        log_dir=resultsPath,
        log_debug_log_only=True,
        exp_prefix="MLE_cartpole_swingup",
        variant={'seed': seed,
                 'n_itr': n_itr,
                 'batchsize': batchsize,
                 'L2_reg': L2_reg,
                 'optimizer': optimizer,
                 'var_x': var_x,
                 'var_theta': var_theta,
                 "expert_data_path": expert_data_path,
                 },
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=11, type=int, help="seed for the run")
    parser.add_argument('--tempdir', default='/tmp/', type=str, help='temp directory where result are written first')
    parser.add_argument('--exp_prefix', default='', type=str, help='prefix folder ')
    parser.add_argument('--n_itr', default=1001, type=int, help="Number of Iterations / Epochs")
    parser.add_argument('--batchsize', default=100, type=int, help='size of one mini batch used for calculating the loss in one update step')
    parser.add_argument('--L2_reg', default=1e-5, type=float, help='L2 regularization coefficient')
    parser.add_argument('--optimizer', default='Adam', type=str, help='which optimizer to use')
    parser.add_argument('--var_x', default=0.01, type=float, help='variance of x')
    parser.add_argument('--var_theta', default=0.01, type=float, help='variance of theta')
    parser.add_argument('--expert_data_path', default="/home/thh2rng/Documents/gaml/datasets/expert_paths_n_traj_100_max_path_500_selected_10.pkl", type=str, help="filepath of the expert data rollouts")
    parser.add_argument('--n_step', type=int, default=1)
    parser.add_argument('--num_samples', type=int, default=1)
    parser.add_argument('--only_last_step_log_prob', type=str, default="True")


    args = parser.parse_args()

    run_experiment_custom(
        run_task,
        # Number of parallel workers for sampling
        n_parallel=1,
        # Only keep the snapshot parameters for the last iteration
        snapshot_mode="gap",
        snapshot_gap=25,
        # Specifies the seed for the experiment. If this is not provided, a random seed
        # will be used
        seed=args.seed,
        log_dir=args.tempdir,
        log_debug_log_only=True,
        exp_prefix=args.exp_prefix,
        variant={'seed':args.seed,
                 'n_itr':args.n_itr,
                 'batchsize':args.batchsize,
                 'L2_reg':args.L2_reg,
                 'optimizer':args.optimizer,
                 'var_x':args.var_x,
                 'var_theta':args.var_theta,
                 "expert_data_path": args.expert_data_path,
                 "n_step": args.n_step,
                 "num_samples": args.num_samples,
                 "only_last_step_log_prob": str2bool(args.only_last_step_log_prob),
                 },
    )