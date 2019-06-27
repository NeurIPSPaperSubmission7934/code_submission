# Copyright (c) 2018 Copyright holder of the paper Generative Adversarial Model Learning
# submitted to NeurIPS 2019 for review
# All rights reserved.

from rllab.misc.instrument import run_experiment_custom
from rllab.casadi.algos.parameter_estimation import ParameterEstimation
from rllab.dynamic_models.cartpole_model import CartPoleModel
from rllab.casadi.dynamic_models.cartpole_model_casadi import CartpoleModelCasadi
from rllab.casadi.dynamic_models.integrators import euler_integration, rk_integration
import numpy as np
import joblib
import argparse

def run_task(v):
    # load expert paths
    file_name = v["expert_data_path"]
    expert_paths = joblib.load(file_name)

    # create casadi model and init param_guess according to seed
    if v["env"] == 'CartpoleSwingup':
        imitation_env = CartPoleModel(initRandom=True)
        model = CartpoleModelCasadi()
        integration_func_options = {"integrator_dt": imitation_env.tau, "integrator_stepsize": 1, "angular_idx":[2]}
        # we divide by scale to have the params for casadi in the range of 0.1 and 100, which works better
        # according to the tutorial
        param_guess = imitation_env.theta.detach().numpy()
        scale = param_guess / 10 * np.random.rand(param_guess.shape[0])
        param_guess = param_guess / scale
        print(param_guess)
    else:
        raise NotImplementedError("Env not implemented")
    if v["integration_func"] == 'Euler':
        integration_func = euler_integration
    elif v["integration_func"] == 'RK4':
        integration_func = rk_integration

    algo = ParameterEstimation(expert_paths, model, v["estimationMode"], integration_func, param_guess, modeOptions={"n_step_pred_sp":v["n_step_pred_sp"],
                                                                                                                     "scale":scale,
                                                                                                                     "n_itr":v["n_itr"]},
                               integrationFuncOptions=integration_func_options)
    algo.train()

def runSingleShooting(dataset_num_traj, seed):
    from pathlib import Path
    mainDir = Path(__file__).parents[1]

    n_itr = 3000
    expert_data_path = str(Path.joinpath(mainDir, "datasets/expert_paths_n_traj_100_max_path_500_selected_" + str(dataset_num_traj) + "_casadi.pkl"))
    resultsPath = str(Path.joinpath(mainDir, "results/"))

    run_experiment_custom(
        run_task,
        # Number of parallel workers for sampling
        n_parallel=1,
        # Only keep the snapshot parameters for the last iteration
        snapshot_mode="gap",
        snapshot_gap=50,
        # Specifies the seed for the experiment. If this is not provided, a random seed
        # will be used
        seed=seed,
        log_dir=resultsPath,
        log_debug_log_only=True,
        exp_prefix="PE_SingleShooting_cartpole_swingup",
        variant={
            "n_itr":n_itr,
            "estimationMode":"singleshooting",
            "integration_func":"Euler",
            "env":"CartpoleSwingup",
            "expert_data_path":expert_data_path,
            "n_step_pred_sp":1
        },
    )

def runMultipleShooting(dataset_num_traj, seed):
    from pathlib import Path
    mainDir = Path(__file__).parents[1]

    n_itr = 3000
    expert_data_path = str(Path.joinpath(mainDir, "datasets/expert_paths_n_traj_100_max_path_500_selected_" + str(dataset_num_traj) + "_casadi.pkl"))
    resultsPath = str(Path.joinpath(mainDir, "results/"))

    run_experiment_custom(
        run_task,
        # Number of parallel workers for sampling
        n_parallel=1,
        # Only keep the snapshot parameters for the last iteration
        snapshot_mode="gap",
        snapshot_gap=50,
        # Specifies the seed for the experiment. If this is not provided, a random seed
        # will be used
        seed=seed,
        log_dir=resultsPath,
        log_debug_log_only=True,
        exp_prefix="PE_MultipleShooting_cartpole_swingup",
        variant={
            "n_itr":n_itr,
            "estimationMode":"multipleshooting",
            "integration_func":"Euler",
            "env":"CartpoleSwingup",
            "expert_data_path":expert_data_path,
            "n_step_pred_sp":20
        },
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=11, type=int, help="seed for the run")
    parser.add_argument('--tempdir', default='/tmp/', type=str, help='temp directory where result are written first')
    parser.add_argument('--exp_prefix', default='', type=str, help='prefix folder ')
    parser.add_argument('--n_itr', default=51, type=int, help="Number of Iterations")
    # parser.add_argument('--expert_data_path',
    #                     default="/dlc/Employees/gaml/trpo-expert-cartpole-swingup/good_noise_policy/friction_obs_noise/0.075/expert_paths_n_traj_200_max_path_500_selected_100_selected_5_casadi.pkl",
    #                     type=str, help="filepath of the expert data rollouts")
    parser.add_argument('--expert_data_path',
                        default="/home/thh2rng/rllab/data/furuta_paths_obs_diff_our_rep_15.pkl",
                        type=str, help="filepath of the expert data rollouts")
    parser.add_argument('--estimationMode', type=str, help="single or multiple shooting", default="multipleshooting")
    parser.add_argument('--integration_func', type=str, help="Euler or RK4", default="Euler")
    parser.add_argument('--n_step_pred_sp', type=int, default=20)
    parser.add_argument('--env', type=str, default="FurutaPendulum")
    args = parser.parse_args()

    run_experiment_custom(
        run_task,
        # Number of parallel workers for sampling
        n_parallel=1,
        # Only keep the snapshot parameters for the last iteration
        snapshot_mode="gap",
        snapshot_gap=50,
        # Specifies the seed for the experiment. If this is not provided, a random seed
        # will be used
        seed=args.seed,
        log_dir=args.tempdir,
        #log_debug_log_only=True,
        exp_prefix=args.exp_prefix,
        variant=args.__dict__
    )


