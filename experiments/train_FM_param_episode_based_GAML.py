# Copyright (c) 2018 Copyright holder of the paper Generative Adversarial Model Learning
# submitted to NeurIPS 2019 for review
# All rights reserved.

from rllab.misc.instrument import run_experiment_custom
from rllab.dynamic_models.cartpole_model import CartPoleModel
from rllab.torch.models.nn_discriminator import NNDiscriminator
from rllab.torch.models.gaussian_upper_level_policy import GaussianUpperLevelPolicy
from rllab.torch.algos.gaml_episode_based_modellearning import GAMLEpisodeBasedModelLearning
from rllab.torch.utils.misc import str2bool
import numpy as np
import joblib
import argparse
import torch

def run_task(v):
    # load policy for gaussian noise for sigma^2: 0.05 and 0.01
    # data = joblib.load(
    data = joblib.load(v["expert_policy_path"])
    policy = data['policy']
    policy.normalized_input = [False, False, False, False]
    policy.normalized_output = [True]

    file_name = v["expert_data_path"]
    expert_paths = joblib.load(file_name)

    # we add 1 to the observation space, because we use sin and cos theta and + 1 for the timestep
    action_dim = 1
    observation_dim = 4
    discriminator = NNDiscriminator(input_dim=action_dim + (observation_dim + 1 ) * 2 +1)

    # we initialize the model with random to have the same initialization as we would have when we do MLE and use
    # the parameters to initialize the upper level policy
    imitation_env = CartPoleModel(initRandom=True, init_std=np.sqrt([v["var_x"], v["var_theta"]]))

    theta = imitation_env.theta.detach().numpy()
    init_std = imitation_env.std.detach().numpy()

    mean = torch.from_numpy(np.concatenate([theta, init_std])).float()
    covariance = torch.from_numpy(np.diag([0.05, 0.05, 0.05, 0.005, 0.005])).float()
    upper_level_policy = GaussianUpperLevelPolicy(mean, covariance)

    algo = GAMLEpisodeBasedModelLearning(policy,
                                         expert_paths,
                                         imitation_env,
                                         discriminator,
                                         upper_level_policy,
                                         n_itr=v["n_itr"],
                                         n_traj=v["n_traj"],
                                         n_samples=v["n_samples"],
                                         max_path_length=v["max_path_length"],
                                         use_timesteps=v["use_timestep"],
                                         use_state_diff_in_discriminator=v["use_state_diff_discrim"],
                                         discount=0.995,
                                         discriminator_updates_per_itr=v["discriminator_updates_per_itr"],
                                         )
    algo.train()


def run_GAML_training(dataset_num_traj, seed):
    from pathlib import Path
    mainDir = Path(__file__).parents[1]

    n_itr = 200
    n_traj = 25
    n_samples = 50
    var_x = 0.01
    var_theta = 0.01
    discriminator_updates_per_itr = 5
    expert_policy_path = str(Path.joinpath(mainDir, "datasets/policy_cartpole_swingup.pkl"))
    expert_data_path = str(Path.joinpath(mainDir, "datasets/expert_paths_n_traj_100_max_path_500_selected_" + str(dataset_num_traj) + ".pkl"))
    max_path_length = 500
    discount = 0.995
    resultsPath = str(Path.joinpath(mainDir, "results/"))

    run_experiment_custom(
        run_task,
        # Number of parallel workers for sampling
        n_parallel=1,
        # Only keep the snapshot parameters for the last iteration
        snapshot_mode="gap",
        snapshot_gap=5,
        # Specifies the seed for the experiment. If this is not provided, a random seed
        # will be used
        seed=seed,
        log_dir=resultsPath,
        log_debug_log_only=True,
        log_tabular_only=True,
        exp_prefix="GAML_cartpole_swingup",
        variant={'n_itr': n_itr,
                 'n_traj': n_traj,
                 'var_x': var_x,
                 'var_theta': var_theta,
                 "expert_policy_path": expert_policy_path,
                 "expert_data_path": expert_data_path,
                 "discriminator_updates_per_itr": discriminator_updates_per_itr,
                 "max_path_length": max_path_length,
                 "discount":discount,
                 "n_samples":n_samples,
                 "use_timestep":True,
                 "use_state_diff_discrim":False,
                 },
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=1, type=int, help="seed for the run")
    parser.add_argument('--tempdir', default='/tmp/', type=str, help='temp directory where result are written first')
    parser.add_argument('--exp_prefix', default='', type=str, help='prefix folder ')
    parser.add_argument('--n_itr', default=201, type=int, help="Number of Iterations")
    parser.add_argument('--n_traj', default=25, type=int, help='Number of used expert trajectories')
    parser.add_argument('--n_samples', default=50, type=int, help='Number of used expert trajectories')
    parser.add_argument('--var_x', default=0.01, type=float, help='variance of x')
    parser.add_argument('--var_theta', default=0.01, type=float, help='variance of theta')
    parser.add_argument('--use_timestep', default="True", type=str, help='if we use the timestep as input for the discriminator')
    parser.add_argument('--use_forward_KL', default="False", type=str, help='if we forward or the reverse KL')
    parser.add_argument('--use_state_diff_discrim', default="False", type=str, help='Use state diff or next_state for the discriminator')
    parser.add_argument('--discriminator_updates_per_itr', default=5, type=int,
                        help="number of updates for the discriminator we want to do in one iteration")
    parser.add_argument('--expert_policy_path',
                        default="/home/thh2rng/Documents/gaml/datasets/policy_cartpole_swingup.pkl",
                        type=str, help="filepath of the expert policy")
    parser.add_argument('--expert_data_path',
                        default="/home/thh2rng/Documents/gaml/datasets/expert_paths_n_traj_100_max_path_500_selected_100.pkl",
                        type=str, help="filepath of the expert data rollouts")
    parser.add_argument('--surrogateDiscriminator', default="ExpertDataDiscriminator", type=str)
    parser.add_argument('--max_path_length', default=500, type=int)
    parser.add_argument('--discount', default=0.995, type=float, help='discount factor')
    args = parser.parse_args()

    run_experiment_custom(
        run_task,
        # Number of parallel workers for sampling
        n_parallel=1,
        # Only keep the snapshot parameters for the last iteration
        snapshot_mode="gap",
        snapshot_gap=10,
        # Specifies the seed for the experiment. If this is not provided, a random seed
        # will be used
        seed=args.seed,
        log_dir=args.tempdir,
        #log_debug_log_only=True,
        #log_tabular_only=True,
        exp_prefix=args.exp_prefix,
        variant={'n_itr': args.n_itr,
                 'n_traj': args.n_traj,
                 'var_x': args.var_x,
                 'var_theta': args.var_theta,
                 'use_timestep': str2bool(args.use_timestep),
                 'use_forward_KL': str2bool(args.use_forward_KL),
                 'use_state_diff_discrim': str2bool(args.use_state_diff_discrim),
                 "expert_policy_path": args.expert_policy_path,
                 "expert_data_path": args.expert_data_path,
                 "discriminator_updates_per_itr": args.discriminator_updates_per_itr,
                 "surrogateDiscriminator": args.surrogateDiscriminator,
                 "max_path_length": args.max_path_length,
                 "discount":args.discount,
                 "n_samples":args.n_samples,
                 },
    )