# Copyright (c) 2018 Copyright holder of the paper Generative Adversarial Model Learning
# submitted to NeurIPS 2019 for review
# All rights reserved.

import joblib
import numpy as np
import torch
from rllab.envs.cartpole_swingup_env import CartPoleSwingUpEnv
from rllab.torch.metrics.eval_metrics import TrajSampler
import argparse

def processExpertCartpoleData(file_name, store_only_good):

    expert_paths = joblib.load(file_name)

    if store_only_good:
        # # let's filter the trajectories that we only have good ones (total reward > 260.0)
        indicies = []
        for idx, path in enumerate(expert_paths):
            if (sum(path["rewards"])) > 260:
                indicies.append(idx)
            else:
                print(sum(path["rewards"]))
                print(len(path["rewards"]))
    else:
        indicies = range(len(expert_paths))

    print(indicies)
    print(len(indicies))

    # select n_traj from them and save them
    perm_indices = np.random.permutation(np.array(indicies))
    for n_traj in [1, 5, 10, 25, 50, 100]:
        selected_indices = perm_indices[:n_traj]
        print(selected_indices)
        print(selected_indices.tolist())
        selected_expert_paths = [expert_paths[i] for i in selected_indices]

        for idx, path in enumerate(selected_expert_paths):
            print(sum(path["rewards"]))
            print(len(path["rewards"]))

        if store_only_good:
            selected_expert_file_path = file_name[:-4] + "_selected_good_" + str(n_traj) + ".pkl"
        else:
            selected_expert_file_path = file_name[:-4] + "_selected_" + str(n_traj) + ".pkl"
        print(selected_expert_file_path)
        joblib.dump(selected_expert_paths, selected_expert_file_path, compress=3)

        for path in selected_expert_paths:
            path["agent_infos"] = {}

        joblib.dump(selected_expert_paths, selected_expert_file_path[:-4] + "_casadi.pkl")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--f', type=str, help='file name')
    parser.add_argument('--good', action='store_true')
    args = parser.parse_args()

    processExpertCartpoleData(args.f, args.good)