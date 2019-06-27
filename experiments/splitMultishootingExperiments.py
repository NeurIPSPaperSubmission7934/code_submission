# Copyright (c) 2018 Copyright holder of the paper Generative Adversarial Model Learning
# submitted to NeurIPS 2019 for review
# All rights reserved.

import argparse
import os
import shutil

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--f', type=str, default='/dlc/Employees/thh2rng/ICML/fixed-FM-param-furuta-comparison-sample-efficiency/FM-Param-MultipleShooting', help='folder name')

    args = parser.parse_args()

    sps = [500, 250, 50, 20]

    experiment_master_folder_path = args.f
    experiment_master_folder_name = experiment_master_folder_path.split("/")[-1]
    experiment_master_folder_parent_path = os.path.dirname(experiment_master_folder_path)

    print(experiment_master_folder_name)
    print(experiment_master_folder_parent_path)

    num_seeds = 0
    # count the seeds
    # find all trials of the experiment
    for seed_fn in os.listdir(experiment_master_folder_path):
        path = os.path.join(experiment_master_folder_path, seed_fn)
        if not os.path.exists(path) or not os.path.isdir(path):
            continue
        num_seeds += 1

    print("number of seeds:", num_seeds)

    for seed_fn in os.listdir(experiment_master_folder_path):
        path = os.path.join(experiment_master_folder_path, seed_fn)
        if not os.path.exists(path) or not os.path.isdir(path):
            continue
        seed = str(path).split("-")[-1]

        experiment_folder_name = path

        i = 0
        for fn in sorted(os.listdir(experiment_folder_name)):
            path = os.path.join(experiment_folder_name, fn)
            print(path)
            print(sps[i % len(sps)])
            newFolder = os.path.join(experiment_master_folder_parent_path, experiment_master_folder_name + "-"+ str(sps[i % len(sps)]), seed_fn, fn)
            print(newFolder)
            shutil.copytree(path, newFolder)
            i += 1
