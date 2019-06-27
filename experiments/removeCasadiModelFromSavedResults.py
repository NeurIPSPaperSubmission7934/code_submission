# Copyright (c) 2018 Copyright holder of the paper Generative Adversarial Model Learning
# submitted to NeurIPS 2019 for review
# All rights reserved.

import argparse
import os
import shutil
import joblib

def removeCasadiModelAndSaveCopy(folderName, iteration):

    experiment_folder_name = folderName

    num_experiment_trial = 0
    # find all trials of the experiment
    for fn in os.listdir(experiment_folder_name):
        path = os.path.join(experiment_folder_name, fn)
        model_path = os.path.join(path, "itr_0.pkl")
        if not os.path.exists(model_path) and not os.path.exists(os.path.join(path, "itr_"+str(iteration)+".pkl")):
            continue
        num_experiment_trial += 1

    print("number of experiments", num_experiment_trial)

    for fn in sorted(os.listdir(experiment_folder_name)):
        path = os.path.join(experiment_folder_name, fn)
        print(path)
        # take the newest iteration we have
        for itr in range(iteration, -1, -1):
            if os.path.exists(os.path.join(path, "itr_" + str(itr) + ".pkl")):
                # check if casadi_file is already there
                    model_path = os.path.join(path, "itr_" + str(itr) + ".pkl")
                    print("found model", model_path)
                    if not os.path.exists(os.path.join(path, "itr_" + str(itr) + "_casadi.pkl")):
                        print("no casadi copy", model_path)
                        shutil.copy2(model_path, os.path.join(path, "itr_" + str(itr) + "_casadi.pkl"))
                    # open file and remove the casadi imitationModel and resave
                    model = joblib.load(model_path)
                    model['imitationModel'] = None
                    model['integration_func'] = None
                    joblib.dump(model, model_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--f', type=str, default='/dlc/Employees/thh2rng/ICML/fixed-FM-param-furuta-comparison-sample-efficiency/FM-Param-MultipleShooting/', help='folder name')
    parser.add_argument('--i', type=int, default=3000)

    args = parser.parse_args()

    experiment_master_folder_name = args.f
    num_seeds = 0
    # count the seeds
    # find all trials of the experiment
    for fn in os.listdir(experiment_master_folder_name):
        path = os.path.join(experiment_master_folder_name, fn)
        if not os.path.exists(path) or not os.path.isdir(path):
            continue
        num_seeds += 1

    print("number of seeds:", num_seeds)

    for fn in os.listdir(experiment_master_folder_name):
        path = os.path.join(experiment_master_folder_name, fn)
        if not os.path.exists(path) or not os.path.isdir(path):
            continue
        seed = str(path).split("-")[-1]

        removeCasadiModelAndSaveCopy(path + "/", args.i)

