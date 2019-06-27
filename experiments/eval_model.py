# Copyright (c) 2018 Copyright holder of the paper Generative Adversarial Model Learning
# submitted to NeurIPS 2019 for review
# All rights reserved.

import joblib
import torch
from rllab.torch.metrics.eval_metrics import TrajSampler, calc_avg_displacement_timesteps, calc_leaving_boundaries_rate, calc_success_rate
from rllab.torch.utils import torch as torch_utils
import numpy as np
import argparse
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import json
import warnings
from pathlib import Path


def processResults(algo):
    # define where the gaml results are and run summarize and plot script
    mainDir = Path(__file__).parents[1]
    if algo =="GAML":
        resultsPath = Path.joinpath(mainDir, "results/GAML-cartpole-swingup/")
        itr = 200
    elif algo == "MLE":
        resultsPath = Path.joinpath(mainDir, "results/MLE-cartpole-swingup/")
        itr = 2500
    elif algo == "MultipleShooting":
        resultsPath = Path.joinpath(mainDir, "results/PE-MultipleShooting-cartpole-swingup/")
        itr = 3000
    elif algo == "SingleShooting":
        resultsPath = Path.joinpath(mainDir, "results/PE-SingleShooting-cartpole-swingup/")
        itr = 3000
    else:
        raise NotImplementedError()

    expert_policy_path = str(Path.joinpath(mainDir, "datasets/policy_cartpole_swingup.pkl"))
    expert_data_path = str(Path.joinpath(mainDir, "datasets/expert_paths_n_traj_100_max_path_500_selected_100.pkl"))

    # create 100 trajectories for each model
    summarize_and_plot_results(resultsPath, itr, 100, expert_policy_path, expert_data_path)


def summarize_and_plot_results(folderName, iteration, sample_n_traj, policy_path, expert_data_path, plot_trajectories=True):
    experiment_folder_name=str(folderName)+"/"

    eval_labels = ['avg_discounted_return', 'avg_undiscounted_return', 'avg_traj_length',
                   'avg_displacement', 'avg_min_displacement','avg_boundaries_left', 'avg_success_rate', 'avg_norm_displacement']

    print(experiment_folder_name)

    num_experiment_trial = 0
    # find all trials of the experiment
    for fn in os.listdir(experiment_folder_name):
        path = os.path.join(experiment_folder_name, fn)
        model_path = os.path.join(path, "itr_0.pkl")
        if not os.path.exists(model_path) and not os.path.exists(os.path.join(path, "itr_"+str(iteration)+".pkl")):
            continue
        if num_experiment_trial == 0:
            # read one variant.json testwise to obtain the columns for the data frame
            fileName = path + '/variant.json'
            with open(fileName, 'r') as read_file:
                data = json.load(read_file)
                col = ['experiment_itr']
                variant_keys = data.keys()
                col += data.keys()
                col += eval_labels
        num_experiment_trial += 1

    # create a empty data frame for the results
    df = pd.DataFrame(index=np.arange(0, num_experiment_trial), columns=col)

    # create different data frame for plotting
    eval_measures_label = eval_labels
    num_eval_measures = len(eval_measures_label)
    df_plot = pd.DataFrame(index=np.arange(0, num_experiment_trial*num_eval_measures),
                           columns=('expr_itr', 'variant', 'eval_type', 'value'))

    expr_idx = 0
    df_plot_idx = 0

    # create a data_frame which is used to plot results of the training iterations
    eval_measures_itr_label = ['NumTrajs', 'AverageReturn', 'Entropy']
    dataFramePlotItr = []

    for fn in sorted(os.listdir(experiment_folder_name)):
        path = os.path.join(experiment_folder_name, fn)

        model_path = os.path.join(path, "itr_0.pkl")
        if not os.path.exists(model_path) and not os.path.exists(os.path.join(path, "itr_"+str(iteration)+".pkl")):
            continue
        # check if we have a model for the iteration, if yes load this
        if os.path.exists(os.path.join(path, "itr_"+str(iteration)+".pkl")):
            model_path = os.path.join(path, "itr_"+str(iteration)+".pkl")
        else:
            # take the newest iteration we have
            for itr in range(iteration, 0, -1):
                if os.path.exists(os.path.join(path, "itr_" + str(itr) + ".pkl")):
                    model_path = os.path.join(path, "itr_" + str(itr) + ".pkl")
                    break
        model_param_path = "empty"
        # check if we have a model parameters
        if os.path.exists(os.path.join(path, "itr_"+str(iteration)+"_model.pkl")):
            model_param_path = os.path.join(path, "itr_" + str(iteration) + "_model.pkl")
        else:
            # take the newest iteration we have
            for itr in range(iteration, 0, -1):
                if os.path.exists(os.path.join(path, "itr_" + str(itr) + "_model.pkl")):
                    model_param_path = os.path.join(path, "itr_" + str(itr) + "_model.pkl")
                    break

        fileName = path + '/variant.json'
        with open(fileName, 'r') as read_file:
            data = json.load(read_file)

        # get the values and convert them to strings to store them easier
        variant_values = [str(data[x]) if hasattr(data, x) else "None" for x in variant_keys]


        if os.stat(path + '/progress.csv').st_size > 0:
            # only process csv if it is not empty
            unprocessed_df = pd.read_csv(path + '/progress.csv')
            temp_frames = []
            isGailExperiment = True
            try:
                for label in eval_measures_itr_label:
                    temp_frame=pd.DataFrame(unprocessed_df[label])
                    temp_frame.columns = ['value']
                    temp_frame['eval_type'] = label
                    temp_frames.append(temp_frame)
                processed_df = pd.concat(temp_frames)
                processed_df['Iteration'] = unprocessed_df['Iteration']
                processed_df['expr_idx'] = expr_idx
                processed_df['variant'] = "v" + str(expr_idx)
                dataFramePlotItr.append(processed_df)
            except KeyError:
                isGailExperiment = False

        # load policy for gaussian noise for sigma^2: 0.05 and 0.01
        data = joblib.load(policy_path)
        policy = data['policy']
        # policy.action_log_std = torch.nn.Parameter(
        #     torch.zeros(1, 1))  # set variance to zero to have a deterministic policy
        policy.normalized_input = [False, False, False, False]
        policy.normalized_output = [True]

        shooting_experiment = False
        # load learned model
        model = joblib.load(model_path)
        if 'imitationModel' in model:
            imitation_env = model['imitationModel']
        elif 'imitationEnv' in model:
            imitation_env = model['imitationEnv']
        elif 'parameters' in model:
            learned_params = model['parameters'].squeeze(0)
            shooting_experiment = True
            # create a new environment and set the parameters to the learned ones
            from rllab.dynamic_models.cartpole_model import CartPoleModel
            imitation_env = CartPoleModel()
            # set the variance to 0 to have a determinisitic environment
            imitation_env.set_param_values(torch.from_numpy(np.concatenate([learned_params, np.zeros(2)])).float())
            # use original variance to have a stochastic environment
            # get variance from true env
            # print(env.std)
            # imitation_env.set_param_values(torch.from_numpy(np.concatenate([learned_params, env.std])).float())
        if 'upperLevelPolicy' in model:
            upperLevelPolicy = model['upperLevelPolicy']

            torch_utils.set_flat_params_to(imitation_env, upperLevelPolicy.mean)
        elif not shooting_experiment:
            imitation_env.load_state_dict(torch.load(model_param_path))

        # load fixed expert trajectories
        expert_paths = joblib.load(expert_data_path)

        # # collect batch of expert_data containing n trajectories
        sampler = TrajSampler(policy, imitation_env, sample_n_traj, 500, discount=0.995, useImitationPolicy=False,
                              useImitationEnv=False, terminate_only_max_path=True)
        expert_processed_paths = sampler.process_samples(0, expert_paths)

        ## check if we need to do the rollouts or if we can just load in the processed_paths
        if os.path.exists(os.path.join(path, "itr_" + str(iteration) + "_n_traj_" + str(sample_n_traj) + "_generated_paths.pkl")) and \
                os.path.exists(os.path.join(path, "itr_" + str(iteration) + "_n_traj_" + str(sample_n_traj) + "_generated_processed_paths.pkl")):
            generated_paths = joblib.load(os.path.join(path, "itr_" + str(iteration) + "_n_traj_" + str(sample_n_traj) + "_generated_paths.pkl"))
            generated_processed_paths = joblib.load(os.path.join(path, "itr_" + str(iteration) + "_n_traj_" + str(sample_n_traj) + "_generated_processed_paths.pkl"))
        else:
            sampler.start_worker()
            generated_paths = sampler.obtain_samples(0)
            generated_processed_paths = sampler.process_samples(0, generated_paths)
            # save processed paths s.t. we don't need to do the rollouts every time
            joblib.dump(generated_paths, os.path.join(path, "itr_" + str(iteration) + "_n_traj_" + str(sample_n_traj) + "_generated_paths.pkl"),compress=3)
            joblib.dump(generated_processed_paths, os.path.join(path, "itr_" + str(iteration) + "_n_traj_" + str(sample_n_traj) + "_generated_processed_paths.pkl"), compress=3)

        avg_discounted_return = sampler.calc_avg_discounted_return(generated_processed_paths)
        avg_undiscounted_return = sampler.calc_avg_undiscounted_return(generated_processed_paths)
        avg_traj_length = sampler.calc_avg_traj_length(generated_processed_paths)

        avg_displacements = calc_avg_displacement_timesteps(expert_processed_paths, generated_processed_paths, 500, "CartPole")
        avg_displacement = np.mean(avg_displacements)
        avg_norm_displacements = calc_avg_displacement_timesteps(expert_processed_paths, generated_processed_paths, 500, "CartPole", normalize=True)
        avg_norm_displacement = np.mean(avg_norm_displacements)
        avg_min_displacements = calc_avg_displacement_timesteps(expert_processed_paths, generated_processed_paths, 500, "CartPole", "min")
        avg_min_displacement = np.mean(avg_min_displacements)
        avg_leave_boundaries = calc_leaving_boundaries_rate(generated_processed_paths)
        success_rate = calc_success_rate(generated_processed_paths)

        values = [avg_discounted_return, avg_undiscounted_return, avg_traj_length,
                  avg_displacement, avg_min_displacement, avg_leave_boundaries, success_rate, avg_norm_displacement]

        if not os.path.exists(experiment_folder_name + "displacements_" + str(iteration) + "/"):
            os.makedirs(experiment_folder_name + "displacements_" + str(iteration) + "/")
        all_displacements = {'avg_displacements': avg_displacements,
                             'avg_min_displacements': avg_min_displacements,
                             'avg_norm_displacements': avg_norm_displacements}
        joblib.dump(all_displacements, experiment_folder_name+ "displacements_" + str(iteration)+ "/"+ fn + ".pkl")

        df.loc[expr_idx] = [expr_idx] + variant_values + values

        for j, value in zip(range(num_eval_measures), values):
            # build variant string
            variant_string = "v" + str(expr_idx)
            df_plot.loc[df_plot_idx+j] = [expr_idx, variant_string, eval_measures_label[j], value]

        expr_idx += 1
        df_plot_idx += num_eval_measures

        # write results to folder for plotting
        df_plot.to_csv(experiment_folder_name + "results_plot_frame_"+ str(iteration)+".csv", sep=',', encoding='utf-8' )

        if plot_trajectories:
            plot_n_traj = 20 # plot the same number as in the supplement
            # check if plot already are already existing if yes don't plot
            if os.path.exists(experiment_folder_name + "no_cut_"+ str(iteration)+"/" + fn + ".png") and \
                    os.path.exists(experiment_folder_name + "cut_"+ str(iteration)+"/" + fn + ".png"):
                continue

            # create plot with 25 trajectory from real env and learned env
            sns.set(context="notebook", style="darkgrid")
            title = ['x_pos', 'x_dot', 'sin theta', 'cos theta', 'theta_dot']
            fig, axes = plt.subplots(len(title), 1, tight_layout=True)
            # the size of A4 paper + 4 inch addtional height
            fig.set_size_inches(11.7, 12.27)
            # merge all observations of all paths into one big data frame
            dataFrames = []
            print("building data frame")
            for i in range(plot_n_traj):
                trajlen = expert_paths[i]["observations"].shape[0]
                modObservations = np.zeros((trajlen, len(title)))
                modObservations[:, 0:2] = expert_paths[i]["observations"][:, 0:2]
                modObservations[:, 2] = np.sin(expert_paths[i]["observations"][:, 2])
                modObservations[:, 3] = np.cos(expert_paths[i]["observations"][:, 2])
                modObservations[:, 4] = expert_paths[i]["observations"][:, 3]
                trajDataFrame = pd.DataFrame(np.concatenate(
                    [i * np.ones(trajlen)[:, np.newaxis], np.arange(trajlen)[:, np.newaxis], modObservations], axis=1),
                    columns=['traj', 'timestep'] + title)
                dataFrames.append(trajDataFrame)
            all_traj_true_env = pd.concat(dataFrames)
            all_traj_true_env["model"] = "true environment"

            # now add 10 traj from learned env
            dataFrames = []
            for i in range(plot_n_traj):
                trajlen = generated_paths[i]["observations"].shape[0]
                modObservations = np.zeros((trajlen, len(title)))
                modObservations[:, 0:2] = generated_paths[i]["observations"][:, 0:2]
                modObservations[:, 2] = np.sin(generated_paths[i]["observations"][:, 2])
                modObservations[:, 3] = np.cos(generated_paths[i]["observations"][:, 2])
                modObservations[:, 4] = generated_paths[i]["observations"][:, 3]
                trajDataFrame = pd.DataFrame(np.concatenate(
                    [i * np.ones(trajlen)[:, np.newaxis], np.arange(trajlen)[:, np.newaxis], modObservations], axis=1),
                    columns=['traj', 'timestep'] + title)
                dataFrames.append(trajDataFrame)
            all_traj_learned_env = pd.concat(dataFrames)
            all_traj_learned_env["model"] = "learned environment"
            all_traj = pd.concat([all_traj_true_env, all_traj_learned_env])

            print("finished building data frame")

            print("start plotting")
            for i in range(len(title)):
                ax = sns.lineplot(x='timestep', y=title[i], hue="model", estimator=None, units='traj', data=all_traj,
                                  ax=axes[i], legend=False)
                if i == 0:
                    # add additional 2 dashed lines for the boundary
                    ax.plot([0, 500], [3, 3], color='k', linestyle='--')
                    ax.plot([0, 500], [-3, -3], color='k', linestyle='--')
                ax.set_xlim(-5, 505)
                ax.set_title(title[i])
            plt.savefig(path+"/cartpole_results.png", bbox_inches='tight',
                        pad_inches=0.1)

            plt.close()

if __name__ == '__main__':
    warnings.filterwarnings(action='once')
    parser = argparse.ArgumentParser()
    parser.add_argument('--f', type=str, default='results/', help='folder name')
    parser.add_argument('--i', type=int, help='iteration that should be loaded')
    parser.add_argument('--n_traj', default='10' ,type=int, help='number of trajectories we want to sample for the visualization')
    parser.add_argument('--policy_path', type=str)
    parser.add_argument('--expert_data_path', type=str)
    parser.add_argument('--no_plot_traj', action='store_false')

    args = parser.parse_args()

    summarize_and_plot_results(args.f, args.i, args.n_traj, args.policy_path, args.expert_data_path, args.no_plot_traj)