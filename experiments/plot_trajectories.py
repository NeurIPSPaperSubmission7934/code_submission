# Copyright (c) 2018 Copyright holder of the paper Generative Adversarial Model Learning
# submitted to NeurIPS 2019 for review
# All rights reserved.

import argparse
import pandas as pd
import warnings
import os
import seaborn as sns
import numpy as np
import joblib
import matplotlib.pyplot as plt

if __name__ == '__main__':
    warnings.filterwarnings(action='once')
    parser = argparse.ArgumentParser()
    parser.add_argument('--f', type=str, default='results/', help='folder name')
    parser.add_argument('--i', type=int)
    parser.add_argument('--expert_data_path', type=str)
    parser.add_argument('--n', type=int)
    parser.add_argument('--plot_n', type=int)
    parser.add_argument('--algo', type=str)
    parser.add_argument('--env', type=str)

    args = parser.parse_args()

    path = args.f
    iteration = args.i
    expert_data_path = args.expert_data_path
    sample_n_traj = args.n
    plot_n_traj = args.plot_n
    experiment_folder_name = path

    # load fixed expert trajectories
    expert_paths = joblib.load(expert_data_path)

    if os.path.exists(
            os.path.join(path, "itr_" + str(iteration) + "_n_traj_" + str(sample_n_traj) + "_generated_paths.pkl")) and \
            os.path.exists(os.path.join(path, "itr_" + str(iteration) + "_n_traj_" + str(
                sample_n_traj) + "_generated_processed_paths.pkl")):
        generated_paths = joblib.load(
            os.path.join(path, "itr_" + str(iteration) + "_n_traj_" + str(sample_n_traj) + "_generated_paths.pkl"))
        generated_processed_paths = joblib.load(os.path.join(path, "itr_" + str(iteration) + "_n_traj_" + str(
            sample_n_traj) + "_generated_processed_paths.pkl"))

        if args.env == "cartpole":
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
            plt.savefig("/home/thh2rng/Plots/cartpole_results_"+ args.algo +"_new2.png", bbox_inches='tight',
                                    pad_inches=0.1)
            plt.savefig("/home/thh2rng/Plots/cartpole_results_"+ args.algo +"_new2.pgf", bbox_inches='tight',
                                    pad_inches=0.1)
            plt.show()

        elif args.env == "furuta":
            tau = 1 / 100
            # create plot with 25 trajectory from real env and learned env
            sns.set(context="notebook", style="darkgrid")
            title = ['theta1', 'cos_theta2', 'sin_theta2', 'theta1_dot', 'theta2_dot']
            fig, axes = plt.subplots(len(title), 1, tight_layout=True)
            # the size of A4 paper + 4 inch addtional height
            fig.set_size_inches(11.7, 12.27)
            # merge all observations of all paths into one big data frame
            dataFrames = []
            print("building data frame")
            for i in range(sample_n_traj):
                trajlen = expert_paths[i]["observations"].shape[0]
                observations = expert_paths[i]["observations"]
                cosalpha = np.cos(observations[:, [1]])
                sinalpha = np.sin(observations[:, [1]])
                trajDataFrame = pd.DataFrame(np.concatenate(
                    [i * np.ones(trajlen)[:, np.newaxis], np.arange(trajlen)[:, np.newaxis] * tau,
                     observations[:, [0]], cosalpha, sinalpha, observations[:, 2:]], axis=1),
                    columns=['traj', 'timestep'] + title)
                dataFrames.append(trajDataFrame)
            all_traj_true_env = pd.concat(dataFrames)
            all_traj_true_env["model"] = "true environment"

            # now add 10 traj from learned env
            dataFrames = []
            for i in range(sample_n_traj):
                traj_len = generated_paths[i]["observations"].shape[0]
                observations = generated_paths[i]["observations"]
                cosalpha = np.cos(observations[:, [1]])
                sinalpha = np.sin(observations[:, [1]])
                trajDataFrame = pd.DataFrame(np.concatenate(
                    [i * np.ones(trajlen)[:, np.newaxis], np.arange(traj_len)[:, np.newaxis] * tau,
                     observations[:, [0]], cosalpha, sinalpha, observations[:, 2:]], axis=1),
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
                ax.set_title(title[i])
            plt.savefig("/home/thh2rng/Plots/furuta_results_" + args.algo + ".png", bbox_inches='tight',
                        pad_inches=0.1)
            plt.savefig("/home/thh2rng/Plots/furuta_results_" + args.algo + ".pgf", bbox_inches='tight',
                        pad_inches=0.1)
            plt.show()

