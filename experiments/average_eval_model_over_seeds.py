# Copyright (c) 2018 Copyright holder of the paper Generative Adversarial Model Learning
# submitted to NeurIPS 2019 for review
# All rights reserved.

from experiments.eval_model import summarize_and_plot_results
import argparse
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import warnings

if __name__ == '__main__':
    warnings.filterwarnings(action='once')
    parser = argparse.ArgumentParser()
    parser.add_argument('--f', type=str, default='results/', help='folder name')
    parser.add_argument('--i', type=int, help='iteration that should be loaded')
    parser.add_argument('--n_traj', default='10' ,type=int, help='number of trajectories we want to sample for the visualization')
    parser.add_argument('--replot', help='if each plot should be replotted', action='store_true')
    parser.add_argument('--policy_path', type=str)
    parser.add_argument('--x_var', type=float)
    parser.add_argument('--theta_var', type=float)
    parser.add_argument('--friction', type=float)
    parser.add_argument('--expert_data_path', type=str)
    parser.add_argument('--obs_noise', action="store_true")
    parser.add_argument('--no_plot_traj', action="store_false")

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

    dataframes = []
    dataframes_learning_curves = []
    # go over each seed and fetch the result to merge them into one large data_frame
    # if result is missing compute them using eval_model

    for fn in os.listdir(experiment_master_folder_name):
        path = os.path.join(experiment_master_folder_name, fn)
        if not os.path.exists(path) or not os.path.isdir(path):
            continue
        seed = str(path).split("-")[-1]

        if os.path.exists(path + "/results_plot_frame_"+ str(args.i)+".csv") and os.path.exists(path + "/learning_curve_plot_frame_"+ str(args.i)+".csv") and not args.replot:
            pass
        else:
            summarize_and_plot_results(path+ "/", args.i, args.n_traj, args.policy_path, args.expert_data_path, args.x_var, args.theta_var, args.friction,
                                       args.obs_noise, plot_trajectories=args.no_plot_traj)

        df = pd.read_csv(path + "/results_plot_frame_" + str(args.i) + ".csv")
        df['seed'] = seed
        dataframes.append(df)

        df_learning_curves = pd.read_csv(path + "/learning_curve_plot_frame_"+ str(args.i)+".csv")
        if not df_learning_curves.empty:
            df_learning_curves["seed"] = seed
            dataframes_learning_curves.append(df_learning_curves)

    def save_fig_and_subfig(dataname, grid):
        f = grid.fig
        axes = grid.axes
        # save the whole plot
        f.savefig(experiment_master_folder_name + dataname + ".png", bbox_inches='tight', pad_inches=0.1)
        # save the subplots in individual files
        for ax, label in zip(axes, grid.row_names):
            bbox = ax[0].get_tightbbox(f.canvas.get_renderer())
            f.savefig(experiment_master_folder_name + dataname + "_" + label + ".png",
                      bbox_inches=bbox.transformed(f.dpi_scale_trans.inverted()))

    result = pd.concat(dataframes)
    g = sns.FacetGrid(result, row="eval_type", sharex=False)

    def plot_means_and_obs(x, y, **kwargs):
        sns.stripplot(x, y , dodge=True, palette='dark', alpha=.25, jitter=True, zorder=1, **kwargs)
        sns.pointplot(x, y, join=False, palette='dark', markers="d", scale=.75, ci=None, **kwargs)

    g.map(plot_means_and_obs, "value", "variant")
    save_fig_and_subfig("result_"+str(args.i), g)

    if len(dataframes_learning_curves) > 0:
        learning_curves = pd.concat(dataframes_learning_curves)
        g = sns.FacetGrid(learning_curves, row="eval_type", sharey=False)
        g.map_dataframe(sns.tsplot, time="Iteration", value="value", unit="seed", condition="variant", color="deep")
        save_fig_and_subfig("learning_curves", g)