# Copyright (c) 2018 Copyright holder of the paper Generative Adversarial Model Learning
# submitted to NeurIPS 2019 for review
# All rights reserved.

import argparse
import pandas as pd
import warnings
import os
import seaborn as sns
import matplotlib.pyplot as plt
import math
import numpy as np
import joblib

if __name__ == '__main__':
    warnings.filterwarnings(action='once')
    parser = argparse.ArgumentParser()
    parser.add_argument('--f', type=str, default='/dlc/Employees/thh2rng/ICML/fixed-FM-param-furuta-comparison-sample-efficiency/', help='folder name')
    # parser.add_argument('--f', type=str,
    #                     default='/dlc/Employees/thh2rng/ICML/fixed-FM-param-cartpole-obs_noise_friction-gumbel-deterministic-comparison/',
    #                     help='folder name')
    # parser.add_argument('--f', type=str, default='/dlc/Employees/thh2rng/ICML/fixed-FM-param-cartpole-obs_noise_friction-gumbel-train-var/', help='folder name')
    # parser.add_argument('--f', type=str,
    #                     default='/dlc/Employees/thh2rng/ICML/fixed-FM-param-cartpole-obs_noise_friction-sample_efficiency-comparison2/',
    #                     help='folder name')
    args = parser.parse_args()

    experiment_master_folder_name = args.f

    num_seeds = 0
    num_experiments = 0
    # count the seeds
    # find all trials of the experiment
    for fn in os.listdir(experiment_master_folder_name):
        # need to go one deeper
        subfolder = os.path.join(experiment_master_folder_name, fn)
        if not os.path.exists(subfolder) or not os.path.isdir(subfolder):
            continue
        for fn2 in os.listdir(subfolder):
            path = os.path.join(subfolder, fn2)
            if not os.path.exists(path) or not os.path.isdir(path):
                continue
            num_seeds += 1

        num_experiments += 1

    print("total number of seeds:", num_seeds)
    print("num_experiments:", num_experiments)

    dataframes = []
    #types = ["GAML", "MLE", "MultipleShooting500", "MultipleShooting250", "MultipleShooting50", "MultipleShooting20", "SingleShooting"]
    # types = ["MLE", "SingleShooting", "MultipleShooting250", "MultipleShooting100", "MultipleShooting50", "MultipleShooting20","GAML", "GAML2"]
    # types = ["MLE", "SingleShooting", "MultipleShooting50", "MultipleShooting100", "MultipleShooting250",
    #          "MultipleShooting20", "GAML"]
    # types = ["GAML", "MLE", "MultipleShooting20", "MultipleShooting50", "MultipleShooting100", "MultipleShooting250", "SingleShooting"]
    # types = ["MultipleShooting20", "GAML", "MLE", "SingleShooting", "MultipleShooting50", "MultipleShooting250", "MultipleShooting100"]
    # types = ["GAML", "SingleShooting", "MLE", "MultipleShooting500", "MultipleShooting250", "MultipleShooting50","MultipleShooting20"]
    types = ["GAML", "MLE", "MultipleShooting", "SingleShooting", ]
    types = ["GAML", "SingleShooting", "MLE", "MultipleShooting"]
    trajs = [1, 5, 10, 15, 20]
    # trajs = [1, 5, 10, 25, 50, 100]
    experiment_itr = 0
    for fn in os.listdir(experiment_master_folder_name):
        # need to go one deeper
        subfolder = os.path.join(experiment_master_folder_name, fn)
        if not os.path.exists(subfolder) or not os.path.isdir(subfolder):
            continue
        print(subfolder)
        for fn2 in os.listdir(subfolder):
            path = os.path.join(subfolder, fn2)
            if not os.path.exists(path) or not os.path.isdir(path):
                continue
            seed = str(path).split("-")[-1]
            displacementFolderPaths = [filename for filename in os.listdir(path) if filename.startswith("displacements")]
            if len(displacementFolderPaths) == 0:
                continue
            if len(displacementFolderPaths) > 1:
                newest_date = os.path.getmtime(path + "/" + displacementFolderPaths[0])
                open_file_path = path + "/" + displacementFolderPaths[0]
                # we have multiple files need to select the right one to plot
                for plot_frame_file in displacementFolderPaths:
                    date = os.path.getmtime(path + "/" + plot_frame_file)
                    if newest_date < date:
                        newest_date = os.path.getmtime(path + "/" + plot_frame_file)
                        open_file_path = path + "/" + plot_frame_file
                displacementFolderPath = open_file_path
            else:
                displacementFolderPath = displacementFolderPaths[0]

            print("found file", displacementFolderPath)
            # go over all different settings for the experiment and add their displacement to the dataframe
            traj = 0
            for displacementFN in os.listdir(os.path.join(path, displacementFolderPath)):
                displacementPath = os.path.join(path, displacementFolderPath, displacementFN)
                if not os.path.exists(displacementPath) or os.path.isdir(displacementPath) or not displacementPath.endswith('.pkl'):
                    continue

                print("open file", displacementPath)
                displacementDicts = joblib.load(displacementPath)
                avg_displacements = displacementDicts["avg_norm_displacements"]

                # create a dataframe containing all avg_displacements
                df = pd.DataFrame({"value":avg_displacements, "type":types[experiment_itr], "trajs":trajs[traj], "seed":seed, "time":np.arange(len(avg_displacements))})
                dataframes.append(df)

                traj += 1

        experiment_itr += 1

    # create data frame for expert data
    displacementsDict = joblib.load(experiment_master_folder_name + "expertDisplacements.pkl")
    avg_displacements = displacementsDict["avg_norm_displacements"]
    for traj in trajs:
        df = pd.DataFrame({"value": avg_displacements, "type": "True Environment", "trajs": traj, "seed": seed,
                           "time": np.arange(len(avg_displacements))})
        dataframes.append(df)

    # merge the all dataframes into one large frame and plot it
    results = pd.concat(dataframes)

    def save_fig_and_subfig(dataname, grid):
        f = grid.fig
        axes = grid.axes

        for ax, label in zip(axes[0], grid.col_names):
            bbox = ax.get_tightbbox(f.canvas.get_renderer())
            f.savefig(experiment_master_folder_name + dataname + "_" + str(label) + ".png",
                      bbox_inches=bbox.transformed(f.dpi_scale_trans.inverted()))
            ax.set_ylim(0, 4)

        grid.add_legend()
        # save the whole plot
        f.savefig(experiment_master_folder_name + dataname + ".png", bbox_inches='tight', pad_inches=0.1)
        f.savefig(experiment_master_folder_name + dataname + ".pgf", bbox_inches='tight', pad_inches=0.1)

    # typesHueOrder = ["True Environment", "GAML", "MLE", "MultipleShooting250", "MultipleShooting100", "MultipleShooting50", "MultipleShooting20", "SingleShooting"]
    typesHueOrder = ["True Environment", "GAML", "MLE", "SingleShooting", "MultipleShooting500", "MultipleShooting250", "MultipleShooting50", "MultipleShooting20"]
    typesHueOrder = ["True Environment", "GAML", "MLE", "SingleShooting", "MultipleShooting"]

    # g = sns.FacetGrid(hue="type", col="trajs", data=results, hue_order=typesHueOrder)
    # g.map(sns.lineplot, "time", "value", n_boot=50)
    #
    # save_fig_and_subfig("displacements_norm_mean", g)
    # joblib.dump(g, os.path.join(experiment_master_folder_name, "figure_norm_mean.pkl"))

    g = sns.FacetGrid(hue="type", col="trajs", data=results, hue_order=typesHueOrder)
    g.map(sns.lineplot, "time", "value", n_boot=50, estimator="Custom")

    save_fig_and_subfig("displacements_norm_median", g)
    joblib.dump(g, os.path.join(experiment_master_folder_name, "figure_norm_median.pkl"))

    plt.show()