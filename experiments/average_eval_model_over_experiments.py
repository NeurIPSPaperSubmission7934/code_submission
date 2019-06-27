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
from pathlib import Path

def loadAndPlotResults():
    mainDir = Path(__file__).parents[1]
    # load the expert data frame
    # rename eval_types
    replace_dict = {'avg_undiscounted_return': 'diff_avg_undiscounted_return',
                    'avg_discounted_return': 'diff_avg_discounted_return',
                    'avg_success_rate': 'diff_avg_success_rate',
                    'avg_boundaries_left': 'diff_avg_boundaries_left',
                    'avg_traj_length': 'diff_avg_traj_length',
                    'avg_norm_displacement': 'avg_norm_displacement',
                    }

    dataframes = []
    expert_dataframes = []

    expertPlotFramePath = Path.joinpath(mainDir, "datasets/expert_plot.csv")
    expert_df = pd.read_csv(str(expertPlotFramePath))
    expert_df["algo"] = "True Environment"

    # create set with all different expert eval types
    expert_eval_types = set(expert_df["eval_type"])

    expert_values = {}
    for eval_type in expert_eval_types:
        expert_values[eval_type] = expert_df[expert_df["eval_type"] == str(eval_type)]["value"].iloc[0]

    expert_df = expert_df[expert_df.eval_type != 'avg_discounted_return']
    expert_df = expert_df[expert_df.eval_type != 'avg_min_displacement']
    expert_df = expert_df[expert_df.eval_type != 'avg_fixed_action_min_displacement']
    expert_df = expert_df[expert_df.eval_type != 'avg_fixed_action_displacement']
    expert_df = expert_df[expert_df.eval_type != 'avg_displacement']
    expert_df = expert_df[expert_df.eval_type != 'avg_boundaries_left']

    # add the expert df as the last
    for eval_type in expert_eval_types:
        if not (eval_type == 'avg_displacement' or eval_type == 'avg_norm_displacement'):
            expert_df["value"][expert_df[expert_df["eval_type"] == eval_type].index] = expert_df["value"][
                expert_df[expert_df["eval_type"] == eval_type].index].apply(
                lambda x: math.fabs(expert_values[eval_type] - x))
        expert_df["eval_type"][expert_df[expert_df["eval_type"] == eval_type].index] = \
            expert_df["eval_type"][
                expert_df[expert_df["eval_type"] == eval_type].index].apply(
                lambda x: replace_dict[x] if (x in replace_dict) else x)

    dataframes.append(expert_df)

    algorithms = ["GAML", "MLE", "MultipleShooting", "SingleShooting"]

    for algo in algorithms:
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

        path = str(resultsPath)
        # load the results data frame for algorithm
        plot_frame_files = [filename for filename in os.listdir(path) if filename.startswith("results_plot_frame_")]
        if len(plot_frame_files) == 0:
            continue
        if len(plot_frame_files) > 1:
            newest_date = os.path.getmtime(path + "/" + plot_frame_files[0])
            open_file_path = path + "/" + plot_frame_files[0]
            # we have multiple files need to select the right one to plot
            for plot_frame_file in plot_frame_files:
                date = os.path.getmtime(path + "/" + plot_frame_file)
                if newest_date < date:
                    newest_date = os.path.getmtime(path + "/" + plot_frame_file)
                    open_file_path = path + "/" + plot_frame_file
            print("found file", open_file_path)
            df = pd.read_csv(open_file_path)
        else:
            print("found file", plot_frame_files[0])
            df = pd.read_csv(path + "/" + plot_frame_files[0])
        df["algo"] = algo

        # filter extreme values
        df["value"][df[df["eval_type"] == 'kl_div'].index] = df["value"][df[df["eval_type"] == 'kl_div'].index].apply(
            lambda x: min(x, 50))
        df["value"][df[df["eval_type"] == 'inverse_kl_div'].index] = df["value"][
            df[df["eval_type"] == 'inverse_kl_div'].index].apply(lambda x: min(x, 50))
        df["value"][df[df["eval_type"] == 'expected_kl_div_to_learned_model'].index] = df["value"][
            df[df["eval_type"] == 'expected_kl_div_to_learned_model'].index].apply(lambda x: min(x, 50))
        df["value"][df[df["eval_type"] == 'pseudo_KL'].index] = df["value"][
            df[df["eval_type"] == 'pseudo_KL'].index].apply(lambda x: max(x, -50))

        df = df[df.eval_type != 'avg_discounted_return']
        df = df[df.eval_type != 'avg_min_displacement']
        df = df[df.eval_type != 'avg_fixed_action_min_displacement']
        df = df[df.eval_type != 'avg_fixed_action_displacement']
        df = df[df.eval_type != 'avg_displacement']
        df = df[df.eval_type != 'avg_success_rate']
        df = df[df.eval_type != 'avg_boundaries_left']

        # we want absolute difference to expert environment
        # rename eval_types
        for eval_type in expert_eval_types:
            if not (eval_type == 'avg_displacement' or eval_type == 'avg_norm_displacement'):
                df["value"][df[df["eval_type"] == eval_type].index] = df["value"][
                    df[df["eval_type"] == eval_type].index].apply(lambda x: math.fabs(expert_values[eval_type] - x))

            df["eval_type"][df[df["eval_type"] == eval_type].index] = df["eval_type"][
                df[df["eval_type"] == eval_type].index].apply(lambda x: replace_dict[x] if (x in replace_dict) else x)

        dataframes.append(df)

    # put all dataframes in one frame and plot it
    result = pd.concat(dataframes)

    colOrder = ["avg_norm_displacement", "diff_avg_undiscounted_return", "diff_avg_traj_length"]

    g = sns.FacetGrid(result, col="eval_type", sharey=True, sharex=False, col_order=colOrder)

    g.map(sns.barplot, "value", "algo", palette="deep")

    # set custom title
    ax = g.axes.flatten()
    for a, label in zip(ax, ["Average displacement", "Difference in average undiscounted return", "Difference in average trajectory length"]):
        a.set_title(label, {'fontsize': 9})

    plt.tight_layout(pad=2.5, w_pad=5.0)
    plt.show()

if __name__ == '__main__':
    warnings.filterwarnings(action='once')
    parser = argparse.ArgumentParser()
    parser.add_argument('--f', type=str, default='results/', help='folder name')
    parser.add_argument('--env', type=str)
    parser.add_argument('--load_csv', action='store_true')

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

    # rename eval_types
    replace_dict = {'avg_undiscounted_return': 'diff_avg_undiscounted_return',
                    'avg_discounted_return': 'diff_avg_discounted_return',
                    'avg_success_rate': 'diff_avg_success_rate',
                    'avg_boundaries_left': 'diff_avg_boundaries_left',
                    'avg_traj_length': 'diff_avg_traj_length',
                    'avg_norm_displacement': 'avg_displacement',
                    }

    dataframes = []
    expert_dataframes = []
    #types = ["GAML", "MLE"]
    # types = ["MLE", "GAML", "SingleShooting", "MultipleShooting20"]
    #types = ["MultipleShooting20", "MultipleShooting50", "MultipleShooting100", "MultipleShooting250", "SingleShooting"]
    # types = ["GAML", "MLE", "MultipleShooting500", "MultipleShooting250", "MultipleShooting50", "MultipleShooting20", "SingleShooting"]
    #types = ["MLE", "SingleShooting", "MultipleShooting250", "MultipleShooting100", "MultipleShooting50", "MultipleShooting20","GAML", "GAML2"]
    # types = ["MLE", "SingleShooting", "MultipleShooting50", "MultipleShooting100", "MultipleShooting250",
    #          "MultipleShooting20", "GAML"]
    #types = ["GAML", "MLE", "MultipleShooting20", "MultipleShooting50", "MultipleShooting100", "MultipleShooting250", "SingleShooting"]
    if args.env == 'cartpole':
        # types = ["MultipleShooting20", "GAML", "MLE", "SingleShooting", "MultipleShooting50", "MultipleShooting250",
        #          "MultipleShooting100"]
        # types = ["GAML", "MLE", "MultipleShooting20", "MultipleShooting50", "MultipleShooting100",
        #          "MultipleShooting250", "SingleShooting",]
        types = ["GAML", "MLE", "MultipleShooting", "SingleShooting",]
    elif args.env == 'furuta':
        types = ["GAML", "SingleShooting", "MLE", "MultipleShooting500", "MultipleShooting250", "MultipleShooting50",
                  "MultipleShooting20"]
        types = ["GAML", "SingleShooting", "MLE", "MultipleShooting"]
    experiment_itr = 0
    read_expert = False

    if args.load_csv:
        result = pd.read_csv(experiment_master_folder_name + "results_plot.csv", sep=',', encoding='utf-8')
    else:

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
                #print("found seed", seed)
                plot_frame_files = [filename for filename in os.listdir(path) if filename.startswith("results_plot_frame_")]
                if len(plot_frame_files) == 0:
                    continue
                if len(plot_frame_files) > 1:
                    newest_date = os.path.getmtime(path + "/" + plot_frame_files[0])
                    open_file_path = path + "/" + plot_frame_files[0]
                    # we have multiple files need to select the right one to plot
                    for plot_frame_file in plot_frame_files:
                        date = os.path.getmtime(path + "/" + plot_frame_file)
                        if newest_date < date:
                            newest_date = os.path.getmtime(path + "/" + plot_frame_file)
                            open_file_path = path + "/" + plot_frame_file
                    print("found file", open_file_path)
                    df = pd.read_csv(open_file_path)
                else:
                    print("found file", plot_frame_files[0])
                    df = pd.read_csv(path + "/" + plot_frame_files[0])
                df['seed'] = seed
                if experiment_itr == 0:
                    max_variant = df["expr_itr"].max()
                df["type"] = types[experiment_itr]

                if args.env == 'cartpole':
                    trajs = ["  1", "  5", " 10", " 25", " 50", "100"]
                elif args.env == 'furuta':
                    trajs = [" 1", " 5", "10", "15", "20"]

                if experiment_itr == 0 and not read_expert:
                    # create expert_data_frame
                    expert_df = pd.read_csv(experiment_master_folder_name + "/expert_plot.csv")
                    df['seed'] = seed
                    expert_df["type"] = "True Environment"
                    expert_df["numTraj"] = expert_df["expr_itr"].apply(lambda x: trajs[x])

                    # create set with all different expert eval types
                    expert_eval_types = set(expert_df["eval_type"])



                    expert_values = {}
                    for eval_type in expert_eval_types:
                        print(eval_type)
                        expert_values[eval_type] = expert_df[expert_df["eval_type"] == str(eval_type)]["value"].iloc[0]

                    expert_df = expert_df[expert_df.eval_type != 'avg_discounted_return']
                    expert_df = expert_df[expert_df.eval_type != 'avg_min_displacement']
                    expert_df = expert_df[expert_df.eval_type != 'avg_fixed_action_min_displacement']
                    expert_df = expert_df[expert_df.eval_type != 'avg_fixed_action_displacement']
                    expert_df = expert_df[expert_df.eval_type != 'avg_displacement']
                    #expert_df = expert_df[expert_df.eval_type != 'avg_traj_length']
                    expert_df = expert_df[expert_df.eval_type != 'avg_success_rate']
                    expert_df = expert_df[expert_df.eval_type != 'avg_boundaries_left']

                    # add the expert df as the last
                    for eval_type in expert_eval_types:
                        if not (eval_type == 'avg_displacement' or eval_type == 'avg_norm_displacement'):
                            expert_df["value"][expert_df[expert_df["eval_type"] == eval_type].index] = expert_df["value"][
                                expert_df[expert_df["eval_type"] == eval_type].index].apply(
                                lambda x: math.fabs(expert_values[eval_type] - x))
                        expert_df["eval_type"][expert_df[expert_df["eval_type"] == eval_type].index] = \
                        expert_df["eval_type"][
                            expert_df[expert_df["eval_type"] == eval_type].index].apply(
                            lambda x: replace_dict[x] if (x in replace_dict) else x)

                    dataframes.append(expert_df)

                    read_expert = True

                df["numTraj"] = df["expr_itr"].apply(lambda x: trajs[x])
                # filter extreme values
                df["value"][df[df["eval_type"] == 'kl_div'].index] = df["value"][df[df["eval_type"] == 'kl_div'].index].apply(lambda x: min(x, 50))
                df["value"][df[df["eval_type"] == 'inverse_kl_div'].index] = df["value"][
                    df[df["eval_type"] == 'inverse_kl_div'].index].apply(lambda x: min(x, 50))
                df["value"][df[df["eval_type"] == 'expected_kl_div_to_learned_model'].index] = df["value"][
                    df[df["eval_type"] == 'expected_kl_div_to_learned_model'].index].apply(lambda x: min(x, 50))
                df["value"][df[df["eval_type"] == 'pseudo_KL'].index] = df["value"][
                    df[df["eval_type"] == 'pseudo_KL'].index].apply(lambda x: max(x, -50))

                df = df[df.eval_type != 'avg_discounted_return']
                df = df[df.eval_type != 'avg_min_displacement']
                df = df[df.eval_type != 'avg_fixed_action_min_displacement']
                df = df[df.eval_type != 'avg_fixed_action_displacement']
                df = df[df.eval_type != 'avg_displacement']
                df = df[df.eval_type != 'avg_traj_length']
                #df = df[df.eval_type != 'avg_success_rate']
                df = df[df.eval_type != 'avg_boundaries_left']

                # we want absolute difference to expert environment
                # rename eval_types
                for eval_type in expert_eval_types:
                    if not(eval_type == 'avg_displacement' or eval_type == 'avg_norm_displacement'):
                        df["value"][df[df["eval_type"] == eval_type].index] = df["value"][
                            df[df["eval_type"] == eval_type].index].apply(lambda x: math.fabs(expert_values[eval_type] - x))

                    df["eval_type"][df[df["eval_type"] == eval_type].index] = df["eval_type"][
                        df[df["eval_type"] == eval_type].index].apply(lambda x: replace_dict[x] if (x in replace_dict) else x)

                dataframes.append(df)

            experiment_itr += 1

        result = pd.concat(dataframes)

        result.to_csv(experiment_master_folder_name + "results_plot.csv", sep=',', encoding='utf-8')

    def save_fig_and_subfig(dataname, grid):
        f = grid.fig
        axes = grid.axes
        # save the subplots in individual files
        # rename col_names where we substract from expert
        replace_dict = {'avg_undiscounted_return':'diff_avg_undiscounted_return',
                        'avg_discounted_return':'diff_avg_discounted_return',
                        'avg_success_rate':'diff_avg_success_rate',
                        'avg_boundaries_left': 'diff_avg_boundaries_left',
                        'avg_traj_length':'diff_avg_traj_length',
                        'avg_norm_displacement':'avg_displacement'
                        }
        new_col_names = []
        for name in grid.col_names:
            if name in replace_dict.keys():
                new_col_names.append(replace_dict[name])
            else:
                new_col_names.append(name)
        for ax, label in zip(axes[0], new_col_names):
            bbox = ax.get_tightbbox(f.canvas.get_renderer())
            if label == 'avg_displacement':
                if args.env == 'cartpole':
                    ax.set_ylim(0.9 , 3)
                if args.env == "furuta":
                    ax.set_ylim(0, 4)
            f.savefig(experiment_master_folder_name + dataname + "_" + label + ".png",
                      bbox_inches=bbox.transformed(f.dpi_scale_trans.inverted()))
        grid.add_legend()
        # save the whole plot
        f.savefig(experiment_master_folder_name + dataname + ".png", bbox_inches='tight', pad_inches=0.1)
        f.savefig(experiment_master_folder_name + dataname + ".pgf", bbox_inches='tight', pad_inches=0.1)

    if args.env == 'cartpole':
        typesHueOrder = ["True Environment", "GAML", "MLE", "SingleShooting", "MultipleShooting250", "MultipleShooting100", "MultipleShooting50",
                  "MultipleShooting20"]
        colOrder= ["avg_displacement", "diff_avg_undiscounted_return", "diff_avg_traj_length"]
    elif args.env == 'furuta':
        typesHueOrder = ["True Environment", "GAML", "MLE", "SingleShooting", "MultipleShooting500", "MultipleShooting250", "MultipleShooting50",
                  "MultipleShooting20"]
        colOrder = ["avg_displacement", "diff_avg_undiscounted_return", "diff_avg_success_rate"]

    typesHueOrder = ["True Environment", "GAML", "MLE", "SingleShooting", "MultipleShooting"]

    d = {'dashes': [(2,2)]+[""] * len(types), "marker": ['']+['o'] * len(types)}
    g = sns.FacetGrid(result, col="eval_type", hue="type", sharey=False,
                      hue_order=typesHueOrder, hue_kws=d , col_order=colOrder)

    def plot_median(x, y, hue, **kwargs):
        ax = sns.lineplot(x,y, estimator="Custom", **kwargs)
        ax.lines[0].set_linestyle("--")

    d = {'dashes': [(2,2)]+[""] * len(types), "marker": ['']+['o'] * len(types)}
    g = sns.FacetGrid(result, col="eval_type", hue="type", sharey=False,
                      hue_order=typesHueOrder, hue_kws=d, col_order=colOrder)
    g.map(plot_median, "numTraj", "value", "type")

    save_fig_and_subfig("/results_median", g)

    plt.show()