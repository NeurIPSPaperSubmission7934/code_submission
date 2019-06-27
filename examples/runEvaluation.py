# Copyright (c) 2018 Copyright holder of the paper Generative Adversarial Model Learning
# submitted to NeurIPS 2019 for review
# All rights reserved.

# Run this code with the GAML conda environment (uses PyTorch 0.35)

### Code snippet for evaluation ###

# It loads the learned models, generates trajectories using the policy and plots the results.
# This script uses a more simplified script for generating the plots, as the original plotting script used in the paper
# assumes results for multiple seeds and multiple number of used trajectories (1, 5, 10, 25, 50, 100).
# However, obtaining all the results on a single PC would take a long time. Hence, we offer this simplified evaluation
# script to show the results on a single seed and one data set of a fixed number of trajectories

# load the results of the different algorithms if they exist and generate trajectories
# plots 25 trajectories and saves it in cartpole_results.png in the same folder where the model is located
from experiments.eval_model import processResults
import warnings
warnings.filterwarnings('ignore')
processResults("GAML")
processResults("MLE", )
processResults("SingleShooting")
processResults("MultipleShooting")

# create a simple comparison plot between the 4 methods
# IMPORTANT: the plotting script is simplified such that it assumes that the results folder contains only one result for each algorithm
from experiments.average_eval_model_over_experiments import loadAndPlotResults
loadAndPlotResults()

