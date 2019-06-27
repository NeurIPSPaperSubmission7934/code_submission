# Copyright (c) 2018 Copyright holder of the paper Generative Adversarial Model Learning
# submitted to NeurIPS 2019 for review
# All rights reserved.

# Run this snippet with the GAML conda environment (uses PyTorch 0.41)

### Code snippet to run MLE ###

import experiments.train_FM_param
import warnings
warnings.filterwarnings('ignore')

dataset_num_traj = 10 # parameter specifying which dataset (how many demonstrations) we take for training
seed = 11

# starts the training via MLE with default parameters and saves the results in the results folder for plotting
experiments.train_FM_param.run_MLE_training(dataset_num_traj, seed)
