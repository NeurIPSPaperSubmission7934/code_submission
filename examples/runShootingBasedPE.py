# Copyright (c) 2018 Copyright holder of the paper Generative Adversarial Model Learning
# submitted to NeurIPS 2019 for review
# All rights reserved.

# Run this snippet with the GAML-casadi conda environment (the other environment is not compatible with casadi, uses pytorch 0.4.0)

### Code snippet to run shooting based parameter estimation methods ###
# the the shooting based parameter estimation methods were implemented using casadi

import experiments.train_FM_param_PE
import warnings
warnings.filterwarnings('ignore')

dataset_num_traj = 10 # parameter specifying which dataset (how many demonstrations) we take for training
seed = 11

experiments.train_FM_param_PE.runSingleShooting(dataset_num_traj, seed)
experiments.train_FM_param_PE.runMultipleShooting(dataset_num_traj, seed)
