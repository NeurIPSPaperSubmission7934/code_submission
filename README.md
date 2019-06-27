# Code for paper Generative Adversarial Model Learning (#7934) submitted to NeurIPS 2019 for review 

Dear reviewers,

this repository contains a cleaned up minimal version of the implementation of GAML and the used baselines, which can be run on a single machine. The code was not optimized to run the experiments on a single machine. Hence, the scripts in the example folder show exemplary the how to run the computation for one single seed and one single dataset for the cartpole swingup environment. For the experiments in the paper the script was run parallel for 10 different seeds and different datasets (number of demonstrated trajectories).

## Installation and instructions to run the code

### Installation:
1. Clone the repository 
2. Create 2 new conda environment via the provided gaml.yml and gaml_casadi.yml (2 environments are needed as the shooting based parameter estimation methods were implemented using casadi, which was not compatible to pytorch 0.41)

### How to run the code
* The run scripts in the example folder start training for the different methods and save the results in the results folder
* The runEvaulation.py script loads in the results of the learned model and generates plots
* More information can be found in the scripts
