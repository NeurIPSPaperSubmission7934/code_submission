import argparse

import joblib
import torch
from pathlib import Path

from rllab.misc.console import query_yes_no
from rllab.sampler.utils import rollout, rollout_torch

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--max_path_length', type=int, default=1000,
                        help='Max length of rollout')
    parser.add_argument('--speedup', type=float, default=1,
                        help='Speedup')
    parser.add_argument('--show_discretized', type=str2bool, default=True, help='Visualize discretized env')
    parser.add_argument('--policy_path', type=str)
    parser.add_argument('--n', type=int)
    args = parser.parse_args()

    filename = args.file
    data = joblib.load(filename)
    if 'imitationModel' in data:
        imitation_env = data['imitationModel']
    else:
        imitation_env = data['imitationEnv']
    parent_path = Path(filename).parent
    imitation_env.load_state_dict(torch.load(str(parent_path)+"/itr_"+str(args.n)+"_model.pkl"))
    if args.policy_path is not None:
        policy_data = joblib.load(args.policy_path)
        policy = policy_data['policy']
    else:
        policy = data['policy']
    policy.normalized_input = [False, False, False, False]
    policy.normalized_output = [True]
    while True:
        path = rollout_torch(imitation_env, policy, max_path_length=args.max_path_length,
                       animated=True, speedup=args.speedup)
        if not query_yes_no('Continue simulation?'):
            break
