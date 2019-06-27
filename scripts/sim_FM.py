import argparse

import joblib
import tensorflow as tf
import torch

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
    args = parser.parse_args()

    with tf.Session() as sess:
        data = joblib.load(args.file)
        imitation_env = data['imitationModel']
        imitation_env.load_state_dict(torch.load(args.file[:-4]+"_model.pkl"))
        if args.policy_path is None:
            policy = data['policy']
        else:
            policy_data = joblib.load(args.policy_path)
            policy = policy_data['policy']
        for param in imitation_env.parameters():
            print("params", param)
        while True:
            path = rollout_torch(imitation_env, policy, max_path_length=args.max_path_length,
                           animated=True, speedup=args.speedup)
            if not query_yes_no('Continue simulation?'):
                break
