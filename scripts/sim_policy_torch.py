import argparse

import joblib

from rllab.misc.console import query_yes_no
from rllab.sampler.utils import rollout_torch

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
    args = parser.parse_args()

    data = joblib.load(args.file)
    policy = data['policy']
    env = data['env']
    env.render_mode_render_env = args.show_discretized
    while True:
        path = rollout_torch(env, policy, max_path_length=args.max_path_length,
                       animated=True, speedup=args.speedup)
        if not query_yes_no('Continue simulation?'):
            break
