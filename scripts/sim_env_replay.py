import argparse

import joblib
import torch

from rllab.misc.console import query_yes_no
from rllab.sampler.utils import rollout
from rllab.policies.replay_control_policy import ReplayControlPolicy
from rllab.envs.cartpole_swingup_env import CartPoleSwingUpEnv

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
    args = parser.parse_args()

    data = joblib.load(args.file)
    expert_data = data['expert_data']
    imitation_env = data['imitationModel']
    print(imitation_env.theta)
    print(imitation_env.std)
    #env = CartPoleSwingUpEnv()
    #policy = ReplayControlPolicy(env_spec=env.spec, replay_actions=expert_data["paths"][0]["actions"])

    data = joblib.load(
        "/local/data/r4/thh2rng/rllab/data/local/experiment/experiment_2018_11_09_14_11_15_0001/itr_1000.pkl")
    policy = data['policy']

    imitation_env.load_state_dict(torch.load(args.file[:-4]+"_model.pkl"))
    while True:
        path = rollout(imitation_env, policy, max_path_length=args.max_path_length,
                       animated=True, speedup=args.speedup)
        if not query_yes_no('Continue simulation?'):
            break
