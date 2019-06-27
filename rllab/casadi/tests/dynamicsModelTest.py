# Copyright (c) 2018 Copyright holder of the paper Generative Adversarial Model Learning
# submitted to NeurIPS 2019 for review
# All rights reserved.

import unittest
import numpy as np
import torch
import casadi as ca
from rllab.dynamic_models.cartpole_model import CartPoleModel
from rllab.dynamic_models.furuta_model import FurutaModel
from rllab.casadi.dynamic_models.cartpole_model_casadi import CartpoleModelCasadi
from rllab.casadi.dynamic_models.furuta_pendulum_model_casadi import FurutaPendulumModelCasadi
from rllab.casadi.dynamic_models.integrators import euler_integration, rk_integration
from rllab.policies.replay_control_policy import ReplayControlPolicy
from rllab.sampler.utils import rollout_torch, rollout

class TestDynamicsModel(unittest.TestCase):

    def testCartpoleDynamics(self):
        ########### PyTorch system ###############

        pyTorchEnv = CartPoleModel(initRandom=False)

        ########### casadi system ################

        casadiEnv = CartpoleModelCasadi()

        states, states_d, controls, params = casadiEnv.buildDynamicalSystem()

        ode = ca.Function('ode', [states, controls, params], [states_d])

        # now test output of casadi vs pytorch model for different states and controls and params
        # note that the casadi model gives only the mean

        ##### test random 100 states and input for 10 different params
        for i_param in range(10):
            param = np.abs(np.random.rand(3))
            pyTorchEnv.set_param_values(torch.from_numpy(np.concatenate([param, np.zeros(2)])).float())
            for i in range(100):
                state_input = np.random.rand(4)*5
                controls = np.random.rand(1)*5
                casadi_result = np.array(ode(state_input, controls, param), dtype=np.float32).squeeze()
                pytorch_result = pyTorchEnv.forward(torch.from_numpy(np.concatenate([state_input, controls])[np.newaxis,:]).float())[0].detach().numpy().squeeze()
                np.testing.assert_allclose(casadi_result[[1,3]], pytorch_result, rtol=1e-2, atol=1e-4)

    def test_cartpole_integrator(self):

        ########### PyTorch system ###############

        pyTorchEnv = CartPoleModel(initRandom=False)
        param_truth = pyTorchEnv.theta.detach().numpy()
        # set the variance to 0 to have a determinisitic environmenht
        pyTorchEnv.set_param_values(torch.from_numpy(np.concatenate([param_truth, np.zeros(2)])).float())

        ########### casadi system ################

        casadiEnv = CartpoleModelCasadi()

        states, states_d, controls, params = casadiEnv.buildDynamicalSystem()

        euler_func = euler_integration(states, states_d, controls, pyTorchEnv.tau, integrator_stepsize=1, angular_idx=[2])

        step = ca.Function("step", [states, controls, params], [euler_func])

        ############ Simulating the system ##########

        for traj in range(10): # simulate in total 10 different trajectoriess
            timesteps = 500 # simulate the system for 50 timesteps
            # u_inputs = ca.SX.sym("u_outputs", timesteps)
            #
            # init_z_state = ca.SX(x0)
            # z_states = [init_z_state]
            # for i in range(control_steps):
            #     current_z = step(z_states[-1], u_inputs[i], param_truth)
            #     z_states.append(current_z)

            # simulate 1 trajectory
            sim_one_traj = step.mapaccum("all_steps", timesteps)
            actions = ca.DM(np.random.rand(timesteps))

            policy = ReplayControlPolicy(pyTorchEnv.spec, np.array(actions))
            policy.normalized_input = [False, False, False, False]
            policy.normalized_output = [False]
            path = rollout_torch(pyTorchEnv, policy, timesteps, terminate_only_max_path=True)

            x0 = ca.DM(path["observations"][0,:])
            sim_states = sim_one_traj(x0, actions, ca.repmat(param_truth, 1, timesteps))

            np.testing.assert_allclose(np.array(sim_states.T), path["next_observations"],rtol=1e-2, atol=1e-4)

    def testFurutaPendulumModel(self):
        ########### PyTorch system ###############

        pyTorchEnv = FurutaModel(initRandom=False)
        param_truth = pyTorchEnv.theta.detach().numpy()

        ########### casadi system ################

        casadiEnv = FurutaPendulumModelCasadi(trainMotor=False)

        states, states_d, controls, params = casadiEnv.buildDynamicalSystem()

        ode = ca.Function('ode', [states, controls, params], [states_d])

        # now test output of casadi vs pytorch model for different states and controls and params
        # note that the casadi model gives only the mean

        ##### test random 100 states and input for 10 different params
        for i_param in range(10):
            param = np.abs(np.random.rand(7))
            pyTorchEnv.set_param_values(torch.from_numpy(np.concatenate([param, np.zeros(2)])).float())
            for i in range(100):
                state_input = np.random.rand(4) * 5
                controls = np.random.rand(1) * 5
                casadi_result = np.array(ode(state_input, controls, param), dtype=np.float32).squeeze()
                pytorch_result = \
                pyTorchEnv.forward(torch.from_numpy(np.concatenate([state_input, controls])[np.newaxis, :]).float())[
                    0].detach().numpy().squeeze()
                np.testing.assert_allclose(casadi_result[[2, 3]], pytorch_result, rtol=1e-2, atol=1e-4)

        ###### do a test for the true parameter
        pyTorchEnv.set_param_values(torch.from_numpy(np.concatenate([param_truth, np.zeros(2)])).float())
        for i in range(100):
            state_input = np.random.rand(4) * 5
            controls = np.random.rand(1) * 5
            casadi_result = np.array(ode(state_input, controls, param), dtype=np.float32).squeeze()
            pytorch_result = \
                pyTorchEnv.forward(torch.from_numpy(np.concatenate([state_input, controls])[np.newaxis, :]).float())[
                    0].detach().numpy().squeeze()
            np.testing.assert_allclose(casadi_result[[2, 3]], pytorch_result, rtol=1e-2, atol=1e-4)


    def test_furuta_pendulum_integrator(self):

        ########### PyTorch system ###############

        pyTorchEnv = FurutaModel(initRandom=False, hackOn=False)
        param_truth = pyTorchEnv.theta.detach().numpy()
        # set the variance to 0 to have a determinisitic environmenht
        pyTorchEnv.set_param_values(torch.from_numpy(np.concatenate([param_truth, np.zeros(2)])).float())

        ########### casadi system ################

        casadiEnv = FurutaPendulumModelCasadi(trainMotor=False)

        states, states_d, controls, params = casadiEnv.buildDynamicalSystem()

        rk4_func = rk_integration(states, states_d, controls, pyTorchEnv.tau, integrator_stepsize=1, angular_idx=[0, 1])

        step = ca.Function("step", [states, controls, params], [rk4_func])

        # actions = np.random.rand(1, 1)
        #
        # policy = ReplayControlPolicy(pyTorchEnv.spec, np.array(actions))
        # policy.normalized_input = [False, False, False, False]
        # policy.normalized_output = [False]
        # path = rollout_torch(pyTorchEnv, policy, 1, terminate_only_max_path=True)
        #
        # x0 = ca.DM(path["observations"][0,:])
        # sim_states = step(x0, ca.DM(actions), param_truth)
        #
        # print(sim_states)

        ############ Simulating the system ##########

        for traj in range(10): # simulate in total 10 different trajectoriess
            timesteps = 100 # simulate the system for 50 timesteps
            # u_inputs = ca.SX.sym("u_outputs", timesteps)
            #
            # init_z_state = ca.SX(x0)
            # z_states = [init_z_state]
            # for i in range(control_steps):
            #     current_z = step(z_states[-1], u_inputs[i], param_truth)
            #     z_states.append(current_z)

            # simulate 1 trajectory
            sim_one_traj = step.mapaccum("all_steps", timesteps)
            actions = ca.DM(np.random.rand(timesteps))

            policy = ReplayControlPolicy(pyTorchEnv.spec, np.array(actions))
            policy.normalized_input = [False, False, False, False]
            policy.normalized_output = [False]
            path = rollout_torch(pyTorchEnv, policy, timesteps, terminate_only_max_path=True)

            x0 = ca.DM(path["observations"][0,:])
            sim_states = sim_one_traj(x0, actions, ca.repmat(param_truth, 1, timesteps))

            print(np.abs(np.array(sim_states.T)-path["next_observations"]).sum())

            np.testing.assert_allclose(np.array(sim_states.T), path["next_observations"],rtol=1e-2, atol=1e-4)