# Copyright (c) 2018 Copyright holder of the paper Generative Adversarial Model Learning
# submitted to NeurIPS 2019 for review
# All rights reserved.

import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from collections import OrderedDict
from rllab.misc.overrides import overrides
from rllab.envs.base import Env
from rllab.envs.furuta_pendulum_env import FurutaPendulumEnv
from rllab import spaces
from rllab.torch.utils import torch as torch_utils
import itertools
from rllab.torch.utils.misc import batch_diagonal
from qube.furuta.estimator import VelocityFilter

# TODO: we might need to change everything from double to float to work on the GPU, on the other hand do we need GPU?

class FurutaModel(Env):
    # Sampling frequency
    fs = 100.0
    tau = 1 / fs

    # Gravity
    g = 9.81

    # Motor
    Rm = 8.4  # resistance
    kt = 0.042  # current-torque (N-m/A)
    km = 0.042  # back-emf constant (V-s/rad)

    # Rotary arm
    Mr = 0.095  # mass (kg)
    Lr = 0.085  # length (m)
    lr = 0.5 # where the COM is
    Jr = Mr * Lr ** 2 / 12  # moment of inertia about COM (kg-m^2)
    Dr = 0.0003  # equivalent viscous damping coefficient (N-m-s/rad)

    # Pendulum link
    Mp = 0.024  # mass (kg)
    Lp = 0.129  # length (m)
    lp = 0.5 # where the COM is
    Jp = Mp * Lp ** 2 / 12  # moment of inertia about COM (kg-m^2)
    Dp = 0.00005  # equivalent viscous damping coefficient (N-m-s/rad)

    theta_min = -2.3
    theta_max = 2.3
    alpha_threshold_radians = np.pi
    u_max = 18

    u_dim = 1
    x_dim = 4 # we have theta, alpha, theta_dot, alpha_dot

    # have limits to prevent numerical errors, due to increasing velocity
    x_min = (-2.3, -alpha_threshold_radians*1.2, -1e3, -1e3)
    x_max = (2.3, alpha_threshold_radians*1.2, 1e3, 1e3)

    range = 0.2  # shows a 0.4m x 0.4m excerpt of the scene
    arm_radius = 0.003
    arm_length = 0.085
    pole_radius = 0.0045
    pole_length = 0.129

    x_dot_state_idx = [2, 3]
    x_state_idx = [0, 1]

    _version = 1

    def __init__(self, forwardMode=True, initRandom=True, trainMotor=False, init_std=None, hackOn=True, fs=100, torque_controller=True):

        self._parameters = OrderedDict()
        self._wrapped_env = FurutaPendulumEnv(show_gui=False)

        if initRandom:
            theta = torch.tensor(np.random.uniform(size=7), requires_grad=True, dtype=torch.float32)
            self.km_tensor = torch.tensor(np.random.uniform(), requires_grad=True, dtype=torch.float32)
            self.Rm_tensor = torch.tensor(np.random.uniform(), requires_grad=True, dtype=torch.float32)
        else:
            theta = torch.tensor([self.Mp*self.Lr**2+self.Jr,
                                  1/4*self.Mp*self.Lp**2,
                                  1/2*self.Mp*self.Lp*self.Lr,
                                  self.Jp+1/4*self.Mp*self.Lp**2,
                                  1/2*self.Mp*self.Lp,
                                  self.Dr,
                                  self.Dp], dtype=torch.float32, requires_grad=True)
            # theta = torch.tensor([self.Mp*self.Lr**2+self.Jr,self.Mp*(self.lp*self.Lp)**2, self.Mp*self.lp*self.Lp*self.Lr, self.Jp+self.Mp*(self.lp*self.Lp)**2,
            #                             self.Mp*self.lp*self.Lp, self.Dr, self.Dp], dtype=torch.float32, requires_grad=True)
            km_tensor = torch.tensor(self.km, requires_grad=True, dtype=torch.float32)
            Rm_tensor = torch.tensor(self.Rm, requires_grad=True, dtype=torch.float32)
        if not trainMotor:
            self.km_tensor = torch.tensor(self.km, dtype=torch.float32)
            self.Rm_tensor = torch.tensor(self.Rm, dtype=torch.float32)
        self.trainMotor = trainMotor
        if torch.__version__ == "0.4.1":
            self.loss_fn = torch.nn.MSELoss(reduction='elementwise_mean')

        if type(init_std) == np.ndarray:
            init_std = torch.tensor([init_std[0] / self.tau, init_std[1] / self.tau], dtype=torch.float32, requires_grad=True)
        else:
            init_std = torch.tensor([0.001*self.fs, 0.001*self.fs], requires_grad=True, dtype=torch.float32)

        self.register_parameter('theta', theta)
        self.register_parameter('std', init_std)
        if trainMotor:
            self.register_parameter('km_tensor', km_tensor)
            self.register_parameter('Rm_tensor', Rm_tensor)

        self.forwardMode = forwardMode

        # needed to be compatible with PGSupportingPolicy
        self.is_disc_action=False

        # needed for the rollout
        self._normalized_input_obs = [False, False, False, False]
        self._normalized_input_a = [False]
        self._normalized_output_state_diff = [False, False]
        self._normalized_output = [False, False, False, False]

        # continuous action space between -force_mag and force_mag
        self._action_space = spaces.Box(np.array([-self.u_max]), np.array([self.u_max]))
        self._observation_space = spaces.Box(np.array(self.x_min), np.array(self.x_max))

        # if we want to use hacks, s.t. the model works with controller of Lukas
        self.hackOn = hackOn

        self.fs = fs
        self.tau = 1 / self.fs

        # if we want some custom initialization for the initial state
        self.customInit = False

        self.filter = VelocityFilter(2, dt=self.tau)

        # flag if the controller is outputing torques or voltage
        self.torque_controller = torque_controller

    @property
    def input_dim(self):
        # current state + action
        return self.x_dim+self.u_dim

    @property
    def output_dim(self):
        # next state + action
        return self.x_dim

    @property
    @overrides
    def normalized_output(self):
        return self._normalized_output

    @property
    @overrides
    def normalized_input(self):
        return self._normalized_input_obs + self._normalized_input_a

    @property
    def normalized_input_a(self):
        return self._normalized_input_a

    @property
    def normalized_input_obs(self):
        return self._normalized_input_obs

    @property
    def normalized_output_state_diff(self):
        return self._normalized_output_state_diff

    @property
    def pred_diff(self):
        return True

    @overrides
    @property
    def action_space(self):
        return self._action_space

    @overrides
    @property
    def observation_space(self):
        return self._observation_space

    def f(self, action_tensor, state):
        input = torch.cat([state, action_tensor]).unsqueeze(0)
        output_mean, _, output_std = self.forward(input)
        output = torch.normal(output_mean, output_std) # output_mean # torch.normal(output_mean, output_std)
        return torch.cat([state[2:4], output])

    def step(self, action):

        # # convert the torque into volt, since we assume to have voltage instead of torques
        # HACK: needed for controller of Lukas
        if self.torque_controller:
            action = action*self.Rm/self.km
        # clip the voltage (also done on the real system to prevent to extreme voltages)
        a = np.clip(action, -self.u_max, self.u_max)

        action_tensor = torch.from_numpy(a).float()

        # do a RK4 step
        k1 = self.f(action_tensor, self.current_obs)
        # clip the velocities also in part of the RK4 steps
        for idx in self.x_dot_state_idx:
            k1[idx] = torch.clamp(k1[idx], self.x_min[idx], self.x_max[idx])
        k2 = self.f(action_tensor, self.current_obs + 1 / (2 * self.fs) * k1)
        for idx in self.x_dot_state_idx:
            k2[idx] = torch.clamp(k2[idx], self.x_min[idx], self.x_max[idx])
        k3 = self.f(action_tensor, self.current_obs + 1 / (2 * self.fs) * k2)
        for idx in self.x_dot_state_idx:
            k3[idx] = torch.clamp(k3[idx], self.x_min[idx], self.x_max[idx])
        k4 = self.f(action_tensor, self.current_obs + 1 / self.fs * k3)
        for idx in self.x_dot_state_idx:
            k4[idx] = torch.clamp(k4[idx], self.x_min[idx], self.x_max[idx])
        qdd = (1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)).detach().numpy()

        # # update x states
        obs = self.current_obs.detach().numpy()
        obs = obs + 1 / self.fs * qdd

        # TODO: decide if we want to set velocity to zero when max angle is reached or not. What is more realistic?
        # set velocity to zero if pendulum is at it's max angle
        # if obs[0] < self.theta_min or obs[0] > self.theta_max:
        #     obs[2] = 0

        # clip the angle theta
        obs[0] = np.clip(obs[0], self.theta_min, self.theta_max)

        # clamp the velocities to max values s.t. we cannot get numerical errors?
        for idx in self.x_dot_state_idx:
            obs[idx] = np.clip(obs[idx], self.x_min[idx], self.x_max[idx])

        r, done = self._wrapped_env.reward(obs[:len(self.x_state_idx+self.x_dot_state_idx)]) # take only the original states
        self.current_obs = torch.from_numpy(obs).float()

        # HACK
        # remap alpha between -pi and pi with 0.0 above to be consistent the controller of Lukas, internally we still
        # have 0.0 as hanging down and pi as above
        if self.hackOn:
            alpha_remapped = obs[1] % (2 * np.pi) - np.pi
            newObs = np.array([obs[0], alpha_remapped, obs[2], obs[3]])
        else:
            newObs = obs

        # estimate velocities from the states, instead of using them directly from our model
        # angles = newObs[0:2]
        # vel = self.filter.estimate(angles)
        # newObs = np.r_[angles, vel]

        return newObs, r, done, {'obs_diff': qdd[2:]}

    def reset(self, **kwargs):
        if self.customInit:
            obs = np.random.uniform(low=[-4.29514609e-02, -1.18411663e-06, -8.88178420e-16, -2.27373675e-13], high=[1.74873814e-01, 3.06799413e-03, 1.77635684e-15, 4.54747351e-13])
        else:
            obs = self._wrapped_env.reset(**kwargs)
        self.current_obs = torch.from_numpy(obs).float()

        self.filter = VelocityFilter(2, self.tau)

        # HACK
        # remap alpha between -pi and pi with 0.0 above to be consistent the controller of Lukas, internally we still
        # have 0.0 as hanging down and pi as above
        if self.hackOn:
            alpha_remapped = obs[1] % (2 * np.pi) - np.pi
            obs = np.array([obs[0], alpha_remapped, obs[2], obs[3]])

        # # estimate velocities from the states, instead of using them directly from our model
        # angles = obs[0:2]
        # vel = self.filter.estimate(angles)
        # obs = np.r_[angles, vel]

        return obs

    def forward(self, x):
        state_tensor = x[:, :self.x_dim]
        action_tensor = x[:, self.x_dim:]
        mean_qdd = self.predict_forward_model_deterministic(state_tensor, action_tensor)

        std = self.std.expand_as(mean_qdd)
        log_std = torch.log(std)

        return mean_qdd, log_std, std

    def select_action(self, x, t):
        raise NotImplementedError("Not supposed to be called in our current architecture. Use step!")

    # calculates the features matrix for q, qd, qdd using for the inverse dynamic model
    # assumes that they are given as tensors, where first dimension is the batch
    def getBatchedFeatureMatrix(self, q, qd, qdd):

        # use none to prevent collapsing
        c2 = torch.cos(q[:, 1, None])
        s2 = torch.sin(q[:, 1, None])
        double_s2 = torch.sin(2 * q[:, 1, None])

        qd1 = qd[:, 0, None]
        qd2 = qd[:, 1, None]
        qdd1 = qdd[:, 0, None]
        qdd2 = qdd[:, 1, None]

        # feature matrix will be blockwise, first block is for torque 1
        y11 = qdd1
        y12 = s2 ** 2 * qdd1 + double_s2 * qd1 * qd2
        y13 = -c2 * qdd2 + s2 * qd2**2
        y14 = torch.zeros(q[:, 0, None].shape, dtype=torch.float32)
        y15 = torch.zeros(q[:, 0, None].shape, dtype=torch.float32)
        y16 = qd1
        y17 = torch.zeros(q[:, 0, None].shape, dtype=torch.float32)

        block1 = torch.cat([y11.unsqueeze(1), y12.unsqueeze(1), y13.unsqueeze(1), y14.unsqueeze(1),
                            y15.unsqueeze(1), y16.unsqueeze(1), y17.unsqueeze(1)], dim=2)

        # second block is for torque 2
        y21 = torch.zeros(q[:, 0, None].shape, dtype=torch.float32)
        y22 = - 1 / 2 * double_s2 * qd1 ** 2
        y23 = c2 * qdd1
        y24 = qdd2
        y25 = self.g*s2
        y26 = torch.zeros(q[:, 0, None].shape, dtype=torch.float32)
        y27 = qd2

        block2 = torch.cat([y21.unsqueeze(1), y22.unsqueeze(1), y23.unsqueeze(1), y24.unsqueeze(1),
                            y25.unsqueeze(1), y26.unsqueeze(1), y27.unsqueeze(1)], dim=2)

        batched_feature_mat = torch.cat([block1, block2], dim=1)

        return batched_feature_mat


    def predict_forward_model_deterministic(self, state_tensor, action_tensor):
        q  = state_tensor[:, 0:2]
        qd = state_tensor[:, 2:4]
        qd1 = qd[:, 0, None]
        qd2 = qd[:, 1, None]
        c2 = torch.cos(q[:, 1, None])
        s2 = torch.sin(q[:, 1, None])
        double_s2 = torch.sin(2 * q[:, 1, None])
        batched_theta = self.theta.unsqueeze(0).expand(state_tensor.size(0), len(self.theta)).unsqueeze(2)

        action_tensor = self.km_tensor * (action_tensor - self.km_tensor * qd1) / self.Rm_tensor

        ## TODO: initialize the matrix using a for loop could look somewhat nicer and should be done as soon we move to more complex problems
        ## calculate the feature matrix for the mass matrix
        phi_1_1_11 = torch.ones(q[:, 0, None].shape, dtype=torch.float32)
        phi_1_1_12 = s2**2
        phi_1_1_13 = torch.zeros(q[:, 0, None].shape, dtype=torch.float32)
        phi_1_1_14 = torch.zeros(q[:, 0, None].shape, dtype=torch.float32)
        phi_1_1_15 = torch.zeros(q[:, 0, None].shape, dtype=torch.float32)
        phi_1_1_16 = torch.zeros(q[:, 0, None].shape, dtype=torch.float32)
        phi_1_1_17 = torch.zeros(q[:, 0, None].shape, dtype=torch.float32)
        phi_1_1_21 = torch.zeros(q[:, 0, None].shape, dtype=torch.float32)
        phi_1_1_22 = torch.zeros(q[:, 0, None].shape, dtype=torch.float32)
        phi_1_1_23 = c2
        phi_1_1_24 = torch.zeros(q[:, 0, None].shape, dtype=torch.float32)
        phi_1_1_25 = torch.zeros(q[:, 0, None].shape, dtype=torch.float32)
        phi_1_1_26 = torch.zeros(q[:, 0, None].shape, dtype=torch.float32)
        phi_1_1_27 = torch.zeros(q[:, 0, None].shape, dtype=torch.float32)
        phi_1_1_1 = torch.cat([phi_1_1_11.unsqueeze(1), phi_1_1_12.unsqueeze(1), phi_1_1_13.unsqueeze(1), phi_1_1_14.unsqueeze(1), phi_1_1_15.unsqueeze(1),
                 phi_1_1_16.unsqueeze(1), phi_1_1_17.unsqueeze(1)], dim=2)
        phi_1_1_2 = torch.cat([phi_1_1_21.unsqueeze(1), phi_1_1_22.unsqueeze(1), phi_1_1_23.unsqueeze(1), phi_1_1_24.unsqueeze(1), phi_1_1_25.unsqueeze(1),
                 phi_1_1_26.unsqueeze(1), phi_1_1_27.unsqueeze(1)], dim=2)
        phi_1_1 = torch.cat([phi_1_1_1, phi_1_1_2], dim=1)
        phi_1_c1 = phi_1_1.bmm(batched_theta)

        phi_1_2_11 = torch.zeros(q[:, 0, None].shape, dtype=torch.float32)
        phi_1_2_12 = torch.zeros(q[:, 0, None].shape, dtype=torch.float32)
        phi_1_2_13 = -c2
        phi_1_2_14 = torch.zeros(q[:, 0, None].shape, dtype=torch.float32)
        phi_1_2_15 = torch.zeros(q[:, 0, None].shape, dtype=torch.float32)
        phi_1_2_16 = torch.zeros(q[:, 0, None].shape, dtype=torch.float32)
        phi_1_2_17 = torch.zeros(q[:, 0, None].shape, dtype=torch.float32)
        phi_1_2_21 = torch.zeros(q[:, 0, None].shape, dtype=torch.float32)
        phi_1_2_22 = torch.zeros(q[:, 0, None].shape, dtype=torch.float32)
        phi_1_2_23 = torch.zeros(q[:, 0, None].shape, dtype=torch.float32)
        phi_1_2_24 = torch.ones(q[:, 0, None].shape, dtype=torch.float32)
        phi_1_2_25 = torch.zeros(q[:, 0, None].shape, dtype=torch.float32)
        phi_1_2_26 = torch.zeros(q[:, 0, None].shape, dtype=torch.float32)
        phi_1_2_27 = torch.zeros(q[:, 0, None].shape, dtype=torch.float32)
        phi_1_2_1 = torch.cat([phi_1_2_11.unsqueeze(1), phi_1_2_12.unsqueeze(1), phi_1_2_13.unsqueeze(1), phi_1_2_14.unsqueeze(1), phi_1_2_15.unsqueeze(1),
                 phi_1_2_16.unsqueeze(1), phi_1_2_17.unsqueeze(1)], dim=2)
        phi_1_2_2 = torch.cat([phi_1_2_21.unsqueeze(1), phi_1_2_22.unsqueeze(1), phi_1_2_23.unsqueeze(1), phi_1_2_24.unsqueeze(1), phi_1_2_25.unsqueeze(1),
                 phi_1_2_26.unsqueeze(1), phi_1_2_27.unsqueeze(1)], dim=2)
        phi_1_2 = torch.cat([phi_1_2_1, phi_1_2_2], dim=1)
        phi_1_c2 = phi_1_2.bmm(batched_theta)

        phi_1 = torch.cat([phi_1_c1, phi_1_c2], dim=2)

        ## calcualte the feature matrix for the forces vector part
        # define matrix row wise
        phi_2_11 = torch.zeros(q[:, 0, None].shape, dtype=torch.float32)
        phi_2_12 = double_s2 * qd1 * qd2
        phi_2_13 = s2 * qd2**2
        phi_2_14 = torch.zeros(q[:, 0, None].shape, dtype=torch.float32)
        phi_2_15 = torch.zeros(q[:, 0, None].shape, dtype=torch.float32)
        phi_2_16 = qd1
        phi_2_17 = torch.zeros(q[:, 0, None].shape, dtype=torch.float32)
        phi_2_21 = torch.zeros(q[:, 0, None].shape, dtype=torch.float32)
        phi_2_22 = - 1/2 * double_s2 * qd1**2
        phi_2_23 = torch.zeros(q[:, 0, None].shape, dtype=torch.float32)
        phi_2_24 = torch.zeros(q[:, 0, None].shape, dtype=torch.float32)
        phi_2_25 = self.g*s2
        phi_2_26 = torch.zeros(q[:, 0, None].shape, dtype=torch.float32)
        phi_2_27 = qd2
        phi_2_1 = torch.cat([phi_2_11.unsqueeze(1), phi_2_12.unsqueeze(1), phi_2_13.unsqueeze(1), phi_2_14.unsqueeze(1), phi_2_15.unsqueeze(1),
                 phi_2_16.unsqueeze(1), phi_2_17.unsqueeze(1)], dim=2)
        phi_2_2 = torch.cat([phi_2_21.unsqueeze(1), phi_2_22.unsqueeze(1), phi_2_23.unsqueeze(1), phi_2_24.unsqueeze(1), phi_2_25.unsqueeze(1),
                 phi_2_26.unsqueeze(1), phi_2_27.unsqueeze(1)], dim=2)
        phi_2 = torch.cat([phi_2_1, phi_2_2], dim=1)

        forces = phi_2.bmm(batched_theta)

        stackedActions = torch.cat([action_tensor, torch.zeros(action_tensor.shape, dtype=torch.float32)], dim=1)
        b = stackedActions.unsqueeze(2)-forces # need to unsqueeze such that we have # batch x dim1 x 1
        if torch.__version__ == '0.4.0':
            # no batchwise calculation of gesv
            if b.shape[0] > 1:
                raise NotImplementedError("forward does not work with batches in torch 0.4.0")
            X, LU = torch.gesv(b[0, :, :], phi_1[0, :, :])
        else:
            X, LU = torch.gesv(b, phi_1)
        qdd = X.squeeze()

        if torch.isnan(qdd).sum() > 0:
            print("qdd is NAN this should not happen!", phi_1, b)
            print("state", state_tensor)
            print("action", action_tensor)
            print("c2", c2)
            print("s2", s2)
            print("theta", batched_theta)

        return qdd

    def log_prob_forward_model(self, state_tensor, action_tensor, next_state_tensor):
        # calculate qdd from current state and next state
        means = self.predict_forward_model_deterministic(state_tensor, action_tensor)
        multivariate_normal = MultivariateNormal(means, covariance_matrix=torch.diag(self.std.pow(2)))
        log_prob = (multivariate_normal.log_prob(next_state_tensor))
        return log_prob

    def predict_inverse_model(self, state_tensor, next_state_tensor):
        # calculate qdd from current state and next state
        qdd = (next_state_tensor[:, 2:4] - state_tensor[:, 2:4]) * self.fs  # need to devide by tau to scale to the right timestep
        batched_feature_mat = self.getBatchedFeatureMatrix(state_tensor[:, 0:2], state_tensor[:, 2:4], qdd) # batch x dim1 x dim2
        batched_theta = self.theta.unsqueeze(0).expand(state_tensor.size(0), len(self.theta)).unsqueeze(2)  # batch x dim2 x 1
        output = batched_feature_mat.bmm(batched_theta).squeeze() # batch x dim1 x 1
        return output[:, 0, None] * self.Rm_tensor / self.km_tensor + self.km_tensor * state_tensor[:, 2, None]

    def get_log_prob(self, x, next_state_tensor):
        # unpack x to state and action tensor
        state_tensor = x[:, 0:4]
        action_tensor = x[:, 4, None] # use none to prevent collapsing
        if self.forwardMode:
            return self.log_prob_forward_model(state_tensor, action_tensor, next_state_tensor)

    def get_log_prob_n_step(self, x, output_torch_var, n_step, num_sample, only_last_step_log_prob):
        # repeat the input for num_sample times to approximate the next state distribution via samples (note this is not
        # very efficient)
        x = x.repeat(num_sample, 1, 1)
        output_torch_var = output_torch_var.repeat(num_sample, 1, 1)
        state_tensor = x[:, 0, 0:4]
        log_probs = 0
        for t in range(n_step):
            action_tensor = x[:, t, 4]
            means = self.predict_forward_model_deterministic(state_tensor, action_tensor)
            multivariate_normal = MultivariateNormal(means, covariance_matrix=batch_diagonal(self.std.pow(2)))
            if only_last_step_log_prob and t == n_step - 1:
                log_probs = multivariate_normal.log_prob(output_torch_var[:,t,:]).unsqueeze(1)
            else:
                log_probs += multivariate_normal.log_prob(output_torch_var[:,t,:]).unsqueeze(1)
            # do now a step with the means and std to compute the new state
            output = multivariate_normal.sample() # sampled output for euler states
            # update x states
            input[:, self.x_state_idx] += input[:, self.x_dot_state_idx] * self.tau
            # update x_dot states
            input[:, self.x_dot_state_idx] += output * self.tau
            # map theta between -pi and pi
            input[:, 2] = torch.fmod(input[:,2] + np.pi, np.pi * 2) - np.pi
            # clamp the velocities to max values s.t. we cannot get numerical errors?
            input[:, 1] = torch.clamp(input[:,1], -self.high[1], self.high[1])
            input[:, 3] = torch.clamp(input[:,3], -self.high[3], self.high[3])
        return log_probs

    def get_kl_forward(self, state_tensor, action_tensor):
        means = self.predict_forward_model_deterministic(state_tensor, action_tensor)
        multivariate_normal = MultivariateNormal(means, covariance_matrix=torch.diag(self.std.pow(2)))
        fixed_multivariate_normal = MultivariateNormal(torch.tensor(means), covariance_matrix=torch.tensor(torch.diag(self.std.pow(2))))
        # TODO: check order of kl-div
        kl_div = torch.distributions.kl_divergence(multivariate_normal, fixed_multivariate_normal)
        return kl_div #kl.sum(1, keepdim=True)

    def get_kl(self, x):
        #unpack x to state and action tensor
        state_tensor = x[:, 0:4]
        action_tensor = x[:, 4, None] # use none to prevent collapsing
        if self.forwardMode:
            return self.get_kl_forward(state_tensor, action_tensor)

    def set_param_values(self, params):
        # assume that params is a tensor of shape 5, theta has dim 7 and we have 2 dimensional noise
        if not params.shape[0] == 9 and not self.trainMotor:
            raise ValueError("something went wrong")
        elif not params.shape[0] == 11 and self.trainMotor:
            raise ValueError("something went wrong")

        # prevent negative parameters by using absolute value
        torch_utils.set_flat_params_to(self, params)

    # def set_param_values(self, params):
    #     # assume that params is a tensor of shape 5, theta has dim 7 and we have 2 dimensional noise
    #     if not params.shape[0] == 7:
    #         raise ValueError("something went wrong")
    #
    #     # prevent negative parameters by using absolute value
    #     torch_utils.set_flat_params_to(self, torch.cat([params, self.std]))

    # def set_param_values(self, params):
    #     # assume we can set lr, Jr, Dr and lp, Jp, Dp only --> theta has dim 6
    #     if not params.shape[0] == 5:
    #         raise ValueError("something went wrong")
    #
    #     self.lr = params[0]
    #     self.Jr = params[1]
    #     self.Dr = params[2]
    #     self.lp = params[3]
    #     self.Jp = params[4]
    #
    #     torch_utils.set_flat_params_to(self, torch.cat([torch.tensor([self.Mp * self.Lr ** 2 + self.Jr, self.Mp * (self.lp * self.Lp) ** 2,
    #                           self.Mp * self.lp * self.Lp * self.Lr, self.Jp + self.Mp * (self.lp * self.Lp) ** 2,
    #                           self.Mp * self.lp * self.Lp, self.Dr, self.Dp]), self.std]))

    ##### function needed to mimic behavior of nn.module for saving and loading parameters

    def register_parameter(self, name, param):
        r"""Adds a parameter to the module.

        The parameter can be accessed as an attribute using given name.

        Args:
            name (string): name of the parameter. The parameter can be accessed
                from this module using the given name
            parameter (Parameter): parameter to be added to the module.
        """
        if '_parameters' not in self.__dict__:
            raise AttributeError(
                "cannot assign parameter before Module.__init__() call")

        elif not isinstance(name, torch._six.string_classes):
            raise TypeError("parameter name should be a string. "
                            "Got {}".format(torch.typename(name)))
        elif '.' in name:
            raise KeyError("parameter name can't contain \".\"")
        elif name == '':
            raise KeyError("parameter name can't be empty string \"\"")
        elif hasattr(self, name) and name not in self._parameters:
            raise KeyError("attribute '{}' already exists".format(name))

        if param is None:
            self._parameters[name] = None
        else:
            self._parameters[name] = param


    def __getattr__(self, name):
        if '_parameters' in self.__dict__:
            _parameters = self.__dict__['_parameters']
            if name in _parameters:
                return _parameters[name]
        # Default behaviour
        return object.__getattribute__(self, name)


    def __setattr__(self, name, value):
        params = self.__dict__.get('_parameters')
        if params is not None and name in params:
            if value is not None:
                raise TypeError("cannot assign '{}' as parameter '{}' "
                                "(torch.nn.Parameter or None expected)"
                                .format(torch.typename(value), name))
            self.register_parameter(name, value)
        else:
            # Default behaviour
            return object.__setattr__(self, name, value)

    def __delattr__(self, name):
        if name in self._parameters:
            del self._parameters[name]
        else:
            object.__delattr__(self, name)

    def parameters(self):
        r"""Returns an iterator over module parameters.

        This is typically passed to an optimizer.

        Yields:
            Parameter: module parameter

        Example::

            >>> for param in model.parameters():
            >>>     print(type(param.data), param.size())
            <class 'torch.FloatTensor'> (20L,)
            <class 'torch.FloatTensor'> (20L, 1L, 5L, 5L)

        """
        for name, param in self.named_parameters():
            yield param

    def named_parameters(self, memo=None, prefix=''):
        r"""Returns an iterator over module parameters, yielding both the
        name of the parameter as well as the parameter itself

        Yields:
            (string, Parameter): Tuple containing the name and parameter

        Example::

            >>> for name, param in self.named_parameters():
            >>>    if name in ['bias']:
            >>>        print(param.size())

        """
        if memo is None:
            memo = set()
        for name, p in self._parameters.items():
            if p is not None and p not in memo:
                memo.add(p)
                yield prefix + ('.' if prefix else '') + name, p

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()
        destination._metadata[prefix[:-1]] = dict(version=self._version)
        for name, param in self._parameters.items():
            if param is not None:
                destination[prefix + name] = param if keep_vars else param.data
        return destination


    def _load_from_state_dict(self, state_dict, prefix, metadata, strict, missing_keys, unexpected_keys, error_msgs):
        r"""Copies parameters and buffers from :attr:`state_dict` into only
        this module, but not its descendants. This is called on every submodule
        in :meth:`~torch.nn.Module.load_state_dict`. Metadata saved for this
        module in input :attr:`state_dict` is provided as :attr`metadata`.
        For state dicts without meta data, :attr`metadata` is empty.
        Subclasses can achieve class-specific backward compatible loading using
        the version number at `metadata.get("version", None)`.

        .. note::
            :attr:`state_dict` is not the same object as the input
            :attr:`state_dict` to :meth:`~torch.nn.Module.load_state_dict`. So
            it can be modified.

        Arguments:
            state_dict (dict): a dict containing parameters and
                persistent buffers.
            prefix (str): the prefix for parameters and buffers used in this
                module
            metadata (dict): a dict containing the metadata for this moodule.
                See
            strict (bool): whether to strictly enforce that the keys in
                :attr:`state_dict` with :attr:`prefix` match the names of
                parameters and buffers in this module
            missing_keys (list of str): if ``strict=False``, add missing keys to
                this list
            unexpected_keys (list of str): if ``strict=False``, add unexpected
                keys to this list
            error_msgs (list of str): error messages should be added to this
                list, and will be reported together in
                :meth:`~torch.nn.Module.load_state_dict`
        """
        local_name_params = itertools.chain(self._parameters.items())
        local_state = {k: v.data for k, v in local_name_params if v is not None}

        for name, param in local_state.items():
            key = prefix + name
            if key in state_dict:
                input_param = state_dict[key]

                if input_param.shape != param.shape:
                    # local shape should match the one in checkpoint
                    error_msgs.append('size mismatch for {}: copying a param of {} from checkpoint, '
                                      'where the shape is {} in current model.'
                                      .format(key, param.shape, input_param.shape))
                    continue

                # backwards compatibility for serialized parameters
                input_param = input_param.data
                try:
                    param.copy_(input_param)
                except Exception:
                    error_msgs.append('While copying the parameter named "{}", '
                                      'whose dimensions in the model are {} and '
                                      'whose dimensions in the checkpoint are {}.'
                                      .format(key, param.size(), input_param.size()))
            elif strict:
                missing_keys.append(key)

        if strict:
            for key, input_param in state_dict.items():
                if key.startswith(prefix):
                    input_name = key[len(prefix):]
                    input_name = input_name.split('.', 1)[0]  # get the name of param/buffer/child
                    if input_name not in local_state:
                        unexpected_keys.append(key)

    def load_state_dict(self, state_dict, strict=True):
        r"""Copies parameters and buffers from :attr:`state_dict` into
        this module and its descendants. If :attr:`strict` is ``True``, then
        the keys of :attr:`state_dict` must exactly match the keys returned
        by this module's :meth:`~torch.nn.Module.state_dict` function.

        Arguments:
            state_dict (dict): a dict containing parameters and
                persistent buffers.
            strict (bool, optional): whether to strictly enforce that the keys
                in :attr:`state_dict` match the keys returned by this module's
                :meth:`~torch.nn.Module.state_dict` function. Default: ``True``
        """
        missing_keys = []
        unexpected_keys = []
        error_msgs = []

        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

        load(self)

        if strict:
            error_msg = ''
            if len(unexpected_keys) > 0:
                error_msgs.insert(
                    0, 'Unexpected key(s) in state_dict: {}. '.format(
                        ', '.join('"{}"'.format(k) for k in unexpected_keys)))
            if len(missing_keys) > 0:
                error_msgs.insert(
                    0, 'Missing key(s) in state_dict: {}. '.format(
                        ', '.join('"{}"'.format(k) for k in missing_keys)))

        if len(error_msgs) > 0:
            raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                               self.__class__.__name__, "\n\t".join(error_msgs)))
