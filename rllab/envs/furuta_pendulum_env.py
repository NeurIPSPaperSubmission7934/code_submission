# Copyright (c) 2018 Copyright holder of the paper Generative Adversarial Model Learning
# submitted to NeurIPS 2019 for review
# All rights reserved.


import numpy as np
import importlib
from rllab.envs.base import Env
from rllab import spaces
from rllab.misc.overrides import overrides


class FurutaPendulumEnv(Env):
    # Sampling frequency
    fs = 100.0

    # Gravity
    g = 9.81

    # Motor
    Rm = 8.4  # resistance
    kt = 0.042  # current-torque (N-m/A)
    km = 0.042  # back-emf constant (V-s/rad)

    # Rotary arm
    Mr = 0.095  # mass (kg)
    Lr = 0.112  # length (m)
    Jr = Mr * Lr ** 2 / 12  # moment of inertia about COM (kg-m^2)
    Dr = 0.0003  # equivalent viscous damping coefficient (N-m-s/rad)

    # Pendulum link
    Mp = 0.024  # mass (kg)
    Lp = 0.129  # length (m)
    Jp = Mp * Lp ** 2 / 12  # moment of inertia about COM (kg-m^2)
    Dp = 0.00005  # equivalent viscous damping coefficient (N-m-s/rad)

    # Joint angles
    alpha = 0
    alpha_d = 0
    alpha_dd = 0
    theta = 0
    theta_d = 0
    theta_dd = 0

    theta_min = -2.3
    theta_max = 2.3
    u_max = 5.0

    u_dim = 1
    x_dim = 4
    x_labels = ('th', 'alpha', 'th_dot', 'alpha_dot')
    u_labels = ('Vm',)
    x_min = (-2.3, -np.inf, -np.inf, -np.inf)
    x_max = (2.3, np.inf, np.inf, np.inf)

    range = 0.2  # shows a 0.4m x 0.4m excerpt of the scene
    arm_radius = 0.003
    arm_length = 0.085
    pole_radius = 0.0045
    pole_length = 0.129

    shown_points = []

    def __init__(self, show_gui=True, center_camera=False, use_theta=True, use_rk4=True, dynamics_noise=False, noise_sigma=0.01):

        high = np.array([
            2.3,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max])
        self._observation_space = spaces.Box(-high, high) #[theta, alpha, theta_dot, alpha_dot]
        self._action_space = spaces.Box(np.array([-5.0]), np.array([5.0]))
        self.show_gui = show_gui
        self.use_theta = use_theta
        self.use_rk4 = use_rk4
        self.dynamic_noise = dynamics_noise
        self.noise_sigma = noise_sigma*self.fs
        self.noise_mu = 0

        if self.use_theta:
            self.theta_lin_param = np.array([self.Mp*self.Lr**2+self.Jr,
                                   1/4*self.Mp*self.Lp**2,
                                   1/2*self.Mp*self.Lp*self.Lr,
                                   self.Jp+1/4*self.Mp*self.Lp**2,
                                   1/2*self.Mp*self.Lp,
                                   self.Dr,
                                   self.Dp])
        if show_gui:
            self.set_gui(center_camera)

    # Define the GUI given a camera position [centered / normal]
    def set_gui(self, center_camera):
        # import vpython globally
        # if you don't have vpython installed just set show_gui in the constructor to false
        global vp
        # Vpython scene: http://www.glowscript.org/docs/VPythonDocs/canvas.html
        vp = importlib.import_module("vpython")
        vp.scene.width = 800
        vp.scene.height = 800
        vp.scene.background = vp.color.gray(0.95)  # equals browser background -> higher = brighter
        vp.scene.lights = []
        vp.distant_light(direction=vp.vector(0.2,  0.2,  0.5), color=vp.color.white)
        vp.scene.up = vp.vector(0, 0, 1)  # z-axis is showing up
        # front view for projections
        if center_camera:
            vp.scene.range = self.range
            vp.scene.center = vp.vector(0, 0, 0)
            vp.scene.forward = vp.vector(-1, 0, 0)
        # ...or custom view for better observation
        else:
            vp.scene.range = self.range
            vp.scene.center = vp.vector(0.04, 0, 0)
            vp.scene.forward = vp.vector(-2, 1.2, -1)

        vp.box(pos=vp.vector(0, 0, -0.07), length=0.09, width=0.1, height=0.09, color=vp.color.gray(0.5))

        vp.cylinder(axis=vp.vector(0, 0, -1), radius=0.005, length=0.03, color=vp.color.gray(0.5))

        # robot arm
        self.arm = vp.cylinder()
        self.arm.radius = self.arm_radius
        self.arm.length = self.arm_length
        self.arm.color = vp.color.blue
        # robot pole
        self.pole = vp.cylinder()
        self.pole.radius = self.pole_radius
        self.pole.length = self.pole_length
        self.pole.color = vp.color.red

        self.curve = vp.curve(color=vp.color.white, radius=0.0005, retain=2000)

        self.render()

    # Reset global joint angles and re-render the scene
    def reset(self):
        # have small initial state distribution to be more realistic compared to the real system
        x = np.random.uniform(low=-0.05, high=0.05, size=(4,))
        # Joint angles
        self.alpha = x[2]
        self.alpha_d = x[3]
        self.alpha_dd = 0
        self.theta = x[0]
        self.theta_d = x[1]
        self.theta_dd = 0
        if self.show_gui:
            self.curve.clear()
            self.clear_rendered_points()
        self.render()
        return x

    def rk4(self, u, q):
        ## equations taken from Quanser-Servo 2 Workbook

        c1 = self.Mp * self.Lr ** 2 + 1 / 4 * self.Mp * self.Lp ** 2 - 1 / 4 * self.Mp * self.Lp ** 2 * np.cos(
            q[1]) ** 2 + self.Jr
        c2 = 1 / 2 * self.Mp * self.Lp * self.Lr * np.cos(q[1])
        c3 = - self.Dr * q[2] - 0.5 * self.Mp * self.Lp ** 2 * np.sin(q[1]) * np.cos(
            q[1]) * q[2] * q[3] - 0.5 * self.Mp * self.Lp * self.Lr * np.sin(
            q[1]) * q[3] ** 2

        c4 = 0.5 * self.Mp * self.Lp * self.Lr * np.cos(q[1])
        c5 = self.Jp + 1 / 4 * self.Mp * self.Lp ** 2
        c6 = - self.Dp * q[3] + 1 / 4 * self.Mp * self.Lp ** 2 * np.cos(q[1]) * np.sin(
            q[1]) * q[2] ** 2 - 0.5 * self.Mp * self.Lp * self.g * np.sin(q[1])

        a = np.array([[c1, -c2], [c4, c5]])
        b = np.array([u+c3, c6])

        [th_dd, al_dd] = np.linalg.solve(a, b)

        #print(np.dot(a, np.array([th_dd, al_dd]))-np.array([c3, c6]),u)

        # batchmatrix = self.getBatchedFeatureMatrix(torch.from_numpy(np.array([q[0:2]])), torch.from_numpy(np.array([q[2:4]])), torch.from_numpy(np.array([[th_dd, al_dd]]))).squeeze()
        #
        # self.theta = np.array([self.Mp*self.Lr**2+self.Jr,
        #                        1/4*self.Mp*self.Lp**2,
        #                        1/2*self.Mp*self.Lp*self.Lr,
        #                        self.Jp+1/4*self.Mp*self.Lp**2,
        #                        1/2*self.Mp*self.Lp,
        #                        self.Dr,
        #                        self.Dp])
        # print(np.dot(batchmatrix.numpy(), self.theta)[0], u)

        # qdd = self.predict_forward_model(torch.from_numpy(np.array([q])),torch.from_numpy(np.array([[u]]))).squeeze()
        # print(qdd, [th_dd, al_dd])
        return np.array([q[2], q[3], th_dd, al_dd])

    def calcForwardDynamicsTheta(self, u, q): # note u is already in torque form and not as action (voltage)
        c2 = np.cos(q[1])
        s2 = np.sin(q[1])
        double_s2 = np.sin(2 * q[1])
        qd1 = q[2]
        qd2 = q[3]

        # calculate mass matrix part
        phi_1_1 = np.array([[1,s2**2, 0, 0, 0, 0,0 ],[0,0,c2,0,0,0,0]])
        phi_1_c1 = np.dot(phi_1_1, self.theta_lin_param)
        phi_1_2 = np.array([[0,0,-c2, 0, 0, 0,0 ],[0,0,0,1,0,0,0]])
        phi_1_c2 = np.dot(phi_1_2, self.theta_lin_param)
        phi_1 = np.column_stack([phi_1_c1, phi_1_c2])
        # calculate forces vector part
        phi_2 = np.array([[0, double_s2 * qd1 * qd2, s2 * qd2**2, 0, 0, qd1, 0], [0, - 1/2 * double_s2 * qd1**2, 0, 0, self.g*s2, 0, qd2]])
        forces = np.dot(phi_2, self.theta_lin_param)
        # now calculate qdd
        action = np.array([u, 0])
        # assert not np.isnan(u), "nan action"
        # assert not np.any(np.isnan(phi_1)), "nan in phi1"
        # assert not np.any(np.isnan(forces)), "nan in forces"
        [th_dd, al_dd], residuals, rank, sv = np.linalg.lstsq(phi_1, action - forces, rcond=1)
        if np.any(np.isnan(np.array([th_dd, al_dd]))): # check for nans
            print("nan in linalg.lstsq")
            print([th_dd, al_dd])
            print(phi_1)
            print(action)
            print(forces)
            print(phi_1_2)
            print(phi_2)
        return np.array([qd1, qd2, th_dd, al_dd])


    # Execute one step for a given action
    def step(self, action, dt_multiple=1):
        # TODO: FIXME for now hack a max velocity for theta and alpha into the simulation, s.t. we won't get nans
        self.theta_d = np.clip(self.theta_d, -1e4, 1e4)
        self.alpha_d = np.clip(self.alpha_d, -1e4, 1e4)

        # convert the torque into volt, since we assume to have voltage instead of torques
        # NOTE: this is needed since the Controller of Lukas gives only torques as output
        action = action*self.Rm/self.km

        # compute the applied torque from the control voltage (action)
        a = np.clip(action[0], -self.u_max, self.u_max)
        u = self.km * (a - self.km * self.theta_d) / self.Rm

        # clip the angle theta
        self.theta = np.clip(self.theta, self.theta_min, self.theta_max)
        # set torque to zero if pendulum is at it's max angle
        if self.theta == self.theta_min or self.theta == self.theta_max:
            self.theta_d = 0

        u_i = np.array([self.theta, self.alpha, self.theta_d, self.alpha_d])
        if self.use_theta:
            f = self.calcForwardDynamicsTheta
        else:
            f = self.rk4

        if self.use_rk4:
            k1 = f(u, u_i)
            k2 = f(u, u_i + 1 / (2 * self.fs) * k1)
            k3 = f(u, u_i + 1 / (2 * self.fs) * k2)
            k4 = f(u, u_i + 1 / self.fs * k3)
            qdd = (1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4))
        else:
            k1 = f(u, u_i)
            qdd = k1

        # check if qdd is nan
        if np.any(np.isnan(qdd)):  # check for nans
            print(qdd), k1, k2, k3, k4
        if self.dynamic_noise:
            # additive gaussian noise
            qdd += np.random.normal(loc=self.noise_mu, scale=self.noise_sigma, size=qdd.shape)

        [self.theta, self.alpha, self.theta_d, self.alpha_d] = u_i + 1 / self.fs * qdd

        # NOTE: to map alpha between -pi and pi with 0.0 at the top and to be consistent with the controller lukas uses
        alpha_remapped = self.alpha % (2 * np.pi) - np.pi

        if self.show_gui:
            self.render()

        return np.array([self.theta,
                        alpha_remapped,
                        self.theta_d,
                        self.alpha_d]), np.cos(alpha_remapped), False, {"action": u, "qdd":qdd[[1,3]]}

    def reward(self, obs):
        alpha = obs[1]
        return -np.cos(alpha), False

    # Render global state
    def render(self, rate=500):
        if not self.show_gui:
            return

        # Position: End of the arm
        x_pos_arm = self.Lr * np.cos(self.theta)
        y_pos_arm = self.Lr * np.sin(self.theta)
        z_pos_arm = 0

        pos_axis, reachable = self.forw_kin([self.theta, self.alpha])
        if not reachable:
            return

        # Direction: End of the arm -to- End of the pole (normalization not needed)
        x_axis_pole = pos_axis[0] - x_pos_arm
        y_axis_pole = pos_axis[1] - y_pos_arm
        z_axis_pole = pos_axis[2] - z_pos_arm

        # render the computed positions
        self.arm.axis = vp.vector(x_pos_arm, y_pos_arm, z_pos_arm)
        self.pole.pos = vp.vector(x_pos_arm, y_pos_arm, z_pos_arm)
        self.pole.axis = vp.vector(x_axis_pole, y_axis_pole, z_axis_pole)

        self.curve.append(self.pole.pos + self.pole.axis)

        vp.rate(rate)


    # Forward Kinematics
    def forw_kin(self, q):
        th, al = q
        reachable = False
        # check if theta is in reachable space
        if self.theta_min <= th <= self.theta_max:
            x = - self.Lp * np.sin(al) * np.sin(th) + self.Lr * np.cos(th)
            y = self.Lp * np.sin(al) * np.cos(th) + self.Lr * np.sin(th)
            z = - self.Lp * np.cos(al)
            reachable = True
        # if theta not reachable, return some default values
        else:
            x, y, z = [0, 0, 0]

        return [x, y, z], reachable

    def clear_rendered_points(self):
        if not self.show_gui:
            return
        for point in self.shown_points:
            point.visible = False
            del point

    @overrides
    @property
    def action_space(self):
        return self._action_space

    @overrides
    @property
    def observation_space(self):
        return self._observation_space

    def close(self):
        pass

    def get_param_values(self):
        return [self.theta_lin_param, self.km, self.Rm]

    def set_param_values(self, params):
        # do small check
        if not type(params[0]) is np.ndarray:
            return ValueError("Expected np.ndarray")
        if not params[0].shape == (7,):
            return ValueError("Expected 7-dim array")
        # unpack params and env params to them
        self.theta_lin_param = params[0]
        self.km = params[1]
        self.Rm = params[2]


    # # calculates the features matrix for q, qd, qdd using for the inverse dynamic model
    # # assumes that they are given as tensors, where first dimension is the batch
    # def getBatchedFeatureMatrix(self, q, qd, qdd):
    #
    #     # use none to prevent collapsing
    #     c2 = torch.cos(q[:, 1, None])
    #     s2 = torch.sin(q[:, 1, None])
    #     double_s2 = torch.sin(2 * q[:, 1, None])
    #
    #     qd1 = qd[:, 0, None]
    #     qd2 = qd[:, 1, None]
    #     qdd1 = qdd[:, 0, None]
    #     qdd2 = qdd[:, 1, None]
    #
    #     # feature matrix will be blockwise, first block is for torque 1
    #     y11 = qdd1
    #     y12 = s2 ** 2 * qdd1 + double_s2 * qd1 * qd2
    #     y13 = -c2 * qdd2 + s2 * qd2**2
    #     y14 = torch.zeros(q[:, 0, None].shape, dtype=torch.float64)
    #     y15 = torch.zeros(q[:, 0, None].shape, dtype=torch.float64)
    #     y16 = qd1
    #     y17 = torch.zeros(q[:, 0, None].shape, dtype=torch.float64)
    #
    #     block1 = torch.cat([y11.unsqueeze(1), y12.unsqueeze(1), y13.unsqueeze(1), y14.unsqueeze(1),
    #                         y15.unsqueeze(1), y16.unsqueeze(1), y17.unsqueeze(1)], dim=2)
    #
    #     # second block is for torque 2
    #     y21 = torch.zeros(q[:, 0, None].shape, dtype=torch.float64)
    #     y22 = - 1 / 2 * double_s2 * qd1 ** 2
    #     y23 = c2 * qdd1
    #     y24 = qdd2
    #     y25 = s2
    #     y26 = torch.zeros(q[:, 0, None].shape, dtype=torch.float64)
    #     y27 = qd2
    #
    #     block2 = torch.cat([y21.unsqueeze(1), y22.unsqueeze(1), y23.unsqueeze(1), y24.unsqueeze(1),
    #                         y25.unsqueeze(1), y26.unsqueeze(1), y27.unsqueeze(1)], dim=2)
    #
    #     batched_feature_mat = torch.cat([block1, block2], dim=1)
    #
    #     return batched_feature_mat


