# Copyright (c) 2018 Copyright holder of the paper Generative Adversarial Model Learning
# submitted to NeurIPS 2019 for review
# All rights reserved.

# The following class is modified from gym
# (https://github.com/openai/gym)
# licensed under the MIT license


"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https:/e/perma.cc/C9ZM-652R
"""

import math
from gym import logger
from gym.utils import seeding
import numpy as np
from rllab.envs.base import Env
from rllab.misc.overrides import overrides
from rllab import spaces
import torch

class CartPoleSwingUpEnv(Env):
    """
    Description:
        A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The pendulum starts upright, and the goal is to prevent it from falling over by increasing and reducing the cart's velocity.
    Source:
        This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson
    Observation:
        Type: Box(4)
        Num	Observation                 Min         Max
        0	Cart Position             -4.8            4.8
        1	Cart Velocity             -Inf            Inf
        2	Pole Angle                 -pi            pi
        3	Pole Velocity At Tip      -Inf            Inf

    Actions:
        Type: Box(1)
        Num	Action                        Min        Max
        0	Force applied to the cart     -10        10

        Note: The amount the velocity is reduced or increased is not fixed as it depends on the angle the pole is pointing. This is because the center of gravity of the pole increases the amount of energy needed to move the cart underneath it
    Reward:
        Reward is the cosinus of the current pole angle and -100 if the cart leaves the track
    Starting State:
        All observations are assigned a uniform random value between +-0.05
    Episode Termination:
        Cart Position is more than +-2.4 (center of the cart reaches the edge of the display)
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self, dynamics_noise='no_noise', std=np.array([1, 1]), friction=0.0, obs_noise=False, obs_noise_bound=0.075):
        self.gravity = 9.8
        self.masscart = 0.5
        self.masspole = 0.5
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.max_force = 10.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = 'euler'
        self.dynamics_noise = dynamics_noise # if we have noise on the dynamics
        self.friction=friction
        self.obs_noise = obs_noise
        self.obs_noise_bound = obs_noise_bound
        if dynamics_noise not in ['no_noise', 'gaussian', 'gaussian_state_dependent', "gumbel"]:
            raise ValueError("wrong noise")
        if type(std) == np.ndarray:
            # note: we divide here by tau to have N(s_{t+1}|s_{t} + delta_{t} * \mu(s_{t},a_{t}), \sigma)
            self.std = std / self.tau
        else:
            self.std = np.ones(2) * std / self.tau
        self.std_torch = torch.from_numpy(self.std).float()
        # Angle at which to fail the episode
        self.theta_threshold_radians = math.pi
        self.x_threshold = 3

        # set a maximum velocity, s.t. we can clip too high velocities
        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        high = np.array([
            self.x_threshold * 1.2,
            1e2,
            self.theta_threshold_radians * 1.2,
            1e2])

        # continuous action space between -force_mag and force_mag
        self._action_space = spaces.Box(np.array([-self.max_force]), np.array([self.max_force]))
        self._observation_space = spaces.Box(-high, high)

        self.viewer = None
        self.state = None

        self.steps_beyond_done = None


    def calc_acc(self, state, action):
        x, x_dot, theta, theta_dot = state
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        mass_matrix = np.array([[self.total_mass, self.polemass_length*costheta],[self.polemass_length*costheta, 4/3*self.polemass_length*self.length]])
        forces_vec = np.array([[-self.polemass_length*sintheta*theta_dot*theta_dot],[-self.polemass_length*self.gravity*sintheta]])
        torques = np.array([[action[0]],[0]])
        b = torques-forces_vec
        x = np.linalg.solve(mass_matrix, b)
        return x

    def forward(self, x):
        x_pos = x[:, 0]
        x_pos_dot = x[:, 1]
        theta = x[:, 2]
        theta_dot = x[:, 3]
        actions = x[:, 4]

        forces = torch.clamp(actions, min=-self.max_force, max=self.max_force)
        costheta = torch.cos(theta)
        sintheta = torch.sin(theta)
        temp = (forces - x_pos_dot * self.friction + self.polemass_length * theta_dot.pow(2) * sintheta) / self.total_mass
        thetaaccs = (self.gravity * sintheta - costheta * temp) / (
                    self.length * (4.0 / 3.0 - self.masspole * costheta * costheta / self.total_mass))
        xaccs = temp - self.polemass_length * thetaaccs * costheta / self.total_mass

        mean = torch.cat([xaccs.unsqueeze(1),thetaaccs.unsqueeze(1)], dim=1)

        if self.dynamics_noise == 'gaussian':
            std = self.std_torch.expand_as(mean)
            log_std = torch.log(std)
        elif self.dynamics_noise == 'gaussian_state_dependent':
            std = self.std_torch.expand_as(mean)*torch.cat([x_pos_dot.unsqueeze(1), theta_dot.unsqueeze(1)], dim=1)
            log_std = torch.log(std)

        return mean, log_std, std

    def step(self, action):
        state = self.state
        x, x_dot, theta, theta_dot = state
        force = np.clip(action, -self.max_force, self.max_force)[0]
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (force - x_dot * self.friction + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
                    self.length * (4.0 / 3.0 - self.masspole * costheta * costheta / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        x = x + self.tau * x_dot
        theta = theta + self.tau * theta_dot
        if self.dynamics_noise == 'gaussian':
            xacc += np.random.normal(0, self.std[0])
            thetaacc += np.random.normal(0, self.std[1])
        elif self.dynamics_noise == 'gaussian_state_dependent':
            xacc += np.random.normal(0, self.std[0]) * x_dot
            thetaacc += np.random.normal(0, self.std[1]) * theta_dot
        elif self.dynamics_noise == "gumbel":
            xacc += np.random.normal(0, self.std[0])
            thetaacc += np.random.gumbel(0, self.std[1])
        if self.obs_noise:
            x_dot_old = x_dot.copy()
            theta_dot_old = theta_dot.copy()
        x_dot = x_dot + self.tau * xacc
        theta_dot = theta_dot + self.tau * thetaacc

        # map theta between -pi and pi
        theta = np.mod(theta + np.pi, np.pi * 2) - np.pi

        self.state = (x, x_dot, theta, theta_dot)
        if self.obs_noise:
            # add the noise on the observed state and use it also for the observed differences xacc and thetaacc
            obs_state = self.state + np.random.uniform(low=-self.obs_noise_bound, high=self.obs_noise_bound, size=(4,))
            obs_xacc = (obs_state[1] - x_dot_old) / self.tau
            obs_thetaacc = (obs_state[3] - theta_dot_old) / self.tau
            obs_diff = np.array([obs_xacc, obs_thetaacc])
        else:
            obs_state = self.state
            obs_diff = np.array([xacc, thetaacc])


        done = x < -self.x_threshold \
               or x > self.x_threshold
        done = bool(done)

        if not done:
            reward = costheta # return cos of pole angle as return
        elif self.steps_beyond_done is None:
            # Pole just fell! -> return -100
            self.steps_beyond_done = 0
            reward = -500
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(obs_state), reward, done, {'obs_diff': obs_diff, 'true_next_state':np.array(self.state)}

    def reward(self, state):
        x, x_dot, theta, theta_dot = state
        done = x < -self.x_threshold \
               or x > self.x_threshold
        done = bool(done)
        costheta = np.cos(theta)
        if not done:
            r = costheta # return cos of pole angle as return
        elif self.steps_beyond_done is None:
            # Pole just fell! -> return -100
            self.steps_beyond_done = 0
            r = -500
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            r = 0.0
        return r, done

    def reset(self):
        self.state = np.random.uniform(low=-0.05, high=0.05, size=(4,))
        # we want the pole to start at bottom position
        if self.state[2] > 0:
            self.state[2] -= np.pi
        else:
            self.state[2] += np.pi
        self.steps_beyond_done = None
        return np.array(self.state)

    @overrides
    @property
    def action_space(self):
        return self._action_space

    @overrides
    @property
    def observation_space(self):
        return self._observation_space

    def set_state(self, state):
        self.state = state

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * 1.0
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(.8, .6, .4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5, .5, .8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

        if self.state is None: return None

        x = self.state
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        print("close called")
        if self.viewer:
            self.viewer.close()