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
import numpy as np
from rllab.envs.base import Env
from rllab import spaces
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from rllab.envs.cartpole_swingup_env import CartPoleSwingUpEnv
from rllab.misc.overrides import overrides
from collections import OrderedDict
from rllab.torch.utils.misc import batch_diagonal
from rllab.torch.utils import torch as torch_utils
import itertools

class CartPoleModel(Env):
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

    _version = 1

    def __init__(self, initRandom=True, init_std=1, trainOnlyStd=False):
        self._parameters = OrderedDict()
        self.gravity = 9.8
        self.masscart = 0.5
        self.masspole = 0.5
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.max_force = 10.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = 'euler'
        self._wrapped_env = CartPoleSwingUpEnv()

        # Angle at which to fail the episode
        self.theta_threshold_radians = math.pi
        self.x_threshold = 3

        self.x_dot_state_idx = [1, 3]
        self.x_state_idx = [0, 2]

        self.state_dim = 4
        self.u_dim = 1

        # set a maximum velocity, s.t. we can clip too high velocities
        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        self.high = np.array([
            self.x_threshold * 1.2,
            1e2,
            self.theta_threshold_radians * 1.2,
            1e2])

        # continuous action space between -force_mag and force_mag
        self._action_space = spaces.Box(np.array([-self.max_force]), np.array([self.max_force]))
        self._observation_space = spaces.Box(-self.high, self.high)

        if initRandom:
            theta = torch.tensor(np.random.uniform(size=3), requires_grad=not trainOnlyStd, dtype=torch.float32)
        else:
            theta = torch.tensor([self.masspole+self.masscart,
                                  self.polemass_length,
                                  self.polemass_length*self.length], requires_grad=not trainOnlyStd, dtype=torch.float32)

        # to have similarity with nn.module
        self.register_parameter('theta', theta)
        if type(init_std) == np.ndarray:
            init_std = torch.tensor([init_std[0] / self.tau, init_std[1] / self.tau], dtype=torch.float32, requires_grad=True)
        else:
            init_std = torch.tensor([init_std/self.tau, init_std/self.tau], dtype=torch.float32, requires_grad=True)
        self.register_parameter('std', init_std)

        # needed to be compatible with PGSupportingPolicy
        self.is_disc_action=False

        # needed for the rollout
        self._normalized_input_obs = [False, False, False, False]
        self._normalized_input_a = [False]
        self._normalized_output_state_diff = [False, False]
        self._normalized_output = [False, False, False, False]


    @property
    @overrides
    def input_dim(self):
        # current state + action
        return self.state_dim + self.u_dim

    @property
    @overrides
    def output_dim(self):
        # next state + action
        return self.state_dim

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

    def step(self, action):
        # combine current obs + actions to input
        action_tensor = torch.from_numpy(action).float()
        input = torch.cat([self.current_obs, action_tensor]).unsqueeze(0) # add addtional batch dimension

        output_mean, _ , output_std = self.forward(input)
        output = torch.normal(output_mean, output_std).detach().numpy() # sampled output for euler states

        # update x states
        obs = self.current_obs.detach().numpy()
        obs[self.x_state_idx] += obs[self.x_dot_state_idx] * self.tau
        # update x_dot states
        obs[self.x_dot_state_idx] += output * self.tau
        # map theta between -pi and pi
        obs[2] = np.mod(obs[2] + np.pi, np.pi * 2) - np.pi
        # clamp the velocities to max values s.t. we cannot get numerical errors?
        obs[1] = np.clip(obs[1], -self.high[1], self.high[1])
        obs[3] = np.clip(obs[3], -self.high[3], self.high[3])

        r, done = self._wrapped_env.reward(obs)
        self.current_obs = torch.from_numpy(obs).float()
        return obs, r, done, {'obs_diff': output}

    def reset(self, **kwargs):
        obs = self._wrapped_env.reset(**kwargs)
        self.current_obs = torch.from_numpy(obs).float()
        return obs

    def forward(self, x):

        q = x[:, self.x_state_idx]
        qd = x[:, self.x_dot_state_idx]
        qd1 = qd[:, 0, None]
        qd2 = qd[:, 1, None]
        c2 = torch.cos(q[:, 1, None])
        s2 = torch.sin(q[:, 1, None])
        batched_theta = self.theta.unsqueeze(0).expand(x.size(0), len(self.theta)).unsqueeze(2)

        ## TODO: initialize the matrix using a for loop could look somewhat nicer and should be done as soon we move to more complex problems
        ## calculate the feature matrix for the mass matrix
        phi_1_1_11 = torch.ones(q[:, 0, None].shape, dtype=torch.float32)
        phi_1_1_12 = torch.zeros(q[:, 0, None].shape, dtype=torch.float32)
        phi_1_1_13 = torch.zeros(q[:, 0, None].shape, dtype=torch.float32)
        phi_1_1_21 = torch.zeros(q[:, 0, None].shape, dtype=torch.float32)
        phi_1_1_22 = c2
        phi_1_1_23 = torch.zeros(q[:, 0, None].shape, dtype=torch.float32)
        phi_1_1_1 = torch.cat(
            [phi_1_1_11.unsqueeze(1), phi_1_1_12.unsqueeze(1), phi_1_1_13.unsqueeze(1)], dim=2)
        phi_1_1_2 = torch.cat(
            [phi_1_1_21.unsqueeze(1), phi_1_1_22.unsqueeze(1), phi_1_1_23.unsqueeze(1)], dim=2)
        phi_1_1 = torch.cat([phi_1_1_1, phi_1_1_2], dim=1)
        phi_1_c1 = phi_1_1.bmm(batched_theta)

        phi_1_2_11 = torch.zeros(q[:, 0, None].shape, dtype=torch.float32)
        phi_1_2_12 = c2
        phi_1_2_13 = torch.zeros(q[:, 0, None].shape, dtype=torch.float32)
        phi_1_2_21 = torch.zeros(q[:, 0, None].shape, dtype=torch.float32)
        phi_1_2_22 = torch.zeros(q[:, 0, None].shape, dtype=torch.float32)
        phi_1_2_23 = 4/3*torch.ones(q[:, 0, None].shape, dtype=torch.float32)
        phi_1_2_1 = torch.cat(
            [phi_1_2_11.unsqueeze(1), phi_1_2_12.unsqueeze(1), phi_1_2_13.unsqueeze(1)], dim=2)
        phi_1_2_2 = torch.cat(
            [phi_1_2_21.unsqueeze(1), phi_1_2_22.unsqueeze(1), phi_1_2_23.unsqueeze(1)], dim=2)
        phi_1_2 = torch.cat([phi_1_2_1, phi_1_2_2], dim=1)
        phi_1_c2 = phi_1_2.bmm(batched_theta)

        phi_1 = torch.cat([phi_1_c1, phi_1_c2], dim=2)

        ## calcualte the feature matrix for the forces vector part
        # define matrix row wise
        phi_2_11 = torch.zeros(q[:, 0, None].shape, dtype=torch.float32)
        phi_2_12 = -s2 * qd2 **2
        phi_2_13 = torch.zeros(q[:, 0, None].shape, dtype=torch.float32)
        phi_2_21 = torch.zeros(q[:, 0, None].shape, dtype=torch.float32)
        phi_2_22 = -s2 * self.gravity
        phi_2_23 = torch.zeros(q[:, 0, None].shape, dtype=torch.float32)
        phi_2_1 = torch.cat([phi_2_11.unsqueeze(1), phi_2_12.unsqueeze(1), phi_2_13.unsqueeze(1)], dim=2)
        phi_2_2 = torch.cat([phi_2_21.unsqueeze(1), phi_2_22.unsqueeze(1), phi_2_23.unsqueeze(1)], dim=2)
        phi_2 = torch.cat([phi_2_1, phi_2_2], dim=1)

        forces = phi_2.bmm(batched_theta)
        action_tensor = x[:, -1, None]
        # clip action to max force
        action_tensor = torch.clamp(action_tensor, min=-self.max_force, max=self.max_force)
        stackedActions = torch.cat([action_tensor, torch.zeros(action_tensor.shape, dtype=torch.float32)], dim=1)
        b = stackedActions.unsqueeze(2) - forces  # need to unsqueeze such that we have # batch x dim1 x 1

        if torch.__version__ == '0.4.0':
            # no batchwise calculation of gesv
            if b.shape[0] > 1:
                raise NotImplementedError("forward does not work with batches in torch 0.4.0")
            X, LU = torch.gesv(b[0,:,:], phi_1[0,:,:])
        else:
            X, LU = torch.gesv(b, phi_1)
        mean_qdd = X.squeeze()
        if torch.isnan(mean_qdd).sum() > 0:
            print("qdd is NAN this should not happen!", phi_1, b)
            print("x",x)
            print("c2", c2)
            print("s2", s2)
            print("theta", batched_theta)

        std = self.std.expand_as(mean_qdd)
        log_std = torch.log(std)

        return mean_qdd, log_std, std

    def select_action(self, x, t):
        raise NotImplementedError("Not supposed to be called in our current architecture. Use step!")

    def get_log_prob(self, x, output_torch_var):
        means, log_std, std = self.forward(x)
        # calculate difference in x dot states from actions and scale them by tau
        multivariate_normal = MultivariateNormal(means, covariance_matrix=batch_diagonal(std.pow(2)))
        # NOTE: we need here a unsqueeze(1) to work with the rlrl framework, else we get in trouble when calculating
        # advantage * log_prob
        log_prob = multivariate_normal.log_prob(output_torch_var).unsqueeze(1)
        return log_prob

    def get_entropy(self, x, output_torch_var):
        action_mean, log_std, _ = self.forward(x)
        return normal_entropy(action_mean, log_std)

    def get_fim(self, x):
        import warnings
        warnings.warn("This function might not work properly yet!")
        mean, _ , _ = self.forward(x)
        cov_inv = self.std.data.exp().pow(-2).squeeze(0).repeat(x.size(0))
        param_count = 0
        std_index = 0
        id = 0
        for name, param in self.named_parameters():
            if name == "std":
                std_id = id
                std_index = param_count
            param_count += param.data.view(-1).shape[0]
            id += 1
        return cov_inv, mean, {'std_id': std_id, 'std_index': std_index}

    @overrides
    @property
    def action_space(self):
        return self._action_space

    @overrides
    @property
    def observation_space(self):
        return self._observation_space


    def render(self, *args, **kwargs):
        # set state of environment and then render
        self._wrapped_env.set_state(self.current_obs)
        self._wrapped_env.render(*args, **kwargs)


    def set_param_values(self, params):
        # assume that params is a tensor of shape 5, theta has dim 3 and we have 2 dimensional noise
        if not params.shape[0] == 5:
            raise ValueError("something went wrong")

        torch_utils.set_flat_params_to(self, params)



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



def normal_entropy(mean, log_std):
    # take formula from https://en.wikipedia.org/wiki/Multivariate_normal_distribution for entropy
    # and use knowledge that we only have a diagonal covariance matrix
    return torch.sum(log_std + torch.log(torch.sqrt(torch.tensor(2 * np.pi * np.e))), dim=-1)

def normal_log_density(x, mean, log_std, std):
    var = std.pow(2)
    log_density = -(x - mean).pow(2) / (2 * var) - \
                  0.5 * math.log(2 * math.pi) - log_std
    return log_density.sum(1, keepdim=True)