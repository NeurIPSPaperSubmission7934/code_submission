# Copyright (c) 2018 Copyright holder of the paper Generative Adversarial Model Learning
# submitted to NeurIPS 2019 for review
# All rights reserved.

import casadi as ca
from rllab.casadi.utils import numpy_mod

def euler_integration(z, z_d, u, integrator_dt, integrator_stepsize, angular_idx):
    z_d_generator = ca.Function("z_d_generator", [z, u], [z_d])
    current_z = z
    for i in range(integrator_stepsize):
        current_z_d = z_d_generator(current_z, u)
        current_z += integrator_dt * current_z_d
        current_z[angular_idx] = numpy_mod(current_z[angular_idx] + ca.pi, ca.pi*2) - ca.pi
        #current_z[angular_idx] = ((current_z[angular_idx] + ca.pi) - ca.floor((current_z[angular_idx] + ca.pi)/(ca.pi*2))*ca.pi*2) - ca.pi
    return current_z

def rk_integration(z, z_d, u, integrator_dt, integrator_stepsize, angular_idx):
    f = ca.Function('z_dot', [z, u], [z_d])
    h = integrator_dt * integrator_stepsize

    k1 = f(z, u)
    k2 = f(z + h / 2. * k1, u)
    k3 = f(z + h / 2. * k2, u)
    k4 = f(z + h * k3, u)

    current_z = z + h / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    current_z[angular_idx] = numpy_mod(current_z[angular_idx] + ca.pi, ca.pi*2) - ca.pi

    return current_z # , k1, k2, k3, k4