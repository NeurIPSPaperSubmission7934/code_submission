# Copyright (c) 2018 Copyright holder of the paper Generative Adversarial Model Learning
# submitted to NeurIPS 2019 for review
# All rights reserved.

import casadi as ca

"""
returns a symbolic expression in casadi equivalent to the mod function of numpy which is equivalent to % in python
while casadi mod gives an expression equivalent to fmod in python
"""
def numpy_mod(x1, x2):
    return x1 - ca.floor(x1 / x2) * x2

