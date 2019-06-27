# Copyright (c) 2018 Copyright holder of the paper Generative Adversarial Model Learning
# submitted to NeurIPS 2019 for review
# All rights reserved.

import casadi as ca

class CartpoleModelCasadi:

    def buildDynamicalSystem(self):
        g = 9.8

        x = ca.MX.sym('x')
        dx = ca.MX.sym('dx')
        theta = ca.MX.sym('theta')
        dtheta = ca.MX.sym('dtheta')
        u = ca.MX.sym('u')

        states = ca.vertcat(x, dx, theta, dtheta);
        controls = u;

        param1 = ca.MX.sym('param_1')
        param2 = ca.MX.sym('param_2')
        param3 = ca.MX.sym('param_3')

        params = ca.vertcat(param1, param2, param3)

        print("params_shape", params.shape)

        # build matrix for mass matrix
        phi_1_1 = ca.MX(2, 3)
        phi_1_1[0, 0] = ca.MX.ones(1)
        phi_1_1[1, 1] = ca.cos(theta)
        phi_1_2 = ca.MX(2, 3)
        phi_1_2[0, 1] = ca.cos(theta)
        phi_1_2[1, 2] = 4/3*ca.MX.ones(1)

        mass_matrix = ca.horzcat(ca.mtimes(phi_1_1, params), ca.mtimes(phi_1_2, params))

        phi_2 = ca.MX(2, 3)
        phi_2[0, 1] = -1 * ca.sin(theta) * dtheta * dtheta
        phi_2[1, 1] = -g * ca.sin(theta)

        forces = ca.mtimes(phi_2, params)

        actions = ca.vertcat(controls, ca.MX.zeros(1))

        b = actions - forces
        states_dd = ca.solve(mass_matrix, b)

        states_d = ca.vertcat(dx, states_dd[0], dtheta, states_dd[1])

        return states, states_d, controls, params

