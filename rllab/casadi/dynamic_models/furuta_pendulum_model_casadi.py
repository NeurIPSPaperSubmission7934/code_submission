# Copyright (c) 2018 Copyright holder of the paper Generative Adversarial Model Learning
# submitted to NeurIPS 2019 for review
# All rights reserved.

import casadi as ca

class FurutaPendulumModelCasadi:

    Rm = 8.4  # resistance
    kt = 0.042  # current-torque (N-m/A)
    km = 0.042  # back-emf constant (V-s/rad)
    g = 9.81

    def __init__(self, hackOn=False, trainMotor=False):
        self.trainMotor = trainMotor
        self.hackOn = hackOn

    def buildDynamicalSystem(self):
        # q1 is first link q2 is second link
        q1 = ca.MX.sym('q1')
        dq1 = ca.MX.sym('dq1')
        q2 = ca.MX.sym('q2')
        dq2 = ca.MX.sym('dq2')
        u = ca.MX.sym('u')

        theta = ca.MX.sym("theta", 7, 1)

        if self.trainMotor:
            Rm = ca.MX.sym("Rm")
            km = ca.MX.sym("km")
            params = ca.vertcat(ca.MX.sym("params", 7, 1), Rm, km)
        else:
            Rm = self.Rm
            km = self.km
            params = theta

        states = ca.vertcat(q1, q2, dq1, dq2);
        controls = u # km * (u - km * dq1) / Rm;

        if self.hackOn:
            # convert actions which are given as torques to voltage to be consistent with the controller inputs of Lukas
            u = u*Rm/km

        controls_torque = km * (u - km * dq1) / Rm;

        # build matrix for mass matrix
        phi_1_1 = ca.MX(2, 7)
        phi_1_1[0, 0] = ca.MX.ones(1)
        phi_1_1[0, 1] = ca.sin(q2) * ca.sin(q2)
        phi_1_1[1, 2] = ca.cos(q2)
        phi_1_2 = ca.MX(2, 7)
        phi_1_2[0, 2] = - ca.cos(q2)
        phi_1_2[1, 3] = ca.MX.ones(1)
        mass_matrix = ca.horzcat(ca.mtimes(phi_1_1, params), ca.mtimes(phi_1_2, params))

        phi_2 = ca.MX(2, 7)
        phi_2[0, 1] = ca.sin(2*q2) * dq1 * dq2
        phi_2[0, 2] = ca.sin(q2) * dq2 * dq2
        phi_2[0, 5] = dq1
        phi_2[1, 1] = -1 /2 * ca.sin(2*q2) * dq1 * dq1
        phi_2[1, 4] = self.g * ca.sin(q2)
        phi_2[1, 6] = dq2

        forces = ca.mtimes(phi_2, params)

        actions = ca.vertcat(controls_torque, ca.MX.zeros(1))

        b = actions - forces
        states_dd = ca.solve(mass_matrix, b)

        states_d = ca.vertcat(dq1, dq2, states_dd[0], states_dd[1])

        return states, states_d, controls, params