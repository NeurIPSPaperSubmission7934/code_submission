# Copyright (c) 2018 Copyright holder of the paper Generative Adversarial Model Learning
# submitted to NeurIPS 2019 for review
# All rights reserved.

from rllab.algos.base import Algorithm
import rllab.misc.logger as logger
import casadi as ca
import numpy as np
from rllab.misc.overrides import overrides

############ define callback function to save parameters and log stats ################

class saveParamLogStatsCallback(ca.Callback):

    def __init__(self, name, nx, ng, param_scale, algo ,opts={}):
        ca.Callback.__init__(self)

        self.nx = nx
        self.ng = ng
        self.param_scale = param_scale
        self.itr = 0
        self.algo = algo

        # Initialize internal objects
        self.construct(name, opts)

    def get_n_in(self): return ca.nlpsol_n_out()

    def get_n_out(self): return 1

    def get_name_in(self, i): return ca.nlpsol_out(i)

    def get_name_out(self, i): return "ret"

    def get_sparsity_in(self, i):
        n = ca.nlpsol_out(i)
        if n == 'f':
            return ca.Sparsity.scalar()
        elif n in ('x', 'lam_x'):
            return ca.Sparsity.dense(self.nx)
        elif n in ('g', 'lam_g'):
            return ca.Sparsity.dense(self.ng)
        else:
            return ca.Sparsity(0, 0)

    def init(self):
      print('Initializing object')

    def finalize(self):
      print('Finalizing object construction')

    def eval(self, arg):
        darg = {}
        for (i, s) in enumerate(ca.nlpsol_out()): darg[s] = arg[i]

        solution = np.array((darg['x'][:len(self.param_scale)]*self.param_scale).T)
        loss = darg['f']

        logger.record_tabular('loss', loss)
        logger.record_tabular('solution', solution)

        self.algo.solution = solution
        params, torch_params = self.algo.get_itr_snapshot(self.itr)
        logger.save_itr_params(self.itr, params, torch_params)

        logger.dump_tabular(with_prefix=False)

        self.itr += 1

        return [0]

"""
class which is used to learn the parameters of a given model via single / multiple shooting:
"""
class ParameterEstimation(Algorithm):

    def __init__(self, expert_paths, model, estimationMode, integration_func, param_guess, modeOptions={}, integrationFuncOptions={}):
        # paths / trajectories of expert
        self.expert_paths = expert_paths
        # the model of the environment
        self.model = model
        # mode for estimating the parameters, either single or multiple shooting
        self.estimationMode = estimationMode
        # dict with additional option parameters for the chosen parameter estimation mode
        self.modeOptions = modeOptions
        # function used for numerical integration, e.g. Euler, RK4
        self.integration_func = integration_func
        # dict with additional option parameters for the numerical integration function
        self.integrationFuncOptions = integrationFuncOptions
        # initial parameter guess, scale them to be in range of -0.1 and 100
        # (apparently it works better with param in this range)
        self.param_guess = param_guess

    def train(self):

        # build dynamic system from model
        states, states_d, controls, params = self.model.buildDynamicalSystem()

        integrate_f = self.integration_func(states, states_d, controls, **self.integrationFuncOptions)

        # Create a function that simulates one step propagation in a sample
        step = ca.Function("step", [states, controls, params], [integrate_f])

        # Use just-in-time compilation to speed up the evaluation
        if ca.Importer.has_plugin('clang'):
            with_jit = True
            compiler = 'clang'
            print("using clang")
        elif ca.Importer.has_plugin('shell'):
            with_jit = True
            compiler = 'shell'
            print("using shell")
        else:
            print("WARNING; running without jit. This may result in very slow evaluation times")
            with_jit = False
            compiler = ''

        if self.estimationMode == "singleshooting":
            all_residuals = []

            # go through all paths and create
            for path in self.expert_paths:
                x0 = ca.DM(path["observations"][0, :])
                y_data = ca.DM(path["next_observations"])
                actions = ca.DM(path["actions"])
                # function to simulate a whole trajectory depending on length of length of expert trajectory
                timesteps = y_data.shape[0]
                sim_one_traj = step.mapaccum("all_steps", timesteps)
                traj_states_symbolic = sim_one_traj(x0, actions, ca.repmat(params * self.modeOptions["scale"], 1, timesteps))
                residuals_one_path = y_data - traj_states_symbolic.T
                all_residuals.append(residuals_one_path)

            residuals = ca.vertcat(*all_residuals)

            objective = 0.5 * ca.dot(residuals, residuals)

            nlp = {'x': params, 'f': objective}
            X = params
            x0_guess = self.param_guess

        if self.estimationMode == "multipleshooting":

            all_residuals = []
            all_eq_const = []
            all_init_points = []
            all_shooting_points = []
            for path in self.expert_paths:
                # custom way to set the number of shooting points by specifying a fixed distance between 2 shooting points
                # shooting points will be put equidistant on the time scale
                y_data = ca.DM(path["next_observations"])
                actions = ca.DM(path["actions"])
                timesteps = y_data.shape[0]
                # how many step difference do we want to have between two shooting points
                n_step_pred = self.modeOptions["n_step_pred_sp"]
                shootingPointsCount = np.ceil(timesteps / n_step_pred)

                # print("spc", shootingPointsCount)

                shootingPointStartStates_symbolic_one_path = ca.MX.sym("X", 4, int(shootingPointsCount))

                # predict whole trajectory
                traj_all_parts = []
                sim_one_traj_part = step.mapaccum("all_steps", n_step_pred)
                # TODO: let's use for now a for loop, but think of how this can be rewritten to map
                i = 0

                for j in range(int(shootingPointsCount)):
                    x0 = shootingPointStartStates_symbolic_one_path[:, j]

                    if j == shootingPointsCount - 1:
                        # check if timesteps and n_step_pred match perfectly
                        if not np.floor(timesteps / n_step_pred) == shootingPointsCount:
                            rest = int(timesteps - (shootingPointsCount-1) * n_step_pred)
                            sim_one_short_traj_part = step.mapaccum("all_steps", rest)
                            traj_part_states_symbolic = sim_one_short_traj_part(x0, actions[i:i + rest],
                                                                          ca.repmat(params * self.modeOptions["scale"],
                                                                                    1, rest))
                            traj_all_parts.append(traj_part_states_symbolic)
                            i += n_step_pred
                            continue

                    traj_part_states_symbolic = sim_one_traj_part(x0, actions[i:i + n_step_pred],
                                                                  ca.repmat(params * self.modeOptions["scale"], 1, n_step_pred))
                    traj_all_parts.append(traj_part_states_symbolic)
                    i += n_step_pred

                traj_states_symbolic = ca.horzcat(*traj_all_parts)

                # print("traj_states_symbolic", traj_states_symbolic.shape)

                # equality between simulation start and end point between 2 neighboring shooting points

                # need to select the right points of the traj_states to get the shooting point end states
                shootingPointEndStates_symbolic = traj_states_symbolic[:, ::n_step_pred]

                # print("end_states", shootingPointEndStates_symbolic.shape)
                # print(shootingPointEndStates_symbolic.shape)

                eq_gaps_one_path = shootingPointEndStates_symbolic[:, :-1] - shootingPointStartStates_symbolic_one_path[
                                                                             :, 1:]

                #  TODO: cut off too long shooting point end states

                residuals_one_path = y_data - traj_states_symbolic.T

                all_eq_const.append(eq_gaps_one_path)
                all_residuals.append(residuals_one_path)
                all_shooting_points.append(shootingPointStartStates_symbolic_one_path)
                all_init_points.append(y_data[::n_step_pred, :].T)

            residuals = ca.vertcat(*all_residuals)
            eq_gaps = ca.horzcat(*all_eq_const)
            shootingPointStartStates_symbolic = ca.horzcat(*all_shooting_points)
            init_points = ca.horzcat(*all_init_points)

            # print("residuals", residuals)
            # print("eq_gaps", eq_gaps)
            # print("shootingpoints", shootingPointStartStates_symbolic)

            # free variables to optimize for
            X = ca.veccat(params, shootingPointStartStates_symbolic)

            objective = 0.5 * ca.dot(residuals, residuals)

            nlp = {'x': X, 'f': 0.5 * objective, 'g': ca.vec(eq_gaps)}

            x0_guess = ca.veccat(self.param_guess, init_points)

        if "g" in nlp:
            number_eq_cons = nlp["g"].shape[0]
        else:
            number_eq_cons = 0

        mycallback = saveParamLogStatsCallback('saveParamCallback', nx=nlp["x"].shape[0], ng=number_eq_cons, param_scale=self.modeOptions["scale"], algo=self)

        ############ Create a Gauss-Newton solver ##########
        def gauss_newton(e, nlp, V):
            J = ca.jacobian(e, V)
            H = ca.triu(ca.mtimes(J.T, J))
            sigma = ca.MX.sym("sigma")
            hessLag = ca.Function('nlp_hess_l', {'x': V, 'lam_f': sigma, 'hess_gamma_x_x': sigma * H},
                                  ['x', 'p', 'lam_f', 'lam_g'], ['hess_gamma_x_x'])
                                  #,dict(jit=with_jit, compiler=compiler))

            return ca.nlpsol("solver", "ipopt", nlp, {"hess_lag": hessLag, "ipopt.linear_solver":'mumps', "ipopt.max_iter":self.modeOptions["n_itr"],
                                                      "iteration_callback":mycallback,})

        # name = 'hessLag'
        # cname = hessLag.generate("test.c")
        #
        # from os import system
        # import time
        #
        # oname_O3 = name + '_O3.so'
        # print('Compiling with O3 optimization: ', oname_O3)
        # t1 = time.time()
        # system('gcc -fPIC -shared -O3 ' + cname + ' -o ' + oname_O3)
        # t2 = time.time()
        # print('time = ', (t2 - t1) * 1e3, ' ms')
        # hessLag = ca.external(name, './' + oname_O3)

        # solver = gauss_newton(residuals, nlp, X)
        # solver.generate_dependencies("nlp.c")

        # from os import system
        # flag = system("gcc -fPIC -shared -O3 nlp.c -o nlp.so")

        # solver = ca.nlpsol("solver", "ipopt", "./nlp.c", {"compiler":"shell"});
        # solver = ca.external("gauss_newton_solver", './' + 'nlp.so')
        # sol = solver(x0_guess)["x"]

        solver = gauss_newton(residuals, nlp, X)
        lowerBound = - ca.DM.inf(X.shape)
        lowerBound[:self.param_guess.shape[0]] = ca.DM.zeros(self.param_guess.shape[0])
        opt_result = solver(x0=x0_guess, lbx=lowerBound)

        stats = solver.stats()
        logger.log(str(stats))
        sol = opt_result["x"]
        self.solution = np.array((sol[:self.param_guess.shape[0]]*self.modeOptions["scale"]).T)

        # save result
        logger.log("saving snapshot...")
        params, torch_params = self.get_itr_snapshot(self.modeOptions["n_itr"])
        params["stats"] = stats
        logger.save_itr_params(self.modeOptions["n_itr"], params, torch_params)
        logger.log("saved")

    @overrides
    def get_itr_snapshot(self, itr):
        if itr == 0:
            return dict(
                itr=itr,
                expert_paths=self.expert_paths,
                estimationMode=self.estimationMode,
                parameters=self.solution,
                modeOptions=self.modeOptions,
                integrationFuncOptions=self.integrationFuncOptions,
            ), None
        else:
            return dict(
                itr=itr,
                parameters=self.solution,
                modeOptions=self.modeOptions,
                integrationFuncOptions=self.integrationFuncOptions,
            ), None


