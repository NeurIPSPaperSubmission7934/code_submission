# Copyright (c) 2018 Copyright holder of the paper Generative Adversarial Model Learning
# submitted to NeurIPS 2019 for review
# All rights reserved.

# modified file from rlrl repo

import torch
import math
import numpy as np
import scipy
from rllab.torch.utils import torch as torch_utils
from rllab.torch.algos.base import Optimizer
from rllab.torch.algos.advantages import gae
from rllab.torch.models.mlp_critic import StateValue
# TODO: replace Variable as they are depricated
from torch.autograd import Variable
import rllab.misc.logger as logger

class ActorCriticOptimizer(Optimizer):
    def __init__(self, policy, discount=0.99, gae_lambda=0.95, l2_reg=1e-3,
                 **kwargs):
        super(ActorCriticOptimizer, self).__init__(policy, **kwargs)
        self.discount = discount
        self.gae_lambda = gae_lambda
        self.l2_reg = l2_reg

    def update_networks(self, policy,
                        actions, masks, rewards, states, num_episodes, *args):
        values = self.networks["critic"](Variable(states, volatile=True)).data

        # print("network values", values)

        if torch.isnan(values).sum() > 0 or torch.isinf(values).sum() > 0:
            print("values are nan or inf")
        advantages, returns = gae(
            rewards, masks, values, discount=self.discount,
            gae_lambda=self.gae_lambda, use_gpu=self.use_gpu)

        # print("returns", returns)
        # print("advantages", advantages)

        if torch.isnan(advantages).sum() > 0 or torch.isinf(advantages).sum() > 0:
            print("advantages are nan or inf")

        logger.record_tabular("avg_surr_reward", torch.mean(rewards).detach().numpy())
        logger.record_tabular("max_surr_return", torch.max(returns).detach().numpy())
        logger.record_tabular("min_surr_return", torch.min(returns).detach().numpy())
        logger.record_tabular("avg_surr_advantage", torch.mean(advantages).detach().numpy())
        logger.record_tabular("max_surr_advantage", torch.max(advantages).detach().numpy())
        logger.record_tabular("min_surr_advantage", torch.min(advantages).detach().numpy())
        optimizer_str_list = [key for key in
                              ["policy", "critic"][:len(self.optimizers)]]
        optimizers = [self.optimizers[key] for key in optimizer_str_list]
        success = self.step(policy, self.networks["critic"], *optimizers[:2],
                  states, actions, returns, advantages)
        if not success:
            logger.log("policy step has failed")
        return policy

    @classmethod
    def _init_networks(cls, obs_dim, action_dim):
        return {"critic": StateValue(obs_dim)}

    @staticmethod
    def step(*args):
        raise NotImplementedError

class TRPO(ActorCriticOptimizer):
    def __init__(self, policy, max_kl=1e-1, damping=1e-2, use_fim=False, entropy_coeff=0.0,
                 **kwargs):

        self.max_kl = max_kl
        self.damping = damping
        self.use_fim = use_fim
        self.entropy_coeff = entropy_coeff
        super(TRPO, self).__init__(policy, **kwargs)

    def step(self, policy_net, value_net, states, actions, returns, advantages):

        """update critic"""
        values_target = Variable(returns)

        """calculates the mean kl difference between 2 parameter settings"""
        def get_kl_diff(old_param, new_param):
            prev_params = torch_utils.get_flat_params_from(policy_net)
            with torch.no_grad():
                torch_utils.set_flat_params_to(policy_net, old_param)
                log_old_prob = torch.clamp(policy_net.get_log_prob(
                    Variable(states, volatile=True), Variable(actions)), min=np.log(1e-6))
                torch_utils.set_flat_params_to(policy_net, new_param)
                log_new_prob = torch.clamp(policy_net.get_log_prob(
                    Variable(states, volatile=True), Variable(actions)), min=np.log(1e-6))
            torch_utils.set_flat_params_to(policy_net, prev_params)
            return torch.mean(torch.exp(log_old_prob) * (log_old_prob-log_new_prob)).numpy()

        def get_value_loss(flat_params):
            torch_utils.set_flat_params_to(value_net,
                                           torch_utils.torch.from_numpy(flat_params))
            for param in value_net.parameters():
                if param.grad is not None:
                    param.grad.data.fill_(0)
            values_pred = value_net(Variable(states))
            value_loss = (values_pred - values_target).pow(2).mean()

            # weight decay
            for param in value_net.parameters():
                value_loss += param.pow(2).sum() * self.l2_reg
            value_loss.backward()

            # FIX: removed [0] since, mean reduces already it to an int (new functionality of new torch version?
            return value_loss.data.cpu().numpy(), \
                   torch_utils.get_flat_grad_from(
                       value_net.parameters()).data.cpu().numpy(). \
                       astype(np.float64)

        flat_params, _, opt_info = scipy.optimize.fmin_l_bfgs_b(
            get_value_loss,
            torch_utils.get_flat_params_from(value_net).cpu().numpy(), maxiter=25)
        torch_utils.set_flat_params_to(value_net, torch.from_numpy(flat_params))

        """update policy"""
        fixed_log_probs = torch.clamp(policy_net.get_log_prob(
            Variable(states, volatile=True), Variable(actions)), min=np.log(1e-6)).data
        """define the loss function for TRPO"""

        def get_loss(volatile=False):
            # more numberical stable: have a minimum value, s.t. we don't get -inf
            log_probs = torch.clamp(policy_net.get_log_prob(
                Variable(states, volatile=volatile), Variable(actions)), min=np.log(1e-6))
            ent = policy_net.get_entropy(Variable(states, volatile=volatile), Variable(actions)).mean()
            action_loss = -Variable(advantages) * torch.exp(
                log_probs - Variable(fixed_log_probs))
            # logger.log("advantage"+str(advantages))
            # logger.log("log_probs"+str(log_probs))
            # logger.log("mean"+str(torch.mean(torch.exp(
            #     log_probs - Variable(fixed_log_probs)))))
            # logger.log("action_loss_no_mean"+str(-action_loss))
            return action_loss.mean() - self.entropy_coeff * ent

        """use fisher information matrix for Hessian*vector"""

        def Fvp_fim(v):
            M, mu, info = policy_net.get_fim(Variable(states))
            mu = mu.view(-1)
            filter_input_ids = set() if policy_net.is_disc_action else \
                {info['std_id']}

            t = M.new(mu.size())
            t[:] = 1
            t = Variable(t, requires_grad=True)
            mu_t = (mu * t).sum()
            Jt = torch_utils.compute_flat_grad(mu_t, policy_net.parameters(),
                                               filter_input_ids=filter_input_ids,
                                               create_graph=True)
            Jtv = (Jt * Variable(v)).sum()
            Jv = torch.autograd.grad(Jtv, t, retain_graph=True)[0]
            MJv = Variable(M * Jv.data)
            mu_MJv = (MJv * mu).sum()
            JTMJv = torch_utils.compute_flat_grad(mu_MJv, policy_net.parameters(),
                                                  filter_input_ids=filter_input_ids,
                                                  retain_graph=True).data
            JTMJv /= states.shape[0]
            if not policy_net.is_disc_action:
                std_index = info['std_index']
                JTMJv[std_index: std_index + M.shape[0]] += \
                    2 * v[std_index: std_index + M.shape[0]]
            return JTMJv + v * self.damping

        """directly compute Hessian*vector from KL"""

        def Fvp_direct(v):
            kl = policy_net.get_kl(Variable(states))
            kl = kl.mean()

            grads = torch.autograd.grad(kl, policy_net.parameters(),
                                        create_graph=True)
            flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])

            kl_v = (flat_grad_kl * Variable(v)).sum()
            grads = torch.autograd.grad(kl_v, policy_net.parameters())
            flat_grad_grad_kl = torch.cat(
                [grad.contiguous().view(-1) for grad in grads]).data

            return flat_grad_grad_kl + v * self.damping

        Fvp = Fvp_fim if self.use_fim else Fvp_direct

        loss = get_loss()
        grads = torch.autograd.grad(loss, policy_net.parameters())
        loss_grad = torch.cat([grad.view(-1) for grad in grads]).data
        stepdir = conjugate_gradients(Fvp, -loss_grad, 10)

        shs = (stepdir.dot(Fvp(stepdir)))
        lm = np.sqrt(2 * self.max_kl / (shs + 1e-8))
        if np.isnan(lm):
            lm = 1.
        fullstep = stepdir * lm
        expected_improve = -loss_grad.dot(fullstep)

        prev_params = torch_utils.get_flat_params_from(policy_net)
        success, new_params = \
            line_search(policy_net, get_loss, prev_params, fullstep, expected_improve, get_kl_diff, self.max_kl)
        logger.record_tabular('TRPO_linesearch_success', int(success))
        logger.record_tabular("KL_diff", get_kl_diff(prev_params,new_params))
        torch_utils.set_flat_params_to(policy_net, new_params)
        logger.log("old_parameters" + str(prev_params.detach().numpy()))
        logger.log("new_parameters" + str(new_params.detach().numpy()))
        return success

    @staticmethod
    def _init_optimizers(networks, lr_rates=None):
        return []

def conjugate_gradients(Avp_f, b, nsteps, rdotr_tol=1e-10):
    x = b.clone()
    x[:] = 0
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)
    for i in range(nsteps):
        Avp = Avp_f(p)
        alpha = rdotr / torch.dot(p, Avp)
        x += alpha * p
        r -= alpha * Avp
        new_rdotr = torch.dot(r, r)
        betta = new_rdotr / rdotr
        p = r + betta * p
        rdotr = new_rdotr
        if rdotr < rdotr_tol:
            break
    return x


def line_search(model, f, x, fullstep, expected_improve_full, get_kl_diff, max_kl, max_backtracks=20,
                accept_ratio=0.01):
    fval = f(True).data[0]
    steps = [.5 ** x for x in range(max_backtracks)]
    for stepfrac in steps:
        x_new = x + stepfrac * fullstep
        # check x_new for NAN, it happened when optimizing cartpole model that somehow theta became NAN
        if torch.sum(torch.isnan(x_new)) > 0:
            logger.log("we somehow got NAN in linesearch", x, stepfrac, fullstep)
            continue
        torch_utils.set_flat_params_to(model, x_new)
        fval_new = f(True).data[0]
        actual_improve = fval - fval_new
        mean_kl = get_kl_diff(x,x_new)
        tolerance = max_kl*0.5
        if actual_improve > 0 and mean_kl <= max_kl + tolerance:
            return True, x_new
        logger.log("backtrack")
        if actual_improve <= 0 :
            logger.log("Violated because loss not improving. New loss: %f Old loss: %f" % (fval_new, fval))
        if mean_kl > max_kl + tolerance:
            logger.log("Violated because kl bound does not hold. MaxKL: %f MeanKL: %f" % (max_kl, mean_kl))
    if actual_improve <= 0:
        logger.log("Violated because loss not improving. New loss: %f Old loss: %f" % (fval_new, fval))
    if mean_kl > max_kl + tolerance:
        logger.log("Violated because kl bound does not hold. MaxKL: %f MeanKL: %f" % (max_kl, mean_kl))
    return False, x