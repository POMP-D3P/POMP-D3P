# sac-svg(1) rollout one step when update policy
import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.optim import Adam
from buffer import ReplayMemory
from nets import DeterministicPolicy
from nets import GaussianPolicy
from nets import QNetwork
from utility import soft_update, hard_update
import copy
from model import EnsembleEnv as EnsembleEnv
import logging
import math
from functorch import jacrev, jacfwd
import metrics
from utility import get_lrschedule

logger = logging.getLogger(__name__)


def reward_logpi(x, y, alpha, z=None, reverse=False):
    if not reverse:
        alpha = -alpha
    if z is None:
        z = 0

    return x + alpha * y + z
    # if reverse:
    #     return x + y
    # else:
    #     return x - y


def batch_jacobian(func, x):
    assert len(x.shape) == 2

    def _func_sum(x):
        return func(x).sum(dim=0)

    return jacrev(_func_sum, argnums=0)(x).movedim(1, 0)


def batch_jacobian_tupleinput(func, x, fwd=False):
    assert all([len(a.shape) == 2 for a in x])

    def _func_sum(*args):
        return func(*args).sum(dim=0)

    jacobian = jacrev if not fwd else jacfwd
    result = jacobian(_func_sum, argnums=(0, 1))(*x)
    return result[0].movedim(1, 0), result[1].movedim(1, 0)


def batch_jacobian_multitupleinput(func, x, fwd=False, argnum=2):
    assert all([len(a.shape) == 2 for a in x])
    assert argnum > 0 and len(x) == argnum

    def _func_sum(*args):
        return func(*args).sum(dim=0)

    jacobian = jacrev if not fwd else jacfwd
    result = jacobian(_func_sum, argnums=tuple(range(argnum)))(*x) 
    return [x.movedim(1, 0) for x in result]


def torch_clip(x, min_x, max_x):
    min_x = min_x.reshape(1, -1)
    max_x = max_x.reshape(1, -1)
    return torch.min(torch.max(x, min_x), max_x)


class Termination_Fn(object):
    def __init__(self, env_name):
        self.env_name = env_name
        logger.info(f"Env: {env_name}")

    def done(self, obs, act, next_obs):
        if self.env_name == "HalfCheetah-v2" or self.env_name == "Reacher-v2":
            done = np.array([False]).repeat(len(obs))
            # done = done[:,None]
            return done
        elif self.env_name == "Hopper-v2":
            height = next_obs[:, 0]
            angle = next_obs[:, 1]
            not_done = (
                np.isfinite(next_obs).all(axis=-1)
                * np.abs(next_obs[:, 1:] < 100).all(axis=-1)
                * (height > 0.7)
                * (np.abs(angle) < 0.2)
            )
            done = ~not_done
            # done = done[:,None]
            return done
        elif self.env_name == "Walker2d-v2":
            height = next_obs[:, 0]
            angle = next_obs[:, 1]
            not_done = (height > 0.8) * (height < 2.0) * (angle > -1.0) * (angle < 1.0)
            done = ~not_done
            # done = done[:,None]
            return done
        elif self.env_name == "AntTruncatedObs-v2":
            x = next_obs[:, 0]
            not_done = np.isfinite(next_obs).all(axis=-1) * (x >= 0.2) * (x <= 1.0)

            done = ~not_done
            # done = done[:,None]
            return done
        elif self.env_name == "InvertedPendulum-v2":
            notdone = np.isfinite(next_obs).all(axis=-1) * (np.abs(next_obs[:, 1]) <= 0.2)
            done = ~notdone

            # done = done[:,None]

            return done
        elif self.env_name == "HumanoidTruncatedObs-v2":
            z = next_obs[:, 0]
            done = (z < 1.0) + (z > 2.0)

            # done = done[:,None]
            return done
        else:
            assert 1 == 2


class Agent(object):
    def __init__(self, num_inputs, action_space, args):
        self.device = torch.device("cuda" if args.cuda else "cpu")

        self.model_ensemble = EnsembleEnv(
            args.num_networks,
            args.num_elites,
            state_size=num_inputs,
            action_size=action_space.shape[0],
            use_decay=args.use_decay,
            args=args,
        )

        self.gamma = args.gamma
        self.gamma_tensor = torch.FloatTensor([args.gamma]).to(self.device)
        self.tau = args.tau
        self.alpha = args.alpha

        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        self.critic = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(
            device=self.device
        )
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)
        self.critic_lrscheduler = get_lrschedule(args, self.critic_optim)

        self.critic_target = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(
            self.device
        )
        hard_update(self.critic_target, self.critic)

        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(
                    torch.Tensor(action_space.shape).to(self.device)
                ).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)
                self.alpha_lrscheduler = get_lrschedule(args, self.alpha_optim)

            self.policy = GaussianPolicy(
                num_inputs, action_space.shape[0], args.hidden_size, action_space
            ).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)
            self.policy_lrscheduler = get_lrschedule(args, self.policy_optim)
        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(
                num_inputs, action_space.shape[0], args.hidden_size, action_space
            ).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)
            # print('policy type deterministic')

        # test_sac = False
        # if test_sac:
        #     self.policy.load_state_dict(torch.load('policy_p.file'))
        self.action_space = action_space
        self.DIM_X = num_inputs
        self.DIM_U = action_space.shape[0]
        self.state_sequence = []
        self.p = []
        self.action_sequence = []
        self.log_pi_sequence = []

        self.loss_function = nn.MSELoss()
        self.flag = 0
        self.memory = None

        self.termination_fn = Termination_Fn(args.env_name)
        self.policy_direct_bp = args.policy_direct_bp

        self.ddph = args.H if args.ddph is None else args.ddph
        self.ddp_min_delta = args.ddp_min_delta
        self.ddp_max_delta = args.ddp_max_delta
        self.ddp_delta_decay = args.ddp_delta_decay
        self.ddp_clipk = args.ddp_clipk
        self.ddp_iterations = args.ddp_iterations
        self.ddp_num_samples = args.ddp_num_samples
        self.norm_bound = 1000.0 if args.norm_bound is None else args.norm_bound
        self.args = args
        self.learnq_ddp = getattr(args, "learnq_ddp", False)
        self.learnq_num = getattr(args, "learnq_num", 8)
        self.penalize_var = getattr(args, "penalize_var", False)
        self.reverse_logpi = getattr(args, "reverse_logpi", False)
        self.logpi_alpha = getattr(args, "logpi_alpha", None)
        self.logpi_each_step = getattr(args, "logpi_each_step", False)
        self.clip_gn = getattr(args, "clip_gn", None)
        self.gbp = getattr(args, "gbp", False)
        self.ddp_delta_decay_legacy = getattr(args, "ddp_delta_decay_legacy", False)

    def step_lrscheduler(self):
        self.critic_lrscheduler.step()
        self.alpha_lrscheduler.step()
        self.policy_lrscheduler.step()
        self.model_ensemble.model.ensemble_model.lr_scheduler.step()

    def get_lr(self):
        return {
            "critic_lr": f"{self.critic_lrscheduler.get_last_lr()[0]:.5e}",
            "alpha_lr": f"{self.alpha_lrscheduler.get_last_lr()[0]:.5e}",
            "policy_lr": f"{self.policy_lrscheduler.get_last_lr()[0]:.5e}",
            "model_lr": f"{self.model_ensemble.model.ensemble_model.lr_scheduler.get_last_lr()[0]:.5e}",
        }

    def select_action(self, state, evaluate=False, ddp=False, ddp_iters=None, init_action=None):
        if ddp:
            try:
                if not self.gbp:
                    return (
                        self.select_action_ddp(
                            state, evaluate, ddp_iters=ddp_iters, init_action=init_action
                        )
                        .detach()
                        .cpu()
                        .numpy()
                    )
                else:
                    return (
                        self.select_action_gbp(state, evaluate, ddp_iters=ddp_iters)
                        .detach()
                        .cpu()
                        .numpy()
                    )
            except Exception as e:
                logger.info(f"Catch exception {e}, and fallback to SAC.")
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def select_next_action(self, next_state_batch):
        if self.learnq_ddp and self.model_ensemble.trained:
            try:
                next_state_action = self.select_action_ddp(
                    next_state_batch,
                    evaluate=False,
                    ddp_iters=None,
                )
                next_state_log_pi = self.policy.log_prob(next_state_batch, next_state_action)
                return next_state_action, next_state_log_pi
            except Exception as e:
                logger.info(f"Catch exception {e}, and fallback to SAC.")
        next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
        return next_state_action, next_state_log_pi

    def select_action_ddp(self, state, evaluate=False, ddp_iters=None, init_action=None):
        enable_logpi = evaluate and self.logpi_alpha is not None
        if isinstance(state, torch.Tensor):
            # update q
            bsz = state.shape[0]
            ddp_num_samples = self.learnq_num
        else:
            # interact with env
            state = torch.tensor(
                state, dtype=torch.float, device=self.device, requires_grad=False
            ).unsqueeze(0)
            bsz = 1
            ddp_num_samples = self.ddp_num_samples

        assert len(state.shape) == 2
        if self.automatic_entropy_tuning:
            self.alpha = self.log_alpha.exp()

        pi_actions = [None] * self.ddph
        pi_states = [None] * self.ddph
        pi_not_dones = [None] * self.ddph
        pi_rewards = [None] * self.ddph

        # init
        with torch.no_grad():
            discount, init_values = 1, 0
            # state = torch.repeat_interleave(state, self.ddp_num_samples, dim=0)
            not_done = torch.as_tensor([True] * state.size(0)).to(self.device).reshape(-1, 1)

            for t in range(self.ddph):
                action, log_pi, mean = self.policy.sample(state)
                action = mean if evaluate else action
                if init_action is not None and t == 0:
                    init_action = torch.tensor(
                        init_action, dtype=torch.float, device=self.device, requires_grad=False
                    ).unsqueeze(0)
                    assert len(init_action.shape) == len(action.shape)
                    action = init_action
                log_pi = self.policy.log_prob(state, action)
                pi_states[t] = state
                pi_actions[t] = action
                pi_not_dones[t] = not_done

                next_state, reward = self.model_ensemble.step_tensor(
                    state,
                    action,
                    ensemble=True,
                    deterministic=evaluate,
                    use_penalty=evaluate and self.penalize_var,
                )
                pi_rewards[t] = reward
                done_batch = self.termination_fn.done(
                    state.cpu().numpy(),
                    action.cpu().numpy(),
                    next_state.cpu().numpy(),
                )

                if t == self.ddph - 1:
                    qf1, qf2 = self.critic(state, action)
                    min_qf = torch.min(qf1, qf2)
                    init_values += (
                        discount
                        * (
                            reward_logpi(
                                min_qf,
                                log_pi,
                                self.alpha,
                                z=self.policy.log_prob2(
                                    state,
                                    action,
                                    enable=enable_logpi and (t == 0 or self.logpi_each_step),
                                    coeff=self.logpi_alpha,
                                ),
                                reverse=evaluate and self.reverse_logpi,
                            )
                        )
                        * not_done.float()
                    )
                else:
                    init_values += (
                        discount
                        * (
                            reward_logpi(
                                reward,
                                log_pi,
                                self.alpha,
                                z=self.policy.log_prob2(
                                    state,
                                    action,
                                    enable=enable_logpi and (t == 0 or self.logpi_each_step),
                                    coeff=self.logpi_alpha,
                                ),
                                reverse=evaluate and self.reverse_logpi,
                            )
                        )
                        * not_done.float()
                    )

                state = next_state
                discount *= self.gamma
                done_batch = torch.as_tensor(done_batch).to(self.device).reshape(-1, 1)
                not_done = torch.logical_and(not_done, ~done_batch)

            for t in range(self.ddph):
                pi_actions[t] = torch.repeat_interleave(pi_actions[t], ddp_num_samples, dim=0)
                pi_states[t] = torch.repeat_interleave(pi_states[t], ddp_num_samples, dim=0)
                pi_not_dones[t] = torch.repeat_interleave(pi_not_dones[t], ddp_num_samples, dim=0)
                pi_rewards[t] = torch.repeat_interleave(pi_rewards[t], ddp_num_samples, dim=0)
            init_values = torch.repeat_interleave(init_values, ddp_num_samples, dim=0)

            start, end = math.log(self.ddp_min_delta), math.log(self.ddp_max_delta)
            delta = torch.linspace(
                start, end, ddp_num_samples, dtype=torch.float, device=self.device
            )
            delta = delta.exp()
            delta = delta.reshape(-1, 1).repeat(bsz, 1)
        ddp_iterations = (
            ddp_iters
            if ddp_iters is not None
            else (
                self.ddp_iterations // 2 if evaluate else np.random.randint(self.ddp_iterations + 1)
            )
        )
        for i in range(ddp_iterations):
            for a in pi_actions:
                a.requires_grad_(True)
            for s in pi_states:
                s.requires_grad_(True)

            openloopk = [None] * self.ddph
            feedbackK = [None] * self.ddph
            b_norms = [None] * self.ddph

            # backward
            ## backward last timestep
            current_s = pi_states[self.ddph - 1]
            current_a = pi_actions[self.ddph - 1]
            current_not_done = pi_not_dones[self.ddph - 1]
            i_drdx, i_drda = batch_jacobian_tupleinput(
                lambda x, y: reward_logpi(
                    0,
                    self.policy.log_prob(x, y) * current_not_done.float(),
                    self.alpha,
                    z=self.policy.log_prob2(
                        x,
                        y,
                        enable=enable_logpi and (self.ddph == 1 or self.logpi_each_step),
                        coeff=self.logpi_alpha,
                    ),
                    reverse=evaluate and self.reverse_logpi,
                ),
                (current_s, current_a),
            )  # bsz*num x 1 x S, bsz*num x 1 x A
            i_dqdx, i_dqda = batch_jacobian_tupleinput(
                lambda x, y: torch.min(*self.critic(x, y)) * current_not_done.float(),
                (current_s, current_a),
            )  # bsz*num x 1 x S, bsz*num x 1 x A

            with torch.no_grad():
                a = (i_drdx + i_dqdx).squeeze(1)  # bsz * num x S
                b = (i_drda + i_dqda).squeeze(1)  # bsz * num x A
                b_norm = torch.linalg.norm(b, dim=-1, keepdim=True)
                metrics.log_scalar("b_norm", b_norm.mean().item(), round=2)
                b_norm = torch.clamp(b_norm, min=1, max=self.norm_bound)
                bbt = torch.einsum("bi,bj->bij", b, b) + (
                    torch.eye(self.action_space.shape[0]) * 1e-3
                ).unsqueeze(0).to(b) * b_norm.unsqueeze(-1)
                bbtinvb = torch.linalg.solve(bbt, b.unsqueeze(-1)).squeeze(-1)  # bsz * num x A
                i_feedbackK = -torch.einsum("bi,bj->bij", bbtinvb, a)  # bsz * num x A x S
                openloopk[self.ddph - 1] = bbtinvb
                feedbackK[self.ddph - 1] = i_feedbackK
                b_norms[self.ddph - 1] = b_norm

            for i_horizon in range(self.ddph - 2, -1, -1):
                i_vxprime = (a + torch.einsum("ba,bah->bh", b, i_feedbackK)) * self.gamma
                current_s = pi_states[i_horizon]
                current_a = pi_actions[i_horizon]
                current_not_done = pi_not_dones[i_horizon]

                i_drdx, i_drda = batch_jacobian_tupleinput(
                    lambda x, y: (
                        reward_logpi(
                            self.model_ensemble.step_tensor(
                                x,
                                y,
                                ensemble=True,
                                deterministic=evaluate,
                                use_penalty=evaluate and self.penalize_var,
                            )[1],
                            self.policy.log_prob(x, y),
                            self.alpha,
                            z=self.policy.log_prob2(
                                x,
                                y,
                                enable=enable_logpi and (i_horizon == 0 or self.logpi_each_step),
                                coeff=self.logpi_alpha,
                            ),
                            reverse=evaluate and self.reverse_logpi,
                        )
                    )
                    * current_not_done.float(),
                    (current_s, current_a),
                )  # bsz*num x 1 x S, bsz*num x 1 x A
                i_dfdx, i_dfda = batch_jacobian_tupleinput(
                    lambda x, y: self.model_ensemble.step_tensor(
                        x, y, ensemble=True, deterministic=evaluate
                    )[0]
                    * current_not_done.float(),
                    (current_s, current_a),
                    fwd=False,
                )  # bsz * num x S x S,  bsz * num x S x A

                with torch.no_grad():
                    i_drdx, i_drda = i_drdx.squeeze(1), i_drda.squeeze(1)
                    a = i_drdx + torch.einsum("bi,bih->bh", i_vxprime, i_dfdx)  # bsz * num x S
                    b = i_drda + torch.einsum("bi,bia->ba", i_vxprime, i_dfda)  # bsz * num x A
                    b_norm = torch.linalg.norm(b, dim=-1, keepdim=True)
                    metrics.log_scalar("b_norm", b_norm.mean().item(), round=2)
                    b_norm = torch.clamp(b_norm, min=1, max=self.norm_bound)
                    bbt = torch.einsum("bi,bj->bij", b, b) + (
                        torch.eye(self.action_space.shape[0]) * 1e-3
                    ).unsqueeze(0).to(b) * b_norm.unsqueeze(-1)
                    bbtinvb = torch.linalg.solve(bbt, b.unsqueeze(-1)).squeeze(-1)  # bsz * num x A
                    i_feedbackK = -torch.einsum("bi,bj->bij", bbtinvb, a)  # bsz * num x A x S
                    openloopk[i_horizon] = bbtinvb
                    feedbackK[i_horizon] = i_feedbackK
                    b_norms[i_horizon] = b_norm

            # forward pass
            with torch.no_grad():
                delta_x = pi_states[0].new_zeros(pi_states[0].size())
                not_done = pi_not_dones[0]
                for i_horizon in range(self.ddph):
                    # delta_a = torch_clip(
                    #     delta * openloopk[i_horizon] * b_norms[i_horizon],
                    #     max_x=self.ddp_clipk * self.policy.action_scale,
                    #     min_x=-self.ddp_clipk * self.policy.action_scale,
                    # ) + torch_clip(
                    #     torch.einsum("bah,bh->ba", feedbackK[i_horizon], delta_x),
                    #     max_x=self.ddp_clipk * self.policy.action_scale,
                    #     min_x=-self.ddp_clipk * self.policy.action_scale,
                    # )
                    delta_a = torch_clip(
                        delta * openloopk[i_horizon],
                        max_x=self.ddp_clipk * self.policy.action_scale,
                        min_x=-self.ddp_clipk * self.policy.action_scale,
                    ) + torch_clip(
                        torch.einsum("bah,bh->ba", feedbackK[i_horizon], delta_x),
                        max_x=self.ddp_clipk * self.policy.action_scale,
                        min_x=-self.ddp_clipk * self.policy.action_scale,
                    )
                    pi_actions[i_horizon] = self.policy.clip(pi_actions[i_horizon] + delta_a)
                    next_x, reward = self.model_ensemble.step_tensor(
                        pi_states[i_horizon],
                        pi_actions[i_horizon],
                        ensemble=True,
                        deterministic=evaluate,
                        use_penalty=evaluate and self.penalize_var,
                    )
                    pi_rewards[i_horizon] = reward
                    if i_horizon < self.ddph - 1:
                        delta_x = next_x - pi_states[i_horizon + 1]
                        pi_states[i_horizon + 1] = next_x

                        done_batch = self.termination_fn.done(
                            pi_states[i_horizon].cpu().numpy(),
                            pi_actions[i_horizon].cpu().numpy(),
                            pi_states[i_horizon + 1].cpu().numpy(),
                        )
                        done_batch = torch.as_tensor(done_batch).to(self.device).reshape(-1, 1)
                        not_done = torch.logical_and(not_done, ~done_batch)
                        pi_not_dones[i_horizon + 1] = not_done

                    if self.ddp_delta_decay_legacy:
                        delta = delta * self.ddp_delta_decay
                        delta = torch.clip(delta, min=self.ddp_min_delta)
                if not self.ddp_delta_decay_legacy:
                    delta = delta * self.ddp_delta_decay
                    delta = torch.clip(delta, min=self.ddp_min_delta)

        # evaluate
        with torch.no_grad():
            discount, last_values = 1, 0
            for time_step in range(self.ddph):
                log_prob = self.policy.log_prob(pi_states[time_step], pi_actions[time_step])
                if time_step == self.ddph - 1:
                    qf1, qf2 = self.critic(pi_states[time_step], pi_actions[time_step])
                    min_qf = torch.min(qf1, qf2)
                    last_values += (
                        discount
                        * (
                            reward_logpi(
                                min_qf,
                                log_prob,
                                self.alpha,
                                z=self.policy.log_prob2(
                                    pi_states[time_step],
                                    pi_actions[time_step],
                                    enable=enable_logpi
                                    and (time_step == 0 or self.logpi_each_step),
                                    coeff=self.logpi_alpha,
                                ),
                                reverse=self.reverse_logpi and evaluate,
                            )
                        )
                        * pi_not_dones[time_step].float()
                    )
                else:
                    last_values += (
                        discount
                        * (
                            reward_logpi(
                                pi_rewards[time_step],
                                log_prob,
                                self.alpha,
                                z=self.policy.log_prob2(
                                    pi_states[time_step],
                                    pi_actions[time_step],
                                    enable=enable_logpi
                                    and (time_step == 0 or self.logpi_each_step),
                                    coeff=self.logpi_alpha,
                                ),
                                reverse=self.reverse_logpi and evaluate,
                            )
                        )
                        * pi_not_dones[time_step].float()
                    )
                discount *= self.gamma
            value_improved = (last_values - init_values).reshape(bsz, -1)
            metrics.log_scalar("Q_mean", init_values.mean().item(), round=2)
            metrics.log_scalar("improved_mean", value_improved.mean().item(), round=2)
            value_improved_backup = value_improved

            if evaluate:
                argmax = torch.argmax(value_improved, dim=1)
            else:
                value_improved = value_improved - value_improved.max(dim=1, keepdim=True)[0]
                prob = F.softmax(value_improved, dim=-1).cumsum(dim=1)
                random = prob.new_zeros(bsz, 1).uniform_()
                argmax = torch.argmax((random <= prob).float(), dim=1)

            # values_delta_clip = torch.clamp(value_improved, min=1e-6)
            # values_delta_prob = values_delta_clip / values_delta_clip.sum(dim=-1, keepdim=True)
            # argmax = []
            # for idx in range(bsz):
            #     argmax.append(
            #         np.random.choice(self.ddp_num_samples, p=values_delta_prob[idx].cpu().numpy())
            #     )
            # argmax = torch.as_tensor(argmax).to(self.device).long()
            # argmax = torch.argmax(values, dim=-1)
            metrics.log_scalar(
                "improved_selected",
                torch.gather(value_improved_backup, 1, argmax.unsqueeze(-1)).mean().item(),
                round=2,
            )
            top_idx = (
                argmax.unsqueeze(dim=-1)
                .unsqueeze(dim=-1)
                .expand(-1, -1, self.action_space.shape[0])
            )  # (bsz, 1, 1)
            actions = pi_actions[0].reshape(bsz, -1, self.action_space.shape[0])
            elite_actions = torch.gather(
                actions,
                1,
                top_idx,
            ).squeeze(1)
        if bsz == 1:
            return elite_actions.squeeze(0)
        else:
            return elite_actions

    def select_action_gbp(self, state, evaluate=False, ddp_iters=None):
        enable_logpi = evaluate and self.logpi_alpha is not None
        if isinstance(state, torch.Tensor):
            bsz = state.shape[0]
            ddp_num_samples = self.learnq_num
        else:
            state = torch.tensor(
                state, dtype=torch.float, device=self.device, requires_grad=False
            ).unsqueeze(0)
            bsz = 1
            ddp_num_samples = self.ddp_num_samples

        assert len(state.shape) == 2
        if self.automatic_entropy_tuning:
            self.alpha = self.log_alpha.exp()

        pi_actions = [None] * self.ddph
        pi_states = [None] * self.ddph
        pi_not_dones = [None] * self.ddph
        pi_rewards = [None] * self.ddph

        # init
        with torch.no_grad():
            discount, init_values = 1, 0
            not_done = torch.as_tensor([True] * state.size(0)).to(self.device).reshape(-1, 1)

            for t in range(self.ddph):
                action, log_pi, mean = self.policy.sample(state)
                action = mean if evaluate else action
                log_pi = self.policy.log_prob(state, action)
                pi_states[t] = state
                pi_actions[t] = action
                pi_not_dones[t] = not_done

                next_state, reward = self.model_ensemble.step_tensor(
                    state,
                    action,
                    ensemble=True,
                    deterministic=evaluate,
                    use_penalty=evaluate and self.penalize_var,
                )

                pi_rewards[t] = reward
                done_batch = self.termination_fn.done(
                    state.cpu().numpy(),
                    action.cpu().numpy(),
                    next_state.cpu().numpy(),
                )

                if t == self.ddph - 1:
                    qf1, qf2 = self.critic(state, action)
                    min_qf = torch.min(qf1, qf2)
                    init_values += (
                        discount
                        * (
                            reward_logpi(
                                min_qf,
                                log_pi,
                                self.alpha,
                                z=self.policy.log_prob2(
                                    state,
                                    action,
                                    enable=enable_logpi and (t == 0 or self.logpi_each_step),
                                    coeff=self.logpi_alpha,
                                ),
                                reverse=evaluate and self.reverse_logpi,
                            )
                        )
                        * not_done.float()
                    )
                else:
                    init_values += (
                        discount
                        * (
                            reward_logpi(
                                reward,
                                log_pi,
                                self.alpha,
                                z=self.policy.log_prob2(
                                    state,
                                    action,
                                    enable=enable_logpi and (t == 0 or self.logpi_each_step),
                                    coeff=self.logpi_alpha,
                                ),
                                reverse=evaluate and self.reverse_logpi,
                            )
                        )
                        * not_done.float()
                    )

                state = next_state
                discount *= self.gamma
                done_batch = torch.as_tensor(done_batch).to(self.device).reshape(-1, 1)
                not_done = torch.logical_and(not_done, ~done_batch)

            for t in range(self.ddph):
                pi_actions[t] = torch.repeat_interleave(pi_actions[t], ddp_num_samples, dim=0)
                pi_states[t] = torch.repeat_interleave(pi_states[t], ddp_num_samples, dim=0)
                pi_not_dones[t] = torch.repeat_interleave(pi_not_dones[t], ddp_num_samples, dim=0)
                pi_rewards[t] = torch.repeat_interleave(pi_rewards[t], ddp_num_samples, dim=0)
            init_values = torch.repeat_interleave(init_values, ddp_num_samples, dim=0)
            start, end = (
                math.log(self.ddp_min_delta / self.ddp_max_delta * 0.1),
                math.log(self.ddp_max_delta / self.ddp_max_delta * 0.1),
            )
            delta = torch.linspace(
                start, end, ddp_num_samples, dtype=torch.float, device=self.device
            )
            delta = delta.exp()
            delta = delta.reshape(-1, 1).repeat(bsz, 1)

        ddp_iterations = (
            ddp_iters
            if ddp_iters is not None
            else (
                self.ddp_iterations // 2 if evaluate else np.random.randint(self.ddp_iterations + 1)
            )
        )

        for i in range(ddp_iterations):
            for a in pi_actions:
                a.requires_grad_(True)

            def value_func(*pi_actions_list):
                ddiscount, vvalue = 1, 0
                sstate = pi_states[0]

                for t in range(self.ddph):
                    aaction = pi_actions_list[t]
                    llog_pi = self.policy.log_prob(sstate, aaction)
                    nnext_state, rrward = self.model_ensemble.step_tensor(
                        sstate,
                        aaction,
                        ensemble=True,
                        deterministic=evaluate,
                        use_penalty=evaluate and self.penalize_var,
                    )
                    nnot_done = pi_not_dones[t]
                    if t == self.ddph - 1:
                        qqf1, qqf2 = self.critic(sstate, aaction)
                        mmin_qf = torch.min(qqf1, qqf2)
                        vvalue += (
                            ddiscount
                            * (
                                reward_logpi(
                                    mmin_qf,
                                    llog_pi,
                                    self.alpha,
                                    z=self.policy.log_prob2(
                                        sstate,
                                        aaction,
                                        enable=enable_logpi and (t == 0 or self.logpi_each_step),
                                        coeff=self.logpi_alpha,
                                    ),
                                    reverse=evaluate and self.reverse_logpi,
                                )
                            )
                            * nnot_done.float()
                        )
                    else:
                        vvalue += (
                            ddiscount
                            * (
                                reward_logpi(
                                    rrward,
                                    llog_pi,
                                    self.alpha,
                                    z=self.policy.log_prob2(
                                        sstate,
                                        aaction,
                                        enable=enable_logpi and (t == 0 or self.logpi_each_step),
                                        coeff=self.logpi_alpha,
                                    ),
                                    reverse=evaluate and self.reverse_logpi,
                                )
                            )
                            * nnot_done.float()
                        )

                    sstate = nnext_state
                    ddiscount *= self.gamma
                return vvalue

            delta_actions = batch_jacobian_multitupleinput(value_func, pi_actions, argnum=self.ddph)
            delta_actions = [x.squeeze(1) for x in delta_actions]

            # forward
            with torch.no_grad():
                state = pi_states[0]
                not_done = torch.as_tensor([True] * state.size(0)).to(self.device).reshape(-1, 1)
                for i_horizon in range(self.ddph):
                    delta_a = torch_clip(
                        delta * delta_actions[i_horizon],
                        max_x=self.ddp_clipk * self.policy.action_scale,
                        min_x=self.ddp_clipk * self.policy.action_scale,
                    )
                    pi_actions[i_horizon] = self.policy.clip(pi_actions[i_horizon] + delta_a)
                    next_x, reward = self.model_ensemble.step_tensor(
                        pi_states[i_horizon],
                        pi_actions[i_horizon],
                        ensemble=True,
                        deterministic=evaluate,
                        use_penalty=evaluate and self.penalize_var,
                    )
                    pi_rewards[i_horizon] = reward
                    if i_horizon < self.ddph - 1:
                        pi_states[i_horizon + 1] = next_x
                        done_batch = self.termination_fn.done(
                            pi_states[i_horizon].cpu().numpy(),
                            pi_actions[i_horizon].cpu().numpy(),
                            pi_states[i_horizon + 1].cpu().numpy(),
                        )
                        done_batch = torch.as_tensor(done_batch).to(self.device).reshape(-1, 1)
                        not_done = torch.logical_and(not_done, ~done_batch)
                        pi_not_dones[i_horizon + 1] = not_done

                delta = delta * self.ddp_delta_decay
                delta = torch.clip(delta, min=self.ddp_min_delta / self.ddp_max_delta * 0.1)

        # evaluate
        with torch.no_grad():
            discount, last_values = 1, 0
            for time_step in range(self.ddph):
                log_prob = self.policy.log_prob(pi_states[time_step], pi_actions[time_step])
                if time_step == self.ddph - 1:
                    qf1, qf2 = self.critic(pi_states[time_step], pi_actions[time_step])
                    min_qf = torch.min(qf1, qf2)
                    last_values += (
                        discount
                        * (
                            reward_logpi(
                                min_qf,
                                log_prob,
                                self.alpha,
                                z=self.policy.log_prob2(
                                    pi_states[time_step],
                                    pi_actions[time_step],
                                    enable=enable_logpi
                                    and (time_step == 0 or self.logpi_each_step),
                                    coeff=self.logpi_alpha,
                                ),
                                reverse=self.reverse_logpi and evaluate,
                            )
                        )
                        * pi_not_dones[time_step].float()
                    )
                else:
                    last_values += (
                        discount
                        * (
                            reward_logpi(
                                pi_rewards[time_step],
                                log_prob,
                                self.alpha,
                                z=self.policy.log_prob2(
                                    pi_states[time_step],
                                    pi_actions[time_step],
                                    enable=enable_logpi
                                    and (time_step == 0 or self.logpi_each_step),
                                    coeff=self.logpi_alpha,
                                ),
                                reverse=self.reverse_logpi and evaluate,
                            )
                        )
                        * pi_not_dones[time_step].float()
                    )
                discount *= self.gamma

            value_improved = (last_values - init_values).reshape(bsz, -1)
            metrics.log_scalar("Q_mean", init_values.mean().item(), round=2)
            metrics.log_scalar("improved_mean", value_improved.mean().item(), round=2)
            value_improved_backup = value_improved

            if evaluate:
                argmax = torch.argmax(value_improved, dim=1)
            else:
                value_improved = value_improved - value_improved.max(dim=1, keepdim=True)[0]
                prob = F.softmax(value_improved, dim=-1).cumsum(dim=1)
                random = prob.new_zeros(bsz, 1).uniform_()
                argmax = torch.argmax((random <= prob).float(), dim=1)

            metrics.log_scalar(
                "improved_selected",
                torch.gather(value_improved_backup, 1, argmax.unsqueeze(-1)).mean().item(),
                round=2,
            )
            top_idx = (
                argmax.unsqueeze(dim=-1)
                .unsqueeze(dim=-1)
                .expand(-1, -1, self.action_space.shape[0])
            )  # (bsz, 1, 1)
            actions = pi_actions[0].reshape(bsz, -1, self.action_space.shape[0])
            elite_actions = torch.gather(
                actions,
                1,
                top_idx,
            ).squeeze(1)

        if bsz == 1:
            return elite_actions.squeeze(0)
        else:
            return elite_actions

    #### not need ####

    def update_parameters_like_sac(self, memory, batch_size, updates):
        # Sample a batch from memory   update Q network
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch, _ = memory.sample(
            batch_size=batch_size
        )

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            # print(next_state_action)
            qf1_next_target, qf2_next_target = self.critic_target(
                next_state_batch, next_state_action
            )
            # min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) ####- self.alpha * next_state_log_pi
            min_qf_next_target = (
                torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            )
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
        qf1, qf2 = self.critic(
            state_batch, action_batch
        )  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(
            qf1, next_q_value
        )  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(
            qf2, next_q_value
        )  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        pi, log_pi, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = (
            (self.alpha * log_pi) - min_qf_pi
        ).mean()  # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone()  # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.0).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha)  # For TensorboardX logs

        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return (
            qf1_loss.item(),
            qf2_loss.item(),
            policy_loss.item(),
            alpha_loss.item(),
            alpha_tlogs.item(),
        )

    def update_parameters_q(
        self,
        memory,
        memory_fake,
        batch_size,
        updates,
        use_decay=False,
        weight_decay=0.1,
        real_ratio=0.05,
    ):
        # Sample a batch from memory   update Q network
        batch_real = int(
            batch_size * real_ratio
        )  # 0.05 for InvertedPendulum-v2   try 0.1 for others
        if batch_size - batch_real > 0 and len(memory_fake) > 0:
            (
                state_batch_real,
                action_batch_real,
                reward_batch_real,
                next_state_batch_real,
                mask_batch_real,
                _,
            ) = memory.sample(batch_size=batch_real)
            (
                state_batch_fake,
                action_batch_fake,
                reward_batch_fake,
                next_state_batch_fake,
                mask_batch_fake,
                _,
            ) = memory_fake.sample(batch_size=batch_size - batch_real)
            state_batch = np.concatenate((state_batch_real, state_batch_fake), axis=0)
            action_batch = np.concatenate((action_batch_real, action_batch_fake), axis=0)
            reward_batch = np.concatenate((reward_batch_real, reward_batch_fake), axis=0)
            next_state_batch = np.concatenate(
                (next_state_batch_real, next_state_batch_fake), axis=0
            )
            mask_batch = np.concatenate((mask_batch_real, mask_batch_fake), axis=0)

            state_batch = torch.FloatTensor(state_batch).to(self.device)
            next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
            action_batch = torch.FloatTensor(action_batch).to(self.device)
            reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
            mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)
        else:
            # Sample a batch from memory   update Q network
            (
                state_batch,
                action_batch,
                reward_batch,
                next_state_batch,
                mask_batch,
                _,
            ) = memory.sample(batch_size=batch_size)

            state_batch = torch.FloatTensor(state_batch).to(self.device)
            next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
            action_batch = torch.FloatTensor(action_batch).to(self.device)
            reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
            mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_state_action, next_state_log_pi = self.select_next_action(next_state_batch)
            # print(next_state_action)
            qf1_next_target, qf2_next_target = self.critic_target(
                next_state_batch, next_state_action
            )
            # min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) ####- self.alpha * next_state_log_pi
            min_qf_next_target = (
                torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            )
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
        qf1, qf2 = self.critic(
            state_batch, action_batch
        )  # Two Q-functions to mitigate positive bias in the policy improvement step
        metrics.log_scalar("q_value", ((qf1 + qf2) / 2).mean().item())
        qf1_loss = F.mse_loss(
            qf1, next_q_value
        )  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(
            qf2, next_q_value
        )  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        if use_decay:
            decay_loss = 0
            for name, value in self.critic.named_parameters():
                decay_loss += weight_decay * torch.sum(torch.square(value)) / 2.0
            qf_loss += decay_loss
        self.critic_optim.zero_grad()
        qf_loss.backward()
        torch.nn.utils.clip_grad_value_(self.critic.parameters(), clip_value=100.0)
        self.critic_optim.step()

        pi, log_pi, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = (
            (self.alpha * log_pi) - min_qf_pi
        ).mean()  # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        # self.policy_optim.zero_grad()
        # policy_loss.backward()
        # self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone()  # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.0).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha)  # For TensorboardX logs

        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        dd = 0
        for name, value in self.critic.named_parameters():
            dd += torch.norm(value).item()

        ee = torch.norm(next_q_value / batch_size).item()

        ff = 0
        for name, value in self.policy.named_parameters():
            ff += torch.norm(value).item()
        return (
            qf1_loss.item(),
            qf2_loss.item(),
            policy_loss.item(),
            alpha_loss.item(),
            alpha_tlogs.item(),
            dd,
            ee,
            ff,
            torch.sum(next_state_log_pi).item() / batch_size,
        )

    #### need ####
    # def rollout_for_update_q(self, memory, memory_fake, rollout_length=1, rollout_batch_size=100000):
    #     with torch.no_grad():
    #         state_batch, _, _, _, _, _ = memory.sample_all_batch(rollout_batch_size)
    #         state_batch = torch.from_numpy(state_batch).float().to(self.device).reshape(rollout_batch_size,self.DIM_X)
    #         not_done_list = [True]*state_batch.size(0)

    #         for i in range(rollout_length):
    #             # action_batch = self.select_action(state_batch, evaluate=False)
    #             action_batch, _, _ = self.policy.sample(state_batch)
    #             next_state_batch, reward_batch = self.model_ensemble.step_tensor(state_batch, action_batch)
    #             next_state_batch = next_state_batch.detach()
    #             # done_batch = [False]*state_batch.size(0)
    #             done_batch = self.termination_fn.done(state_batch.detach().to('cpu').numpy(), action_batch.detach().to('cpu').numpy(), next_state_batch.detach().to('cpu').numpy())

    #             s_np = state_batch.detach().to('cpu').numpy()[not_done_list]
    #             act_np = action_batch.detach().to('cpu').numpy()[not_done_list]
    #             ns_np = next_state_batch.detach().to('cpu').numpy()[not_done_list]
    #             re_np = reward_batch.detach().to('cpu').numpy()[not_done_list]
    #             memory_fake.push_batch([(s_np[j], act_np[j], re_np[j][0], ns_np[j], float(not done_batch[j]), done_batch[j]) for j in range(s_np.shape[0])])

    #             not_done_list = [not_done_list[k] and (not done_batch[k]) for k in range(state_batch.size(0))]
    #             # if sum(not_done_list)==0 :
    #             #     print('done rollout early')
    #             #     break
    #             state_batch = next_state_batch.clone()
    #     return memory_fake

    def update_parameters_policy_direct(self, memory: ReplayMemory, H=10, batch_size=256):
        state_batch, _, _, _, _, _ = memory.sample(batch_size)
        state_batch = (
            torch.from_numpy(state_batch).float().to(self.device).reshape(batch_size, self.DIM_X)
        )
        discount, L = 1, 0
        not_done = torch.as_tensor([True] * state_batch.size(0)).to(self.device).reshape(-1, 1)
        if self.automatic_entropy_tuning:
            self.alpha = self.log_alpha.exp()
        for time_step in range(H):
            action_batch, log_pi_batch, _ = self.policy.sample(state_batch)
            next_state_batch, reward_batch = self.model_ensemble.step_tensor(
                state_batch, action_batch
            )
            done_batch = self.termination_fn.done(
                state_batch.detach().to("cpu").numpy(),
                action_batch.detach().to("cpu").numpy(),
                next_state_batch.detach().to("cpu").numpy(),
            )
            if time_step == H - 1:
                qf1, qf2 = self.critic(state_batch, action_batch)
                min_qf = torch.min(qf1, qf2)
                L -= discount * (min_qf - self.alpha * log_pi_batch) * not_done.float()
            else:
                L -= discount * (reward_batch - self.alpha * log_pi_batch) * not_done.float()

            # update for next step
            state_batch = next_state_batch
            discount *= self.gamma
            done_batch = torch.as_tensor(done_batch).to(self.device).reshape(-1, 1)
            not_done = torch.logical_and(not_done, ~done_batch)

        # update the policy param
        self.policy_optim.zero_grad()
        loss = torch.mean(L)
        loss.backward()
        if self.clip_gn is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.clip_gn)
            metrics.log_scalar("grad_norm", grad_norm.item())
        else:
            torch.nn.utils.clip_grad_value_(self.policy.parameters(), clip_value=5.0)
        self.policy_optim.step()
        return loss.item()

    def update_parameters_policy(self, state, memory, memory_fake, H=10, batch_size=256):
        if self.policy_direct_bp:
            return self.update_parameters_policy_direct(memory, H, batch_size), 0

        state_batch, _, _, _, _, _ = memory.sample(batch_size)
        state_batch = (
            torch.from_numpy(state_batch)
            .float()
            .to(self.device)
            .reshape(batch_size, self.DIM_X)
            .requires_grad_()
        )
        state_seq = []
        action_seq = []
        log_pi_seq = []
        done_seq = []
        p_seq = []
        H_seq = []
        not_done_list = [True] * state_batch.size(0)

        for i in range(H):
            # action_batch = self.select_action(state_batch, evaluate=False)
            action_batch, log_pi_batch, _ = self.policy.sample(state_batch)
            next_state_batch, reward_batch = self.model_ensemble.step_tensor(
                state_batch, action_batch
            )
            next_state_batch = next_state_batch.detach()
            # done_batch = [False]*state_batch.size(0)
            done_batch = self.termination_fn.done(
                state_batch.detach().to("cpu").numpy(),
                action_batch.detach().to("cpu").numpy(),
                next_state_batch.detach().to("cpu").numpy(),
            )
            state_seq.append(state_batch)
            action_seq.append(action_batch)
            log_pi_seq.append(log_pi_batch)
            done_seq.append(done_batch)
            # if i==0 or i==1:
            # if True:
            # if i==0: # halfcheetah
            # s_np = state_batch.detach().to('cpu').numpy()[not_done_list]
            # act_np = action_batch.detach().to('cpu').numpy()[not_done_list]
            # ns_np = next_state_batch.detach().to('cpu').numpy()[not_done_list]
            # re_np = reward_batch.detach().to('cpu').numpy()[not_done_list]
            # memory_fake.push_batch([(s_np[j], act_np[j], re_np[j][0], ns_np[j], float(not done_batch[j]), done_batch[j]) for j in range(s_np.shape[0])])

            not_done_list = [
                not_done_list[k] and (not done_batch[k]) for k in range(state_batch.size(0))
            ]
            # if sum(not_done_list)==0 :
            #     print('done rollout early')
            #     break
            state_batch = next_state_batch.clone().requires_grad_()

        max_length_sequence = H
        mask_batch_np = np.zeros([batch_size, max_length_sequence, 1])  #### 1 if not done
        for i in range(batch_size):
            for j in range(H):
                mask_batch_np[i, j, 0] = 1
                if done_seq[j][i]:
                    break
        mask_batch = torch.from_numpy(mask_batch_np).to(self.device).reshape(batch_size, -1, 1)

        # length_sequence_batch = np.zeros([batch_size,1])
        # for i in range(batch_size):
        #     length_sequence_batch[i,0] = len(self.state_sequence[i])
        # length_sequence_batch_t = torch.from_numpy(length_sequence_batch).float().to(self.device)

        # update pt
        state_input = state_seq[-1]
        action_input = action_seq[-1]
        log_pi = log_pi_seq[-1]

        qf1, qf2 = self.critic(state_input, action_input)
        min_qf = torch.min(qf1, qf2)
        if self.automatic_entropy_tuning:
            self.alpha = self.log_alpha.exp()
        L = (
            (-min_qf + self.alpha * log_pi).view(batch_size, 1)
            * (self.gamma ** (torch.sum(mask_batch, 1) - 1)).view(batch_size, 1)
            * mask_batch[:, -1, 0].view(batch_size, 1)
        )
        H = L
        g = (
            torch.autograd.grad(torch.sum(L), state_input, retain_graph=True)[0]
            .detach()
            .view(batch_size, self.DIM_X)
        )

        cc = (torch.sum(g**2) / batch_size).item()

        p_seq.append(g.clone())
        H_seq.append(H)

        for j in range(max_length_sequence - 2, -1, -1):
            p_tmp = g.clone()
            state_input = state_seq[j]
            action_input = action_seq[j]
            log_pi = log_pi_seq[j]
            next_state, L = self.model_ensemble.step_tensor(state_input, action_input)
            H = (
                (torch.sum(p_tmp * next_state, 1).view(batch_size, 1) - L + self.alpha * log_pi)
                * (
                    self.gamma
                    ** (
                        torch.sum(
                            mask_batch[
                                :,
                                0 : j + 1,
                            ],
                            1,
                        )
                        - 1
                    )
                ).view(batch_size, 1)
                * mask_batch[:, j, 0].view(batch_size, 1)
            )
            # print(H,L)
            H_seq.append(H)
            g = (
                torch.autograd.grad(torch.sum(H), state_input, retain_graph=True)[0]
                .detach()
                .view(batch_size, self.DIM_X)
            )
            p_seq.append(g.clone())
        p_seq = p_seq.reverse()

        self.policy_optim.zero_grad()
        loss = torch.sum(torch.stack(H_seq, 0)) / batch_size
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy.parameters(), clip_value=5.0)
        self.policy_optim.step()

        return loss.item(), cc

    def update_parameters_ensemble_model(self, memory, batch_size, weight_grad, near_n):
        self.model_ensemble.update(memory, batch_size, weight_grad, near_n)

    def save_model(self, env_name, suffix="", model_path=None, policy_path=None, critic_path=None):
        if not os.path.exists("models/"):
            os.makedirs("models/")

        if model_path is None:
            model_path = "models/pmp_model_{}_{}".format(env_name, suffix)
        if policy_path is None:
            policy_path = "models/pmp_policy_{}_{}".format(env_name, suffix)
        if critic_path is None:
            critic_path = "models/pmp_critic_{}_{}".format(env_name, suffix)
        # print('Saving models to {} and {}'.format(model_path, policy_path))
        torch.save(self.policy.state_dict(), policy_path)
        torch.save(self.critic.state_dict(), critic_path)
        torch.save(self.model_ensemble.model.ensemble_model.state_dict(), model_path)

    def save_checkpoint(self, savedir, step):
        fn = os.path.join(savedir, f"checkpoint_{step}.pt")
        checkpoint = {
            "args": self.args,
            "model": self.model_ensemble.model.ensemble_model.state_dict(),
            "model_optimizer": self.model_ensemble.model.ensemble_model.optimizer.state_dict(),
            "model_lrscheduler": self.model_ensemble.model.ensemble_model.lr_scheduler.state_dict(),
            "model_scaler": self.model_ensemble.model.scaler,
            "model_elite_model_idxes": self.model_ensemble.model.elite_model_idxes,
            "log_alpha": self.log_alpha,
            "alpha_optimizer": self.alpha_optim.state_dict(),
            "alpha_lrscheduler": self.alpha_lrscheduler.state_dict(),
            "policy": self.policy.state_dict(),
            "policy_optimizer": self.policy_optim.state_dict(),
            "policy_lrscheduler": self.policy_lrscheduler.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "critic_optimizer": self.critic_optim.state_dict(),
            "critic_lrscheduler": self.critic_lrscheduler.state_dict(),
        }
        torch.save(checkpoint, fn)

    def load_checkpoint(self, checkpoint):
        if isinstance(checkpoint, str):
            checkpoint = torch.load(checkpoint)
        self.model_ensemble.model.ensemble_model.load_state_dict(checkpoint["model"]),
        self.model_ensemble.model.ensemble_model.optimizer.load_state_dict(
            checkpoint["model_optimizer"]
        )
        self.model_ensemble.model.ensemble_model.lr_scheduler.load_state_dict(
            checkpoint["model_lrscheduler"]
        ),
        self.model_ensemble.model.scaler = checkpoint["model_scaler"]
        self.model_ensemble.model.elite_model_idxes = checkpoint["model_elite_model_idxes"]
        # self.log_alpha.data = checkpoint["log_alpha"] # to check optim
        self.log_alpha.data.copy_(checkpoint["log_alpha"])
        self.alpha = self.log_alpha.exp()
        self.alpha_optim.load_state_dict(checkpoint["alpha_optimizer"])
        self.alpha_lrscheduler.load_state_dict(checkpoint["alpha_lrscheduler"])
        self.policy.load_state_dict(checkpoint["policy"])
        self.policy_optim.load_state_dict(checkpoint["policy_optimizer"])
        self.policy_lrscheduler.load_state_dict(checkpoint["policy_lrscheduler"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.critic_target.load_state_dict(checkpoint["critic_target"])
        self.critic_optim.load_state_dict(checkpoint["critic_optimizer"])
        self.critic_lrscheduler.load_state_dict(checkpoint["critic_lrscheduler"])

    def load_model(self, modle_path, policy_path, critic_path):
        logger.info(
            "Loading models from {} and {} and {}".format(modle_path, policy_path, critic_path)
        )
        if modle_path is not None:
            self.model_ensemble.model.ensemble_model.load_state_dict(torch.load(modle_path))
        if policy_path is not None:
            self.policy.load_state_dict(torch.load(policy_path))
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path))
