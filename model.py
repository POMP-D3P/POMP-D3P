import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from utility import get_lrschedule

torch.set_default_tensor_type(torch.cuda.FloatTensor)

from typing import Tuple
import math
import numpy as np
import matplotlib.pyplot as plt
import gzip
import itertools
import logging

logger = logging.getLogger(__name__)
# import copy

device = torch.device("cuda")


class StandardScaler(object):
    def __init__(self):
        pass

    def fit(self, data):
        """Runs two ops, one for assigning the mean of the data to the internal mean, and
        another for assigning the standard deviation of the data to the internal standard deviation.
        This function must be called within a 'with <session>.as_default()' block.
        Arguments:
        data (np.ndarray): A numpy array containing the input
        Returns: None.
        """
        self.mu = np.mean(data, axis=0, keepdims=True)
        self.std = np.std(data, axis=0, keepdims=True)
        self.std[self.std < 1e-12] = 1.0

        self.mu_tensor = torch.from_numpy(self.mu).float().to("cuda")
        self.std_tensor = torch.from_numpy(self.std).float().to("cuda")

    def transform(self, data):
        """Transforms the input matrix data using the parameters of this scaler.
        Arguments:
        data (np.array): A numpy array containing the points to be transformed.
        Returns: (np.array) The transformed dataset.
        """
        return (data - self.mu) / self.std

    def inverse_transform(self, data):
        """Undoes the transformation performed by this scaler.
        Arguments:
        data (np.array): A numpy array containing the points to be transformed.
        Returns: (np.array) The transformed dataset.
        """
        return self.std * data + self.mu

    def transform_tensor(self, data):

        return (data - self.mu_tensor) / self.std_tensor


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        x = x * torch.sigmoid(x)
        return x


def init_weights(m):
    def truncated_normal_init(t, mean=0.0, std=0.01):
        torch.nn.init.normal_(t, mean=mean, std=std)
        while True:
            cond = torch.logical_or(t < mean - 2 * std, t > mean + 2 * std)
            if not torch.sum(cond):
                break
            t = torch.where(cond, torch.nn.init.normal_(torch.ones(t.shape), mean=mean, std=std), t)
        return t

    if type(m) == nn.Linear or isinstance(m, EnsembleFC):
        input_dim = m.in_features
        truncated_normal_init(m.weight, std=1 / (2 * np.sqrt(input_dim)))
        m.bias.data.fill_(0.0)


class EnsembleFC(nn.Module):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    ensemble_size: int
    weight: torch.Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        ensemble_size: int,
        weight_decay: float = 0.0,
        bias: bool = True,
    ) -> None:
        super(EnsembleFC, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size
        self.weight = nn.Parameter(torch.Tensor(ensemble_size, in_features, out_features))
        self.weight_decay = weight_decay
        if bias:
            self.bias = nn.Parameter(torch.Tensor(ensemble_size, out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        pass

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        w_times_x = torch.bmm(input, self.weight)
        return torch.add(w_times_x, self.bias[:, None, :])  # w times x + b

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )


class EnsembleModel(nn.Module):
    # ensemble nn
    def __init__(
        self,
        state_size,
        action_size,
        reward_size,
        ensemble_size,
        hidden_size=200,
        learning_rate=1e-3,
        use_decay=False,
        args=None,
    ):
        super(EnsembleModel, self).__init__()
        self.hidden_size = hidden_size
        self.nn1 = EnsembleFC(
            state_size + action_size, hidden_size, ensemble_size, weight_decay=0.000025
        )
        self.nn2 = EnsembleFC(hidden_size, hidden_size, ensemble_size, weight_decay=0.00005)
        self.nn3 = EnsembleFC(hidden_size, hidden_size, ensemble_size, weight_decay=0.000075)
        self.nn4 = EnsembleFC(hidden_size, hidden_size, ensemble_size, weight_decay=0.000075)
        self.use_decay = use_decay

        self.output_dim = state_size + reward_size
        self.reward_size = reward_size

        # # Add variance output
        self.nn5 = EnsembleFC(hidden_size, self.output_dim * 2, ensemble_size, weight_decay=0.0001)

        self.max_logvar = nn.Parameter(
            (torch.ones((1, self.output_dim)).float() / 2).to(device), requires_grad=False
        )
        self.min_logvar = nn.Parameter(
            (-torch.ones((1, self.output_dim)).float() * 10).to(device), requires_grad=False
        )

        # ####
        # self.output_dim_s = state_size
        # self.output_dim_r = reward_size
        # ####

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.lr_scheduler = get_lrschedule(args, self.optimizer)
        self.apply(init_weights)
        self.swish = Swish()

    def forward(self, x, mode="rs", ret_log_var=False):
        nn1_output = self.swish(self.nn1(x))
        nn2_output = self.swish(self.nn2(nn1_output))
        nn3_output = self.swish(self.nn3(nn2_output))
        nn4_output = self.swish(self.nn4(nn3_output))
        nn5_output = self.nn5(nn4_output)

        mean = nn5_output[:, :, : self.output_dim]

        logvar = self.max_logvar - F.softplus(self.max_logvar - nn5_output[:, :, self.output_dim :])
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)

        if mode == "rs":
            if ret_log_var:
                return mean, logvar
            else:
                return mean, torch.exp(logvar)
        elif mode == "s":
            if ret_log_var:
                return mean[:, :, self.reward_size :], logvar[:, :, self.reward_size :]
            else:
                return mean[:, :, self.reward_size :], torch.exp(logvar[:, :, self.reward_size :])
        elif mode == "r":
            if ret_log_var:
                return mean[:, :, : self.reward_size], logvar[:, :, : self.reward_size]
            else:
                return mean[:, :, : self.reward_size], torch.exp(logvar[:, :, : self.reward_size])

    def get_decay_loss(self):
        decay_loss = 0.0
        for m in self.children():
            if isinstance(m, EnsembleFC):
                decay_loss += m.weight_decay * torch.sum(torch.square(m.weight)) / 2.0
                # print(m.weight.shape)
                # print(m, decay_loss, m.weight_decay)
        return decay_loss

    def loss(self, mean, logvar, labels, inc_var_loss=True):
        """
        mean, logvar: Ensemble_size x N x dim
        labels: N x dim
        """
        assert len(mean.shape) == len(logvar.shape) == len(labels.shape) == 3
        inv_var = torch.exp(-logvar)
        if inc_var_loss:
            # Average over batch and dim, sum over ensembles.
            mse_loss = torch.mean(torch.mean(torch.pow(mean - labels, 2) * inv_var, dim=-1), dim=-1)
            var_loss = torch.mean(torch.mean(logvar, dim=-1), dim=-1)
            total_loss = torch.sum(mse_loss) + torch.sum(var_loss)
        else:
            mse_loss = torch.mean(torch.pow(mean - labels, 2), dim=(1, 2))
            total_loss = torch.sum(mse_loss)
        return total_loss, mse_loss

    def train(self, loss):
        self.optimizer.zero_grad()

        loss += 0.01 * torch.sum(self.max_logvar) - 0.01 * torch.sum(self.min_logvar)
        # print('loss:', loss.item())
        if self.use_decay:
            loss += self.get_decay_loss()
        loss.backward()
        # for name, param in self.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.grad.shape, torch.mean(param.grad), param.grad.flatten()[:5])
        self.optimizer.step()


class EnsembleDynamicsModel:
    def __init__(
        self,
        network_size,
        elite_size,
        state_size,
        action_size,
        reward_size=1,
        hidden_size=200,
        use_decay=False,
        args=None,
    ):
        self.network_size = network_size
        self.elite_size = elite_size
        self.model_list = []
        self.state_size = state_size
        self.action_size = action_size
        self.reward_size = reward_size
        self.network_size = network_size
        self.elite_model_idxes = []
        ####
        self.elite_model_idxes_reward = []

        self.ensemble_model = EnsembleModel(
            state_size,
            action_size,
            reward_size,
            network_size,
            hidden_size,
            use_decay=use_decay,
            args=args,
        )
        self.scaler = StandardScaler()
        self.state_size = state_size
        self.action_size = action_size

    def train(self, inputs, labels, batch_size=256, holdout_ratio=0.0, max_epochs_since_update=5):
        self._max_epochs_since_update = max_epochs_since_update
        self._epochs_since_update = 0
        self._state = {}
        self._snapshots = {i: (None, 1e10) for i in range(self.network_size)}

        num_holdout = int(inputs.shape[0] * holdout_ratio)
        permutation = np.random.permutation(inputs.shape[0])
        inputs, labels = inputs[permutation], labels[permutation]

        train_inputs, train_labels = inputs[num_holdout:], labels[num_holdout:]
        holdout_inputs, holdout_labels = inputs[:num_holdout], labels[:num_holdout]

        self.scaler.fit(train_inputs)
        train_inputs = self.scaler.transform(train_inputs)
        holdout_inputs = self.scaler.transform(holdout_inputs)

        holdout_inputs = torch.from_numpy(holdout_inputs).float().to(device)
        holdout_labels = torch.from_numpy(holdout_labels).float().to(device)
        holdout_inputs = holdout_inputs[None, :, :].repeat([self.network_size, 1, 1])
        holdout_labels = holdout_labels[None, :, :].repeat([self.network_size, 1, 1])

        for epoch in itertools.count():

            train_idx = np.vstack(
                [np.random.permutation(train_inputs.shape[0]) for _ in range(self.network_size)]
            )  # num model * len data
            # train_idx = np.vstack([np.arange(train_inputs.shape[0])] for _ in range(self.network_size))
            for start_pos in range(0, train_inputs.shape[0], batch_size):
                idx = train_idx[:, start_pos : start_pos + batch_size]
                train_input = (
                    torch.from_numpy(train_inputs[idx]).float().to(device)
                )  # num_model * batch * dim in
                train_label = torch.from_numpy(train_labels[idx]).float().to(device)
                mean, logvar = self.ensemble_model(train_input, mode="rs", ret_log_var=True)
                loss, _ = self.ensemble_model.loss(mean, logvar, train_label)
                self.ensemble_model.train(loss)

            with torch.no_grad():
                holdout_mean, holdout_logvar = self.ensemble_model(
                    holdout_inputs, mode="rs", ret_log_var=True
                )
                _, holdout_mse_losses = self.ensemble_model.loss(
                    holdout_mean, holdout_logvar, holdout_labels, inc_var_loss=False
                )
                holdout_mse_losses = holdout_mse_losses.detach().cpu().numpy()
                sorted_loss_idx = np.argsort(holdout_mse_losses)
                self.elite_model_idxes = sorted_loss_idx[: self.elite_size].tolist()
                break_train = self._save_best(epoch, holdout_mse_losses)
                if break_train:
                    break
        logger.info(
            "epoch: {}, holdout mse losses: [{}]".format(
                epoch, " ".join(map(str, holdout_mse_losses.tolist()))
            )
        )

    def _save_best(self, epoch, holdout_losses):
        updated = False
        for i in range(len(holdout_losses)):
            current = holdout_losses[i]
            _, best = self._snapshots[i]
            improvement = (best - current) / best
            if improvement > 0.01:
                self._snapshots[i] = (epoch, current)
                # self._save_state(i)
                updated = True
                # improvement = (best - current) / best

        if updated:
            self._epochs_since_update = 0
        else:
            self._epochs_since_update += 1
        if self._epochs_since_update > self._max_epochs_since_update:
            return True
        else:
            return False

    def predict(self, inputs, batch_size=1024, factored=True):
        inputs = self.scaler.transform(inputs)
        ensemble_mean, ensemble_var = [], []
        for i in range(0, inputs.shape[0], batch_size):
            input = (
                torch.from_numpy(inputs[i : min(i + batch_size, inputs.shape[0])])
                .float()
                .to(device)
            )
            b_mean, b_var = self.ensemble_model(
                input[None, :, :].repeat([self.network_size, 1, 1]), ret_log_var=False
            )
            ensemble_mean.append(b_mean.detach().cpu().numpy())
            ensemble_var.append(b_var.detach().cpu().numpy())
        ensemble_mean = np.hstack(ensemble_mean)
        ensemble_var = np.hstack(ensemble_var)

        if factored:
            return ensemble_mean, ensemble_var
        else:
            assert False, "Need to transform to numpy"
            mean = torch.mean(ensemble_mean, dim=0)
            var = torch.mean(ensemble_var, dim=0) + torch.mean(
                torch.square(ensemble_mean - mean[None, :, :]), dim=0
            )
            return mean, var

    def predict_tensor(self, inputs, batch_size=1024, factored=True):
        inputs = self.scaler.transform_tensor(inputs)
        ensemble_mean, ensemble_var = [], []
        for i in range(0, inputs.shape[0], batch_size):
            input = inputs[i : min(i + batch_size, inputs.shape[0])]
            b_mean, b_var = self.ensemble_model(
                input[None, :, :].repeat([self.network_size, 1, 1]), ret_log_var=False
            )
            ensemble_mean.append(b_mean)
            ensemble_var.append(b_var)
        ensemble_mean = torch.cat(ensemble_mean, 1)  ##
        ensemble_var = torch.cat(ensemble_var, 1)  ##

        if factored:
            return ensemble_mean, ensemble_var
        else:
            assert False, "Need to transform to numpy"
            mean = torch.mean(ensemble_mean, dim=0)
            var = torch.mean(ensemble_var, dim=0) + torch.mean(
                torch.square(ensemble_mean - mean[None, :, :]), dim=0
            )
            return mean, var


class EnsembleEnv:
    def __init__(
        self,
        network_size,
        elite_size,
        state_size,
        action_size,
        reward_size=1,
        hidden_size=200,
        use_decay=False,
        args=None,
    ):
        self.model = EnsembleDynamicsModel(
            network_size,
            elite_size,
            state_size,
            action_size,
            reward_size=reward_size,
            hidden_size=hidden_size,
            use_decay=use_decay,
            args=args,
        )
        self.trained = False
        self.penalize_var = getattr(args, "penalize_var", False)
        self.penalty_coeff = getattr(args, "penalty_coeff", 1.0)
        self.penalty_model_var = getattr(args, "penalty_model_var", False)

    def step(self, obs, act, deterministic=False):
        if len(obs.shape) == 1:
            obs = obs[None]
            act = act[None]
            return_single = True
        else:
            return_single = False

        inputs = np.concatenate((obs, act), axis=-1)
        ensemble_model_means, ensemble_model_vars = self.model.predict(inputs)
        ensemble_model_means[:, :, 1:] += obs
        ensemble_model_stds = np.sqrt(ensemble_model_vars)

        if deterministic:
            ensemble_samples = ensemble_model_means
        else:
            ensemble_samples = (
                ensemble_model_means
                + np.random.normal(size=ensemble_model_means.shape) * ensemble_model_stds
            )

        num_models, batch_size, _ = ensemble_model_means.shape

        batch_idxes = np.arange(0, batch_size)

        model_idxes = np.random.choice(self.model.elite_model_idxes, size=batch_size)

        samples = ensemble_samples[model_idxes, batch_idxes]
        model_means = ensemble_model_means[model_idxes, batch_idxes]
        model_stds = ensemble_model_stds[model_idxes, batch_idxes]

        rewards, next_obs = samples[:, :1], samples[:, 1:]

        if return_single:
            next_obs = next_obs[0]
            rewards = rewards[0]

        return next_obs, rewards

    def step_tensor(self, obs, act, deterministic=False, ensemble=False, use_penalty=False):
        if len(obs.shape) == 1:
            obs = obs[None]
            act = act[None]
            return_single = True
        else:
            return_single = False

        inputs = torch.cat((obs, act), axis=-1)
        ensemble_model_means, ensemble_model_vars = self.model.predict_tensor(inputs)
        ensemble_model_means[:, :, 1:] += obs
        ensemble_model_stds = torch.sqrt(ensemble_model_vars)

        if deterministic:
            ensemble_samples = ensemble_model_means
        else:
            ensemble_samples = (
                ensemble_model_means
                + torch.randn(size=ensemble_model_means.shape) * ensemble_model_stds
            )

        num_models, batch_size, _ = ensemble_model_means.shape
        if ensemble:
            samples = ensemble_samples[self.model.elite_model_idxes].mean(dim=0)
        else:
            batch_idxes = np.arange(0, batch_size)
            model_idxes = np.random.choice(self.model.elite_model_idxes, size=batch_size)
            samples = ensemble_samples[model_idxes, batch_idxes]
        # model_means = ensemble_model_means[model_idxes, batch_idxes]
        # model_stds = ensemble_model_stds[model_idxes, batch_idxes]

        rewards, next_obs = samples[:, :1], samples[:, 1:]

        if self.penalize_var and use_penalty:
            if self.penalty_model_var:
                mean_obs_means = torch.mean(ensemble_model_means, dim=0, keepdim=True)
                diffs = ensemble_model_means - mean_obs_means
                dists = torch.linalg.norm(diffs, dim=2, keepdim=True)
                penalty = torch.max(dists, dim=0)[0]
            else:
                penalty = torch.max(
                    torch.linalg.norm(ensemble_model_stds, dim=2, keepdim=True), dim=0
                )[0]
            rewards = rewards - self.penalty_coeff * penalty

        if return_single:
            next_obs = next_obs[0]
            rewards = rewards[0]

        return next_obs, rewards

    def rollout_H(self, obs, agent, H=10, deterministic=False):
        assert H >= 1
        s_0 = obs.copy()
        s = s_0.copy()
        reward_rollout = []
        len_rollout = 0
        for ii in range(H):
            act = agent.select_action(s)
            if ii == 0:
                a_0 = act.copy()
            next_s, rewards = self.step(s, act)
            reward_rollout.append(rewards)
            len_rollout += 1
            s = next_s.copy()
        s_H = next_s
        a_H = agent.select_action(s_H)
        return s_0, a_0, s_H, a_H, reward_rollout, len_rollout

    def rollout_H_tensor(self, obs, agent, H=10, deterministic=False):
        s_0 = obs.clone().detach()  # clone: remove shared storage; detach(): remove gradient flow
        s = s_0.clone()
        reward_rollout = []
        len_rollout = 0
        for ii in range(H):
            act, _, _ = agent.policy.sample(s)
            if ii == 0:
                a_0 = act.clone()
            next_s, rewards = self.step_tensor(s, act)
            reward_rollout.append(rewards)
            len_rollout += 1
            s = next_s.clone()
        s_H = next_s
        a_H, _, _ = agent.select_action(s_H)
        return s_0, a_0, s_H, a_H, reward_rollout, len_rollout

    def update(self, env_pool, batch_size, weight_grad=0, near_n=5):
        state, action, reward, next_state, mask, done = env_pool.sample(len(env_pool))
        delta_state = next_state - state
        inputs = np.concatenate((state, action), axis=-1)
        labels = np.concatenate((np.reshape(reward, (reward.shape[0], -1)), delta_state), axis=-1)

        self.model.train(inputs, labels, batch_size=256, holdout_ratio=0.2)
