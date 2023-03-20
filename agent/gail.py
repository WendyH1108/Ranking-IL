import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import utils
from nets import Discriminator
from .agent import Agent
from torch.nn.utils import spectral_norm


class MLP(nn.Module):
    def __init__(
        self, input_dim, output_dim, hidden_dims, spectral_norms,
    ):
        super().__init__()

        assert len(hidden_dims) == len(spectral_norms)
        layers = []
        for dim, use_sn in zip(hidden_dims, spectral_norms):
            layers += [
                self.maybe_sn(nn.Linear(input_dim, dim), use_sn),
                nn.ReLU(inplace=True),
            ]
            input_dim = dim

        layers += [nn.Linear(input_dim, output_dim)]

        self.net = nn.Sequential(*layers)

    def maybe_sn(self, m, use_sn):
        return spectral_norm(m) if use_sn else m

    def forward(self, x):
        action = self.net(x)
        return action.clamp(-1.0 + 1e-6, 1.0 - 1e-6)


class GAILAgent(Agent):
    """
    State-Action GAIL with inverse model
    """

    def __init__(
        self,
        name,
        task,
        device,
        algo,
        feature_dim,
        reward_mode,
        representation,
        disc_hidden_dim,
        inverse_hidden_dims,
        inverse_spectral_norms,
    ):
        super().__init__(name, task, device, algo)

        if self.policy.obs_type == "pixels":
            self.representation = representation
            if self.representation == "rl_encoder":
                self.discriminator = Discriminator(feature_dim, disc_hidden_dim).to(
                    device
                )
            elif self.representation == "discriminator":
                self.discriminator = Discriminator(
                    feature_dim, disc_hidden_dim, self.policy.obs_shape[0]
                ).to(device)
        else:
            # (state, action) pairs
            disc_feature_dim = self.policy.obs_shape[0] + self.policy.action_dim
            self.discriminator = Discriminator(disc_feature_dim, disc_hidden_dim).to(
                device
            )
        self.discriminator_opt = torch.optim.Adam(
            self.discriminator.parameters(), lr=self.policy.lr
        )

        self.inverse_model = MLP(
            self.policy.obs_shape[0] * 2,
            self.policy.action_dim,
            inverse_hidden_dims,
            inverse_spectral_norms,
        ).to(device)
        self.inverse_model_loss = nn.MSELoss(reduction="mean")
        self.inverse_model_opt = torch.optim.Adam(
            self.inverse_model.parameters(), lr=self.policy.lr
        )

        self.reward_mode = reward_mode

    def __repr__(self):
        return "semi-supervised GAIL"

    def encode(self, obs):
        if self.representation == "rl_encoder":
            return self.policy.actor.trunk(self.policy.encoder(obs))
        elif self.representation == "discriminator":
            return self.discriminator.encode(obs)

    def compute_rewards(self, obs, action):
        if self.policy.obs_type == "pixels":
            obs = self.encode(obs)
        with torch.no_grad():
            with utils.eval_mode(self.discriminator):
                disc_input = torch.cat([obs, action], dim=1)
                d = self.discriminator(disc_input)
            s = torch.sigmoid(d)
            rewards = s.log() - (1 - s).log()
            return rewards.flatten().detach().reshape(-1, 1)

    def compute_and_store_rewards(self, time_steps, episode):
        # Preprocess observations and demos
        observations = np.stack([time_step.observation for time_step in time_steps], 0)
        observations = torch.tensor(observations).to(self.device)

        action = np.stack([time_step.action for time_step in time_steps], 0)
        action = torch.tensor(action).to(self.device)
        rewards = self.compute_rewards(observations, action).cpu().numpy()
        # Preprocess time steps
        for i in range(len(time_steps)):
            time_steps[i] = time_steps[i]._replace(reward=rewards[i, 0])
        return time_steps, np.sum(rewards)

    def update_inverse_model(self, obs, actions, next_obs):

        ahat = self.inverse_model(torch.cat([obs, next_obs], dim=1))
        loss = self.inverse_model_loss(ahat, actions)

        self.inverse_model_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.inverse_model_opt.step()

        return {"inv_model_loss": loss.item()}

    def update_discriminator(
        self,
        obs,
        action,
        expert_obs,
        expert_action,
        expert_next_obs,
        action_is_available,
    ):
        metrics = dict()

        with torch.no_grad():
            predicted_action = self.inverse_model(
                torch.cat([expert_obs, expert_next_obs], dim=1)
            )
        expert_action[action_is_available < 1] = predicted_action[
            action_is_available < 1
        ]

        expert_obs = expert_obs.float()[: self.policy.batch_size // 2]
        expert_action = expert_action.float()[: self.policy.batch_size // 2]

        policy_obs = obs[: self.policy.batch_size // 2]
        policy_action = action[: self.policy.batch_size // 2]

        half_batch_size = expert_obs.shape[0]

        with torch.no_grad():
            expert_data = torch.cat([expert_obs, expert_action], dim=1)
            policy_data = torch.cat([policy_obs, policy_action], dim=1)
            disc_input = torch.cat([expert_data, policy_data], dim=0)

        if self.policy.obs_type == "pixels":
            with torch.no_grad():
                disc_input = self.encode(disc_input)

        disc_output = self.discriminator(disc_input, encode=False)

        ones = torch.ones(half_batch_size, device=self.device)
        zeros = torch.zeros(half_batch_size, device=self.device)
        disc_label = torch.cat([ones, zeros]).unsqueeze(dim=1)

        dac_loss = F.binary_cross_entropy_with_logits(
            disc_output, disc_label, reduction="mean"
        )

        # expert_data, policy_data = torch.split(disc_input, half_batch_size, dim=0)

        grad_pen = utils.compute_gradient_penalty(
            self.discriminator, expert_data, policy_data
        )  # used to detach
        grad_pen /= policy_data.shape[0]

        metrics["disc_loss"] = dac_loss.item()

        self.discriminator_opt.zero_grad(set_to_none=True)
        dac_loss.backward()
        grad_pen.backward()
        self.discriminator_opt.step()
        return metrics

    def update(self, replay_iter, expert_buffer_iter, inverse_replay_iter, step):

        metrics = dict()

        # 1. update the inverse model
        inverse_model_batch = next(inverse_replay_iter)
        obs, action, reward, discount, next_obs = utils.to_torch(
            inverse_model_batch, self.device
        )
        metrics.update(self.update_inverse_model(obs, action, next_obs))

        # 2. update GAIL
        # a) discriminator
        batch = next(replay_iter)
        obs, action, reward, discount, next_obs = utils.to_torch(batch, self.device)
        (
            expert_obs,
            expert_action,
            expert_next_obs,
            action_is_available,
        ) = utils.to_torch(next(expert_buffer_iter), self.device)
        metrics.update(
            self.update_discriminator(
                obs,
                action,
                expert_obs,
                expert_action,
                expert_next_obs,
                action_is_available,
            )
        )

        # b) policy (analogous to generator)
        if self.reward_mode == "on_the_fly":
            reward = self.compute_rewards(obs, action)
        metrics.update(
            self.policy.update((obs, action, reward, discount, next_obs), step)
        )

        return metrics
