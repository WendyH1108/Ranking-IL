import numpy as np
import torch
import torch.nn.functional as F

import utils

from replay_buffer import make_expert_replay_loader
from nets import Discriminator

from .agent import Agent


class DACAgent(Agent):
    def __init__(
        self,
        name,
        batch_size,
        task,
        device,
        expert_dir,
        num_demos,
        feature_dim,
        reward_mode,
        algo,
        representation,
        disc_hidden_dim,
        disc_type,
    ):

        super().__init__(name, task, device, algo)
        assert disc_type == "s" or disc_type == "ss"
        self.disc_type = disc_type  # r(s) or r(s, s')

        # demos_path = expert_dir + task + "/expert_demos.pkl"

        self.representation = representation
        if self.representation == "rl_encoder":
            self.discriminator = Discriminator(feature_dim, disc_hidden_dim).to(device)
        elif self.representation == "discriminator":
            enc_in_dim = (
                self.policy.obs_shape[0]
                if disc_type == "s"
                else 2 * self.policy.obs_shape[0]
            )
            self.discriminator = Discriminator(
                feature_dim, disc_hidden_dim, enc_in_dim
            ).to(device)

        # self.expert_buffer = iter(
        #    make_expert_replay_loader(demos_path, num_demos, batch_size)
        # )

        self.discriminator_opt = torch.optim.Adam(
            self.discriminator.parameters(), lr=self.policy.lr
        )

        self.reward_mode = reward_mode
        self.device = device
        self.batch_size = batch_size

    def __repr__(self):
        return self.reward_mode

    def encode(self, obs):
        if self.representation == "rl_encoder":
            return self.policy.actor.trunk(self.policy.encoder(obs))
        elif self.representation == "discriminator":
            return self.discriminator.encode(obs)

    def compute_online_rewards(self, obs, step):
        if self.policy.obs_type == "pixels":
            obs = self.encode(obs)
        with torch.no_grad():
            with utils.eval_mode(self.discriminator):
                d = self.discriminator(obs)
            s = torch.sigmoid(d)
            # Mixture Reward....
            # rewards = s.log() - (1 - s).log()

            # Survival Bias (i.e. GAIL/DAC reward)
            rewards = -(1 - s).log()
            return rewards.flatten().detach().reshape(-1, 1)

    def compute_offline_rewards(self, time_steps, episode):
        traj = np.stack([time_step.observation for time_step in time_steps], 0)
        if self.disc_type == "s":
            obs = torch.tensor(traj[:-1]).to(self.device)
        else:
            # Concatenate state and next state
            obs = torch.cat(
                [torch.tensor(traj[:-1]), torch.tensor(traj[1:])], dim=1
            ).to(self.device)
        rewards = self.compute_online_rewards(obs, episode * obs.shape[0]).cpu().numpy()

        # Preprocess time steps
        for i in range(len(time_steps) - 1):
            time_steps[i] = time_steps[i]._replace(reward=rewards[i, 0])
        return time_steps, np.sum(rewards)

    def update_discriminator(self, batch, expert_iter):
        metrics = dict()

        # Grab Expert Data
        expert_obs, _, expert_obs_next = utils.to_torch(next(expert_iter), self.device)
        expert_obs = expert_obs.float()[: self.batch_size // 2]
        batch_size = expert_obs.shape[0]

        # Grab Policy Data
        obs = batch[0]
        policy_obs = obs[: self.batch_size // 2]

        with torch.no_grad():
            if self.disc_type == "ss":
                # (s, s')
                expert_obs_next = expert_obs_next.float()[: self.batch_size // 2]
                obs_next = batch[-1]
                policy_obs_next = obs_next[: self.batch_size // 2]
                expert_obs = torch.cat([expert_obs, expert_obs_next], dim=1)
                policy_obs = torch.cat([policy_obs, policy_obs_next], dim=1)
            disc_input = torch.cat([expert_obs, policy_obs], dim=0)

        if self.policy.obs_type == "pixels":
            with torch.no_grad():
                disc_input = self.encode(disc_input)

        disc_output = self.discriminator(disc_input, encode=False)

        ones = torch.ones(batch_size, device=self.device)
        zeros = torch.zeros(batch_size, device=self.device)
        disc_label = torch.cat([ones, zeros]).unsqueeze(dim=1)

        dac_loss = F.binary_cross_entropy_with_logits(
            disc_output, disc_label, reduction="sum"
        )
        dac_loss /= batch_size

        expert_obs, policy_obs = torch.split(disc_input, batch_size, dim=0)

        grad_pen = utils.compute_gradient_penalty(
            self.discriminator, expert_obs, policy_obs
        )  # used to detach
        grad_pen /= policy_obs.shape[0]

        metrics["disc_loss"] = dac_loss.mean().item()

        self.discriminator_opt.zero_grad(set_to_none=True)
        dac_loss.backward()
        grad_pen.backward()
        self.discriminator_opt.step()
        return metrics

    def update(self, replay_iter, expert_iter, step):

        metrics = dict()
        if step % self.policy.update_every_steps != 0:
            return metrics

        assert repr(self) != "online_imitation"
        batch = next(replay_iter)
        batch = utils.to_torch(batch, self.device)

        # Policy Update
        metrics.update(self.policy.update(batch[:-1], step))

        # Disciminator Update
        metrics.update(self.update_discriminator(batch, expert_iter))

        return metrics
