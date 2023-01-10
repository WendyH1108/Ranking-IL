import numpy as np
import torch
from torch._C import wait
import torch.nn.functional as F

from collections import deque

import utils

from replay_buffer import make_expert_replay_loader
from nets import Discriminator

from .agent import Agent

from time import perf_counter


class BoostingAgent(Agent):
    def __init__(
        self,
        name,
        batch_size,
        task,
        device,
        feature_dim,
        algo,
        representation,
        disc_hidden_dim,
        disc_type,
        disc_update_iter,
        n_learners,
        discount,
        divergence,
    ):

        super().__init__(name, task, device, algo)
        assert disc_type == "s" or disc_type == "ss" or disc_type == "sa"
        self.disc_type = disc_type  # r(s), r(s, s'), r(s, a)

        # demos_path = expert_dir + task + "/expert_demos.pkl"
        self.divergence = divergence
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

        self.discriminator_opt = torch.optim.Adam(
            self.discriminator.parameters(), lr=self.policy.lr, maximize=False
        )

        self.disc_update_iter = disc_update_iter
        self.device = device
        self.batch_size = batch_size
        self.learners = deque(maxlen=n_learners)
        self.discount = discount

    def __repr__(self):
        return "boosting"

    def encode(self, obs):
        if self.representation == "rl_encoder":
            return self.policy.actor.trunk(self.policy.encoder(obs))
        elif self.representation == "discriminator":
            return self.discriminator.encode(obs)

    def get_rewards(self, obs):
        if self.policy.obs_type == "pixels":
            obs = self.encode(obs)
        with torch.no_grad(), utils.eval_mode(self.discriminator):
            d = self.discriminator(obs)
        return d.flatten().detach().reshape(-1, 1)

    def compute_offline_rewards(self, time_steps):
        #Preprocess observations and demos
        observations = np.stack([time_step.observation for time_step in time_steps[:-1]], 0)
        actions = np.stack([time_step.action for time_step in time_steps[1:]], 0)

        disc_in = np.concatenate([observations, actions], axis=1)
        disc_in = torch.tensor(disc_in).to(self.device)
        rewards = self.get_rewards(disc_in)
        #Preprocess time steps
        # JONATHAN: fixed handling of dummy time_step 
        for i in range(1, len(time_steps)):
            time_steps[i] = time_steps[i]._replace(reward=rewards[i-1, 0])
        return time_steps

    def reset_policy(self, reinit_policy=False):
        self.policy.reset_noise()
        if reinit_policy:
            self.policy.reinit_policy()

    def add_learner(self):
        self.learners.append(self.policy.actor.state_dict())

    def sample_learner(self, weights):
        """Returns a policy from ensemble of policies"""
        if len(self.learners) == 0:
            return
        sampled_weights = np.random.choice(self.learners, p=weights)
        self.policy.eval_actor.load_state_dict(sampled_weights)

    @torch.no_grad()
    def boosted_act(self, obs):
        obs = torch.as_tensor(obs, device=self.device).unsqueeze(0)
        dist = self.policy.eval_actor(obs)
        action = dist.mean
        return action.cpu().numpy()[0]

    def update_discriminator(self, replay_iter, expert_iter):
        metrics = dict()

        for _ in range(self.disc_update_iter):
            policy_batch = next(replay_iter)
            expert_batch = next(expert_iter)

            # expert_data = np.concatenate([expert_batch[0], np.squeeze(expert_batch[1], axis=1)], axis=1)
            # policy_data = np.concatenate([policy_batch[0], np.squeeze(policy_batch[1], axis=1)], axis=1)

            expert_data = torch.cat(
                [expert_batch[0], torch.squeeze(expert_batch[1], dim=1)], dim=1
            )

            policy_data = np.concatenate([policy_batch[0], policy_batch[1]], axis=1)
            policy_data = torch.from_numpy(policy_data)

            # batch
            batch_size = self.batch_size // 2
            expert_data = expert_data[:batch_size]
            policy_data = policy_data[:batch_size]

            expert_data, policy_data = utils.to_torch(
                (expert_data, policy_data), self.device
            )
            disc_input = torch.cat([expert_data, policy_data], dim=0)
            disc_output = self.discriminator(disc_input, encode=False)

            if self.divergence == "js":
                ones = torch.ones(batch_size, device=self.device)
                zeros = torch.zeros(batch_size, device=self.device)
                disc_label = torch.cat([ones, zeros]).unsqueeze(dim=1)
                # add constraint here
                dac_loss = F.binary_cross_entropy_with_logits(
                    disc_output, disc_label, reduction="sum"
                )
                dac_loss /= batch_size
            elif self.divergence == "rkl":
                disc_expert, disc_policy = torch.split(disc_output, batch_size, dim=0)
                dac_loss = torch.mean(torch.exp(-disc_expert) + disc_policy)
            elif self.divergence == "wass":
                disc_expert, disc_policy = torch.split(disc_output, batch_size, dim=0)
                dac_loss = torch.mean(disc_policy - disc_expert)

            metrics["train/disc_loss"] = dac_loss.mean().item()

            grad_pen = utils.compute_gradient_penalty(
                self.discriminator, expert_data, policy_data
            )
            grad_pen /= batch_size
            # grad_pen *= -1

            self.discriminator_opt.zero_grad(set_to_none=True)
            dac_loss.backward()
            grad_pen.backward()
            self.discriminator_opt.step()

        return metrics

    def compute_divergence(self, expert_loader, on_policy_data):
        obs = on_policy_data[0]
        actions = on_policy_data[1]
        sample = expert_loader.dataset
        expert_traj_obs = torch.stack(
            list(utils.to_torch(sample.obs, self.device)), dim=0
        )
        expert_traj_actions = torch.stack(
            list(utils.to_torch(sample.act.squeeze(), self.device)), dim=0
        )
        expert_traj = torch.cat([expert_traj_obs, expert_traj_actions], dim=2)
        policy_traj = torch.cat([obs, actions], dim=len(obs.shape) - 1)
        return torch.mean(self.discriminator(expert_traj, encode=False)) - torch.mean(
            self.discriminator(policy_traj, encode=False)
        )

    def update(self, replay_iter, expert_iter, step):

        metrics = dict()
        if step % self.policy.update_every_steps != 0:
            return metrics

        replay_batch = next(replay_iter)
        expert_batch = next(expert_iter)

        # For the actions: TODO: fix
        expert_batch[1] = torch.squeeze(expert_batch[1], dim=1)

        replay_batch = utils.to_torch(replay_batch, self.device)

        # TODO: control expert batch
        expert_batch = utils.to_torch(expert_batch, self.device)

        #state = torch.cat([replay_batch[0], expert_batch[0]], dim=0)
        #action = torch.cat([replay_batch[1], expert_batch[1]], dim=0)
        #next_state = torch.cat([replay_batch[4], expert_batch[2]], dim=0)

        state, action, next_state = replay_batch[0], replay_batch[1], replay_batch[4]

        #discount = replay_batch[3].tile((2, 1))
        discount = replay_batch[3]

        # batch = [state, action, None, discount, next_state]
        batch = [state, action, replay_batch[2], discount, next_state]

        # FOR NOW 50/50
        # TODO: mix proportion control
        # batch = utils.to_torch(torch.cat([replay_batch, expert_batch], dim=0), self.device)

        # Update Reward
        # TODO: Handle different disc input types
        #rewards = self.get_rewards(torch.cat([batch[0], batch[1]], dim=1))

        # Nstep calculation
        #if self.policy.nstep > 1:
        #    n_int = self.policy.nstep - 1

        #    int_obs = replay_batch[-2 * n_int : -n_int]
        #    int_expert_obs = expert_batch[-2 * n_int : -n_int]

        #    int_act = replay_batch[-n_int:]
        #    int_expert_act = expert_batch[-n_int:]

        #    int_batch = []
        #    for o, a, eo, ea in zip(int_obs, int_act, int_expert_obs, int_expert_act):
        #        ea = torch.squeeze(ea, dim=1)
        #        obs = torch.cat([o, eo], dim=0)
        #        act = torch.cat([a, ea], dim=0)
        #        int_batch.append(torch.cat([obs, act], dim=1))
        #    #for o, a in zip(int_obs, int_act):
        #    #    int_batch.append(torch.cat([o, a], dim=1))

        #    for b in int_batch:
        #        int_r = self.get_rewards(b)
        #        rewards += self.discount * int_r

        #batch[2] = rewards

        metrics = self.policy.update(batch, step)

        return metrics
