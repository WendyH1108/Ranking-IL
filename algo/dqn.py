import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle as pkl

import utils
from nets import DQNEncoder, DQNCritic, InverseDynamicsModel


class DQNAgent:
    def __init__(
        self,
        obs_shape,
        num_actions,
        lr,
        critic_target_tau,
        critic_target_update_every_steps,
        train_eps_min,
        train_eps_decay_steps,
        eval_eps,
        update_every_steps,
        hidden_dim,
        device,
        use_tb,
        max_grad_norm,
        clip_reward,
        nstep,
        obs_type,
        feature_dim,
        trunk_type,
        batch_size,
        adam_eps,
        task,
        bc_lam,
        use_idm,
        idm_lr,
    ):
        self.num_actions = num_actions
        self.critic_target_update_every_steps = critic_target_update_every_steps
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.train_eps_min = train_eps_min
        self.train_eps_decay_steps = train_eps_decay_steps
        self.eval_eps = eval_eps
        self.device = device
        self.use_tb = use_tb
        self.max_grad_norm = max_grad_norm
        self.clip_reward = clip_reward
        self.obs_type = obs_type
        self.obs_shape = obs_shape
        self.trunk_type = trunk_type
        self.lr = lr
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim
        self.task = task

        # ========= IDM / BC REG ==========
        self.bc_lam = bc_lam
        self.use_idm = use_idm
        if bc_lam > 0 and use_idm:
            # Prepare Expert Data
            self.process_expert()

            if use_idm:
                # Get IDM
                self.idm = InverseDynamicsModel(
                    obs_shape[0] * 2, feature_dim, hidden_dim, num_actions
                ).to(device)
                self.idm_opt = torch.optim.Adam(
                    self.idm.parameters(), lr=idm_lr, eps=adam_eps
                )
        # =================================

        # Networks
        self.encoder = DQNEncoder(obs_shape[0]).to(device)
        self.encoder_target = DQNEncoder(obs_shape[0]).to(device)
        self.critic = DQNCritic(
            self.encoder.repr_dim, num_actions, hidden_dim, feature_dim, trunk_type
        ).to(device)
        self.critic_target = DQNCritic(
            self.encoder.repr_dim, num_actions, hidden_dim, feature_dim, trunk_type
        ).to(device)
        self.encoder_target.load_state_dict(self.encoder.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizer
        self.opt = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.critic.parameters()),
            lr=lr,
            eps=adam_eps,
        )

        self.encoder_target.train()
        self.critic_target.train()
        self.train()

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.critic.train(training)

    def compute_train_eps(self, step):
        train_eps = (
            max(0, self.train_eps_decay_steps - step) / self.train_eps_decay_steps
        )
        return max(self.train_eps_min, train_eps)

    def act(self, obs, step, eval_mode, eval_eps=None):
        train_eps = self.compute_train_eps(step)
        # eps = self.eval_eps if eval_mode else train_eps
        if eval_mode:
            eps = eval_eps if eval_eps else self.eval_eps
        else:
            eps = train_eps

        if np.random.rand() < eps:
            action = np.random.randint(self.num_actions)
        else:
            obs = torch.as_tensor(obs, device=self.device).unsqueeze(0)
            obs = self.encoder(obs)
            Qs = self.critic(obs)
            action = Qs.argmax(dim=1).item()
        return action

    def process_expert(self, num_demos=10):
        expert_dir = "/private/home/jdchang/state-il/expert_demos/"
        demos_path = expert_dir + self.task + "/expert_demos.pkl"
        with open(demos_path, "rb") as f:
            data = pkl.load(f)
            obs = [x[:-1] for x in data[0][:num_demos]]
            next_obs = [x[:-1] for x in data[0][:num_demos]]
            act = [x[:-1] for x in data[-2][:num_demos]]
            self.expert_obs = torch.from_numpy(
                    np.concatenate(obs, axis=0)
            ).to(self.device)
            self.expert_next_obs = torch.from_numpy(
                    np.concatenate(next_obs, axis=0)
            ).to(self.device)
            if not self.use_idm:
                self.expert_act = torch.from_numpy(
                        np.concatenate(act, axis=0)
                ).to(self.device)
            else:
                self.expert_act = (
                        torch.from_numpy(np.concatenate(act, axis=0))
                    .squeeze()
                    .long()
                )

    def bc_loss(self):
        # TODO: perhaps do batched?
        # FIX: use the new datasets with next states
        if self.use_idm:
            with torch.no_grad():
                acts = self.idm(self.expert_obs, self.expert_next_obs)
        else:
            acts = self.expert_act
        act_pred = self.critic(self.encoder(self.expert_obs))
        return F.cross_entropy(act_pred, acts)

    def eval_idm(self):
        with torch.no_grad():
            acts = (
                self.idm(self.expert_obs, self.expert_next_obs).argmax(dim=1).cpu()
            )

        # Prediction Accuracy
        acc = (torch.sum(acts == self.expert_act) / self.expert_act.size(0)) * 100
        return acc

    def update_idm(self, batch):
        metrics = dict()
        # FIX: note this is different kind of batch than the one used for dqn
        obs, action, _, _, _, next_obs = batch
        action_pred = self.idm(obs, next_obs)
        loss = F.cross_entropy(action_pred, action)
        self.idm_opt.zero_grad(set_to_none=True)
        loss.backward()
        # Clip Gradients important for online model learning
        torch.nn.utils.clip_grad_norm_(self.idm.parameters(), self.max_grad_norm)
        self.idm_opt.step()

        metrics["idm_loss"] = loss.item()
        return metrics

    def update_critic(self, batch, step):
        metrics = dict()
        obs, action, reward, discount, next_obs = batch
        if self.clip_reward:
            reward = torch.sign(reward)

        with torch.no_grad():
            next_action = self.critic(self.encoder(next_obs)).argmax(dim=1).unsqueeze(1)
            next_Qs = self.critic_target(self.encoder_target(next_obs))
            next_Q = next_Qs.gather(1, next_action).squeeze(1)
            target_Q = reward + discount * next_Q

        # get current Q estimates
        Qs = self.critic(self.encoder(obs))
        Q = Qs.gather(1, action.unsqueeze(1)).squeeze(1)

        critic_loss = F.smooth_l1_loss(Q, target_Q)

        if self.bc_lam > 0:
            bc_loss = self.bc_loss()
            critic_loss = critic_loss + self.bc_lam * bc_loss

        # TODO: LOG BC LOSS
        if self.use_tb:
            metrics["q"] = Q.mean().item()
            metrics["batch_reward"] = reward.mean().item()
            metrics["critic_loss"] = critic_loss.item()
            metrics["train_eps"] = self.compute_train_eps(step)

        self.opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        # Clip Gradients
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.opt.step()

        return metrics

    def update(self, batch, step):

        metrics = dict()

        # if step % self.update_every_steps != 0:
        #    return metrics

        # update critic
        metrics.update(self.update_critic(batch, step))

        # update critic target
        if step % self.critic_target_update_every_steps == 0:
            with torch.no_grad():
                utils.soft_update_params(
                    self.encoder, self.encoder_target, self.critic_target_tau
                )
                utils.soft_update_params(
                    self.critic, self.critic_target, self.critic_target_tau
                )

        return metrics
