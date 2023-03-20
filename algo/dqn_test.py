import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as pyd
import pdb
from time import time
import utils


class Encoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()
        self.repr_dim = 64 * 7 * 7

        self.convnet = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU())

    def forward(self, obs):
        obs = obs / 255.
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        return h


class Critic(nn.Module):
    def __init__(self, obs_shape, num_actions, hidden_dim, feature_dim, trunk_type):
        super().__init__()

        self.encoder = Encoder(obs_shape)

        if trunk_type == 'id':
            self.trunk = nn.Identity()
            self.feature_dim = self.encoder.repr_dim
        elif trunk_type == 'proj':
            self.trunk = nn.Sequential(nn.Linear(self.encoder.repr_dim, feature_dim))
            self.feature_dim = feature_dim
        elif trunk_type == 'proj+ln+tanh':
            self.trunk = nn.Sequential(nn.Linear(self.encoder.repr_dim, feature_dim),
                                       nn.LayerNorm(feature_dim), nn.Tanh())
            self.feature_dim = feature_dim
            
        

        self.V = nn.Sequential(nn.Linear(self.feature_dim, hidden_dim),
                               nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.A = nn.Sequential(nn.Linear(self.feature_dim, hidden_dim),
                               nn.ReLU(), nn.Linear(hidden_dim, num_actions))

        self.apply(utils.weight_init)

    def forward(self, obs):
        h = self.encoder(obs)
        h = self.trunk(h)
        v = self.V(h)
        a = self.A(h)
        q = v + a - a.mean(1, keepdim=True)
        return q


class DQNAgent:
    def __init__(self, obs_shape, num_actions, lr,
                 critic_target_tau, critic_target_update_every_steps, train_eps_min,
                 train_eps_decay_steps, eval_eps, update_every_steps, hidden_dim,
                 device, use_tb, max_grad_norm, clip_reward, nstep, obs_type, feature_dim, trunk_type):
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

        self.critic = Critic(obs_shape, num_actions, hidden_dim, feature_dim, trunk_type).to(device)
        self.critic_target = Critic(obs_shape, num_actions,
                                    hidden_dim, feature_dim, trunk_type).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.encoder = self.critic.encoder

        self.opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.critic_target.train()
        self.train()

    def train(self, training=True):
        self.training = training
        self.critic.train(training)

    def compute_train_eps(self, step):
        train_eps = max(
            0, self.train_eps_decay_steps - step) / self.train_eps_decay_steps
        return max(self.train_eps_min, train_eps)

    def act(self, obs, step, eval_mode):
        train_eps = self.compute_train_eps(step)
        eps = self.eval_eps if eval_mode else train_eps
        if np.random.rand() < eps:
            action = np.random.randint(self.num_actions)
        else:
            obs = torch.as_tensor(obs, device=self.device).unsqueeze(0)
            Qs = self.critic(obs)
            action = Qs.argmax(dim=1).item()
        return action

    def update_critic(self, batch, step):
        metrics = dict()
        obs, action, reward, discount, next_obs = batch
        if self.clip_reward:
            reward = torch.sign(reward)

        with torch.no_grad():
            next_action = self.critic(next_obs).argmax(dim=1).unsqueeze(1)
            next_Qs = self.critic_target(next_obs)
            next_Q = next_Qs.gather(1, next_action).squeeze(1)
            target_Q = reward + discount * next_Q

        # get current Q estimates
        Qs = self.critic(obs)
        Q = Qs.gather(1, action.unsqueeze(1)).squeeze(1)

        critic_loss = F.smooth_l1_loss(Q, target_Q)

        if self.use_tb:
            metrics['q'] = Q.mean().item()
            metrics['batch_reward'] = reward.mean().item()
            metrics['critic_loss'] = critic_loss.item()
            metrics['train_eps'] = self.compute_train_eps(step)

        self.opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.opt.step()

        return metrics

    def update(self, batch, step):
        metrics = dict()

        #if step % self.update_every_steps != 0:
        #    return metrics
        
        t0 = time()
        metrics.update(self.update_critic(batch, step))
        #print(f'Critic Update: {time() - t0}')
        t0 = time()
        if step % self.critic_target_update_every_steps == 0:
            utils.soft_update_params(self.critic, self.critic_target,
                                     self.critic_target_tau)
            #print(f'Soft Update: {time() - t0}')

        return metrics
