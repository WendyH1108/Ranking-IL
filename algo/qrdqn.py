import numpy as np
import torch
import torch.nn.functional as F

from nets import DQNEncoder, QRDQNCritic


class QRDQNAgent:
    def __init__(
        self,
        obs_shape,
        num_actions,
        num_quantiles,
        kappa,
        lr,
        adam_eps,
        batch_size,
        update_every_steps,
        critic_target_update_every_steps,
        eval_eps,
        train_eps_min,
        train_eps_decay_steps,
        device,
        use_tb,
        max_grad_norm,
        clip_reward,
        nstep,
        obs_type,
        trunk_type,
    ):
        self.num_actions = num_actions
        self.num_quantiles = num_quantiles
        self.kappa = kappa
        self.critic_target_update_every_steps = critic_target_update_every_steps
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

        # Networks
        self.encoder = DQNEncoder(obs_shape[0], qr=True).to(device)
        self.encoder_target = DQNEncoder(obs_shape[0], qr=True).to(device)
        self.critic = QRDQNCritic(self.encoder.repr_dim, num_actions, num_quantiles).to(
            device
        )
        self.critic_target = QRDQNCritic(
            self.encoder.repr_dim, num_actions, num_quantiles
        ).to(device)
        self.update_target()

        # Optimizer
        self.opt = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.critic.parameters()),
            lr=lr,
            eps=adam_eps / batch_size,
        )

        self.encoder_target.train()
        self.critic_target.train()
        self.train()

        # For Quantile Huber Loss
        taus = (
            torch.arange(0, num_quantiles + 1, device=device, dtype=torch.float32)
            / num_quantiles
        )
        self.tau_hat = ((taus[1:] - taus[:-1]) / 2.0).view(1, num_quantiles)

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.critic.train(training)

    def compute_train_eps(self, step):
        train_eps = (
            max(0, self.train_eps_decay_steps - step) / self.train_eps_decay_steps
        )
        return max(self.train_eps_min, train_eps)

    def act(self, obs, step, eval_mode):
        train_eps = self.compute_train_eps(step)
        eps = self.eval_eps if eval_mode else train_eps
        if np.random.rand() < eps:
            action = np.random.randint(self.num_actions)
        else:
            obs = torch.as_tensor(obs, device=self.device).unsqueeze(0)
            obs = self.encoder(obs)
            action = (
                self.critic(obs).mean(dim=1).argmax(dim=1).item()
            )  # expectation over value dist
        return action

    def eval_quantiles(self, quantiles, actions):
        # Expand action in quantile dimension
        action_idx = actions[..., None, None].expand(self.batch_size, self.num_quantiles, 1)

        # Grab quantile values at specified actions
        return quantiles.gather(dim=2, index=action_idx)

    def huber_loss(self, td_errors):
        return torch.where(td_errors.abs() <= self.kappa, 0.5 * td_errors.pow(2), self.kappa*(td_errors.abs() - 0.5 * self.kappa))

    def update_critic(self, batch, step):
        metrics = dict()
        obs, action, reward, discount, next_obs = batch
        if self.clip_reward:
            reward = torch.sign(reward)

        with torch.no_grad():
            h = self.encoder(next_obs)
            next_q = self.critic(h).mean(dim=1)  # expectation over value dist
            next_action = next_q.argmax(dim=1)

            h_target = self.encoder_target(next_obs)
            next_quantiles = self.critic_target(h_target)
            next_sa_quantiles = self.eval_quantiles(
                next_quantiles, next_action
            ).transpose(1, 2)

            assert next_sa_quantiles.shape == (obs.size(0), 1, self.num_quantiles)

            target_quantiles = reward[..., None, None] + discount[..., None, None] * next_sa_quantiles
            
            assert target_quantiles.shape == (obs.size(0), 1, self.num_quantiles)

        # get current Q estimates
        h = self.encoder(obs)
        quantiles = self.critic(h)
        current_quantiles = self.eval_quantiles(quantiles, action)

        assert current_quantiles.shape == (obs.size(0), self.num_quantiles, 1)

        td_errors = target_quantiles - current_quantiles

        assert td_errors.shape == (obs.size(0), self.num_quantiles, self.num_quantiles)

        elementwise_huber = self.huber_loss(td_errors)

        assert elementwise_huber.shape == (
            obs.size(0),
            self.num_quantiles,
            self.num_quantiles,
        )

        elementwise_quantile_huber = (
            torch.abs(self.tau_hat[..., None] - (td_errors.detach() < 0).float())
            * elementwise_huber
            / self.kappa
        )

        critic_loss = elementwise_quantile_huber.sum(dim=1).mean()

        self.opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        # Clip Gradients
        if self.max_grad_norm:
            torch.nn.utils.clip_grad_norm_(
                self.encoder.parameters(), self.max_grad_norm
            )
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.opt.step()

        if self.use_tb:
            metrics["q"] = next_q.mean().item()
            metrics["batch_reward"] = reward.mean().item()
            metrics["critic_loss"] = critic_loss.item()
            metrics["train_eps"] = self.compute_train_eps(step)
        return metrics

    def update_target(self):
        self.encoder_target.load_state_dict(self.encoder.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

    def update(self, batch, step):

        metrics = dict()

        # Update Target
        if step % self.critic_target_update_every_steps == 0:
            self.update_target()

        # Update Critic
        if step % self.update_every_steps == 0:
            metrics.update(self.update_critic(batch, step))

        return metrics
