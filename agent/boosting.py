import numpy as np
import torch
import torch.nn.functional as F

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
        expert_dir,
        num_demos,
        feature_dim,
        reward_mode,
        algo,
        representation,
        disc_hidden_dim,
        disc_type,
        discriminator_iter,
        policy_iter
    ):

        super().__init__(name, task, device, algo)
        assert disc_type == "s" or disc_type == "ss" or disc_type == "sa"
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
            self.discriminator_reward = Discriminator(
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
        self.discriminator_iter = discriminator_iter
        self.policy_iter = policy_iter
        self.discriminator_list = []

    def __repr__(self):
        return self.reward_mode

    def encode(self, obs):
        if self.representation == "rl_encoder":
            return self.policy.actor.trunk(self.policy.encoder(obs))
        elif self.representation == "discriminator":
            return self.discriminator.encode(obs)

    def compute_online_rewards(self, obs, step, fictitious = False):
        if self.policy.obs_type == "pixels":
            obs = self.encode(obs)
        with torch.no_grad():
            with utils.eval_mode(self.discriminator):
                if not fictitious:
                    d = self.discriminator(obs)
                else:
                    reward_list = []
                    for dis_state in self.discriminator_list:
                        self.discriminator_reward.load_state_dict(dis_state)
                        reward_list.append(self.discriminator_reward(obs))
                    if len(reward_list)!=0:
                        d = sum(reward_list) / len(reward_list)
                    else:
                        d = self.discriminator(obs)
                
            rewards = d
            return rewards.flatten().detach().reshape(-1, 1)
            # return d

    def compute_offline_rewards(self, time_steps, episode):
        traj = np.stack([time_step.observation for time_step in time_steps], 0)
        action_traj = np.stack([time_step.action for time_step in time_steps], 0)
        if self.disc_type == "s":
            obs = torch.tensor(traj[:-1]).to(self.device)
        if self.disc_type == "sa":
            obs = torch.cat(
                [torch.tensor(traj[:-1]), torch.tensor(action_traj[1:])], dim=1
            ).to(self.device)
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

    def update_discriminator(self, batch, expert_loader, expert_iter, step):
        metrics = dict()

        # Grab Expert Data
        expert_obs, expert_actions, expert_obs_next = utils.to_torch(next(expert_iter), self.device)
        expert_obs = expert_obs.float()[: self.batch_size // 2]
        batch_size = expert_obs.shape[0]
        expert_actions = np.squeeze(expert_actions,axis=1)[: self.batch_size // 2]
        # Grab Policy Data
        obs = batch[0]
        actions = batch[1]
        policy_obs = obs[: self.batch_size // 2]
        policy_actions = np.squeeze(expert_actions,axis=1)[: self.batch_size // 2]
            
        with torch.no_grad():
            if self.disc_type == "ss":
                # (s, s')
                expert_obs_next = expert_obs_next.float()[: self.batch_size // 2]
                obs_next = batch[-1]
                policy_obs_next = obs_next[: self.batch_size // 2]
                expert_obs = torch.cat([expert_obs, expert_obs_next], dim=1)
                policy_obs = torch.cat([policy_obs, policy_obs_next], dim=1)
            disc_input = torch.cat([expert_obs, policy_obs], dim=0)

            if self.disc_type == "sa":
                expert_data = torch.cat([expert_obs, expert_actions], dim=1)
                policy_data = torch.cat([policy_obs, policy_actions], dim=1)
        if self.policy.obs_type == "pixels":
            with torch.no_grad():
                disc_input = self.encode(disc_input)
                                
        dac_loss = torch.mean(self.discriminator(expert_data,encode=False)) - torch.mean(self.discriminator(policy_data,encode=False))
        dac_loss /= batch_size
        metrics["train/disc_loss"] = dac_loss.mean().item()

        # expert_obs, policy_obs = torch.split(disc_input, batch_size, dim=0)

        grad_pen = utils.compute_gradient_penalty(
            self.discriminator, expert_data, policy_data
        )  # used to detach
        grad_pen /= policy_obs.shape[0]
        
        self.discriminator_opt.zero_grad(set_to_none=True)
        dac_loss = -dac_loss
        dac_loss.backward()
        grad_pen.backward()
        self.discriminator_opt.step()
        if step % 1000 == 0:
            if len(self.discriminator_list) == 20:
                self.discriminator_list = self.discriminator_list[1:]
            self.discriminator_list.append(self.discriminator.state_dict())
        return metrics

    def collect_policy_batch(self,env, step, buffer):
        eval_return = 0
        states = list()
        actions = list()
        next_states = list()
        time_list = list()
        # for b in self.batch_size:
        for traj in range(5):
            time_step = env.reset()
            traj_states = list()
            traj_actions = list()
            traj_next_states = list()
            traj_time = list()
            while not time_step.last():
                traj_time.append(time_step)
                traj_states.append(np.array(time_step.observation))
                with torch.no_grad():
                    action = self.act(
                        time_step.observation, step, eval_mode=True
                    )
                # reward = time_step.reward
                time_step = env.step(action)

                traj_next_states.append(np.array(time_step.observation))
                # rewards.append(time_step.reward)
                traj_actions.append(np.array([time_step.action]))
                # dones.append(bool(time_step.last() == True))
            states.append(np.array(traj_states))
            time_list.append(traj_time)
            actions.append(np.squeeze(np.array(traj_actions),axis = 1))
            next_states.append(np.array(traj_next_states))
        
        batch_states = list()
        batch_actions = list()
        batch_next_states = list()
        traj_idx = np.random.randint(0,5,self.batch_size // 2)
        idx = np.random.randint(0,obs.shape[1],self.batch_size // 2)

        size = self.batch_size // 2
        # random select pairs
        for i in range(size):
            batch_states.append(states[traj_idx[i]][idx[i]])
            buffer.add(time_list[traj_idx[i]][idx[i]])
            batch_actions.append(actions[traj_idx[i]][idx[i]])
            batch_next_states.append(next_states[traj_idx[i]][idx[i]])
        return (np.array(states), np.array(actions), np.array(next_states)) ,(np.array(batch_states), np.array(batch_actions), np.array(batch_next_states))
    
    def compute_divergence(self, expert_loader, on_policy_data):
        obs = on_policy_data[0]
        actions = on_policy_data[1]
        sample = expert_loader.dataset
        expert_traj_obs = torch.stack(list(utils.to_torch(sample.obs,self.device)), dim=0)
        expert_traj_actions = torch.stack(list(utils.to_torch(sample.act.squeeze(),self.device)), dim=0)
        expert_traj = torch.cat([expert_traj_obs, expert_traj_actions], dim=2)
        policy_traj = torch.cat([obs, actions], dim=len(obs.shape)-1)
        return torch.mean(self.discriminator(expert_traj,encode=False)) - torch.mean(self.discriminator(policy_traj,encode=False))    
    
    def online_sample(self):
        pass
    
    def update(self, replay_iter, expert_loader, expert_iter, step):

        metrics = dict()
        if step % self.policy.update_every_steps != 0:
            return metrics

        assert repr(self) != "online_imitation"
        
        self.policy.reset_step()
        for _ in range(self.discriminator_iter):
            batch = next(replay_iter)
            batch = utils.to_torch(batch, self.device)
            metrics.update(self.update_discriminator(batch, expert_loader,expert_iter, step))
        
        # Policy Update
        for _ in range(self.policy_iter):
            online_iter = self.online_sample()
            batch = next(online_iter)
            batch = utils.to_torch(batch, self.device)
            metrics.update(self.policy.update(batch, step))
        
        return metrics
    