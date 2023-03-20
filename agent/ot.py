import numpy as np
import torch
import torch.nn as nn
import time
import pickle as pkl

import utils
from nets import Encoder, DQNEncoder, OTILRep

from .ot_utils import (
    Cost,
    EllipticalCost,
    Solver,
    Aggregator,
    Preprocessor,
    reward_scalings,
)
from .agent import Agent


class OTAgent(Agent):
    def __init__(
        self,
        name,
        task,
        device,
        algo,
        expert_dir,
        num_demos,
        cost_fun,
        aggregator_fun,
        solver_fun,
        preprocessor_fun,
        partial_updates,
        update_preprocessor_every_episode,
        update_encoder_every_episode,
        normalize,
        reward_scaling_fun,
        reward_scaling_alpha,
        topk,
        concatenate,
        fit_to_expert,
        reward_mode,
        center_obs,
        load_encoder,
        use_trunk,
        rep_layer,  # [enc, v, a, None]
        handcrafted,
        inv_reg,
        epsilon,
        disc_type,  # s, ss
    ):
        super().__init__(name, task, device, algo)

        self.reward_scaling = reward_scalings[reward_scaling_fun]
        self.reward_scaling_alpha = reward_scaling_alpha
        self.fit_to_expert = fit_to_expert
        self.center_obs = center_obs
        self.reward_mode = reward_mode
        self.update_encoder_every_episode = update_encoder_every_episode
        self.load_encoder = load_encoder
        self.use_trunk = use_trunk
        self.rep_layer = rep_layer
        self.handcrafted = handcrafted
        self.disc_type = disc_type

        # Grab Expert Demonstrations
        demos_path = expert_dir + task + "/expert_demos.pkl"
        if self.policy.obs_type == "pixels":
            preprocessor_fun = "noScaler"
        with open(demos_path, "rb") as f:

            data = pkl.load(f)

            self.expert_demos = data[0][:num_demos]
            self.expert_return = data[-1][:num_demos]
            # self.expert_handcraft = self.expert_handcraft[:num_demos]

            if concatenate:
                self.expert_demos = [
                    demo[:: len(self.expert_demos)] for demo in self.expert_demos
                ]
                self.expert_demos = [np.concatenate(self.expert_demos, 0)]
        self.expert_demos_rescaled = None

        # Load Expert Encoder
        if load_encoder:
            self.encoder_params = torch.load(
                expert_dir + task + "_noop/expert_encoder.pt",
                map_location=torch.device("cpu"),
            )
            self.critic_params = torch.load(
                expert_dir + task + "_noop/expert_critic.pt",
                map_location=torch.device("cpu"),
            )
            print("Encoder Loaded")

        # Initialize Components
        if cost_fun == "elliptical":
            # TODO: make more general..... we would want the encoded features
            # for now just for handcraft
            self.cost = EllipticalCost(self.expert_handcraft, reg=inv_reg)
        else:
            self.cost = Cost(cost_fun, normalize=normalize)
        self.solver = Solver(solver_fun, epsilon=epsilon)
        self.aggregator = Aggregator(aggregator_fun, topk)
        self.preprocessor = Preprocessor(
            preprocessor_function=preprocessor_fun,
            partial_updates=partial_updates,
            update_preprocessor_every_episode=update_preprocessor_every_episode,
        )
        self.base_algo_is_dqn = "dqn" in str(self.policy)

        # Initialize Targets
        if self.policy.obs_type == "pixels":
            if self.base_algo_is_dqn:

                # NOTE: currently setup for dqn exp
                assert self.policy.trunk_type == "proj"
                if use_trunk:
                    self.trunk_target = nn.Sequential(
                        nn.Linear(
                            self.policy.encoder.repr_dim, self.policy.critic.feature_dim
                        )
                    ).to(device)
                self.encoder_target = DQNEncoder(self.policy.obs_shape[0]).to(device)

                # self.encoder_target = OTILRep(
                #     self.policy.obs_shape[0],
                #     self.policy.num_actions,
                #     self.policy.hidden_dim,
                #     self.policy.feature_dim,
                #     encoder_params=self.encoder_params,
                #     critic_params=self.critic_params,
                # ).to(device)

                # self.trunk_target = nn.Sequential(
                #     nn.Linear(
                #         self.encoder_target.encoder.repr_dim,
                #         self.encoder_target.feature_dim,
                #     )
                # ).to(device)

            else:
                self.trunk_target = nn.Sequential(
                    nn.Linear(self.policy.encoder.repr_dim, self.policy.feature_dim),
                    nn.LayerNorm(self.policy.feature_dim),
                    nn.Tanh(),
                ).to(device)
                self.encoder_target = Encoder(self.policy.obs_shape[0]).to(device)

        if self.fit_to_expert:
            self.preprocessor.fit(np.concatenate(self.expert_demos, 0), 0)

    def __repr__(self):
        return self.reward_mode

    def encode(self, observations, episode, handcrafted=None):
        if self.policy.obs_type == "pixels":
            observations = torch.from_numpy(observations).float().to(self.device)
            if episode % self.update_encoder_every_episode == 0:
                if self.base_algo_is_dqn:

                    if self.load_encoder:
                        # NOTE: Fixing weights to experts
                        self.encoder_target.load_state_dict(self.encoder_params)
                    else:
                        self.encoder_target.load_state_dict(
                            self.policy.encoder.state_dict()
                        )

                    if self.use_trunk:
                        self.trunk_target.load_state_dict(
                            self.policy.critic.trunk.state_dict()
                        )
                else:
                    self.encoder_target.load_state_dict(
                        self.policy.encoder.state_dict()
                    )
                    self.trunk_target.load_state_dict(
                        self.policy.actor.trunk.state_dict()
                    )
            if self.use_trunk:
                return (
                    self.trunk_target(self.encoder_target(observations))
                    .detach()
                    .cpu()
                    .numpy()
                )
            else:
                return self.encoder_target(observations).detach().cpu().numpy()

            # Just return handcrafted features
            # if self.rep_layer == "handcraft":
            #     return handcrafted

            # Get the proper layer for the representations
            # h_enc, h_v, h_a = self.encoder_target(observations)
            # if self.rep_layer == "enc":
            #     rep = self.trunk_target(h_enc)
            # elif self.rep_layer == "v":
            #     rep = h_v
            # elif self.rep_layer == "a":
            #     rep = h_a

            # Concatenate handcrafted features or not
            # if self.handcrafted:
            #     rep = np.concatenate([rep.detach().cpu().numpy(), handcrafted], axis=1)
            # else:
            #     rep = rep.detach().cpu().numpy()
            # return rep

        else:
            return observations

    def compute_offline_rewards(self, time_steps, episode):
        # Preprocess observations and demos
        observations = np.stack([time_step.observation for time_step in time_steps], 0)
        # handcrafted = np.stack([time_step.handcraft for time_step in time_steps], 0)

        if not self.fit_to_expert:
            self.preprocessor.fit(observations, episode)

        # Get Expert Features
        if (self.expert_demos_rescaled is None) or (
            episode % self.update_encoder_every_episode == 0
        ):
            self.expert_demos_rescaled = [
                self.encode(self.preprocessor.transform(demo), episode)
                for demo in self.expert_demos
            ]
            # self.expert_demos_rescaled = [
            #     self.encode(
            #         self.preprocessor.transform(demo), episode, handcrafted=handcraft
            #     )
            #     for demo, handcraft in zip(self.expert_demos, self.expert_handcraft)
            # ]
            if self.disc_type == "ss":
                self.expert_demos_rescaled = [
                    np.concatenate([demo[:-1], demo[1:]], axis=1)
                    for demo in self.expert_demos_rescaled
                ]

        # Get Behavior Agent Features
        observations_rescaled = self.encode(
            self.preprocessor.transform(observations), episode
        )

        if self.disc_type == "ss":
            observations_rescaled = np.concatenate(
                [observations_rescaled[:-1], observations_rescaled[1:]], axis=1
            )

        # observations_rescaled = self.encode(
        #     self.preprocessor.transform(observations), episode, handcrafted=handcrafted
        # )

        scores_list = list()
        ot_rewards_list = list()
        t0 = time.time()
        for demo in self.expert_demos_rescaled:
            # Compute cost
            if self.center_obs:
                observations_rescaled_mean = np.mean(observations_rescaled, axis=0)
                demo_centered = demo - observations_rescaled_mean
                obs_centered = observations_rescaled - observations_rescaled_mean
                cost_matrix = self.cost.compute(obs_centered, demo_centered)
                transport_plan = self.solver.compute(
                    obs_centered, demo_centered, cost_matrix
                )
            else:
                cost_matrix = self.cost.compute(observations_rescaled, demo)
                transport_plan = self.solver.compute(
                    observations_rescaled, demo, cost_matrix
                )
            # Compute alignment
            # Compute rewards
            ot_costs = np.diag(transport_plan @ cost_matrix.T)
            ot_rewards = self.reward_scaling(ot_costs, self.reward_scaling_alpha)
            scores_list.append(np.sum(ot_rewards))
            ot_rewards_list.append(ot_rewards)

        # Aggregate
        ot_rewards = self.aggregator.compute(ot_rewards_list, scores_list)

        # Preprocess time steps
        for i in range(len(time_steps) - 1):
            time_steps[i] = time_steps[i]._replace(reward=ot_rewards[i])
        # print(f"ot: {time.time()-t0}")
        return time_steps, np.sum(ot_rewards)

    def update(self, replay_iter, step):

        metrics = dict()
        # TODO: make general for other algos
        if step % self.policy.update_every_steps != 0:
            return metrics

        assert repr(self) != "online_imitation"
        batch = next(replay_iter)
        batch = utils.to_torch(batch, self.device)

        metrics.update(self.policy.update(batch[:-1], step))

        return metrics
