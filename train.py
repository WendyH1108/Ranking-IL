import datetime

import os
from pathlib import Path
import hydra
import numpy as np
import torch
from dm_env import specs

import utils
from logger import Logger
from replay_buffer import ReplayWrapper, make_expert_replay_loader, ReplayBufferMemory
from video import VideoRecorder

from time import time
import wandb
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = ""
torch.backends.cudnn.benchmark = True


# TODO: Handle imitation learning elements (i.e. cost updates, adding IRL cost to RL pipeline, expert data handling etc)
class Workspace:
    def __init__(self, cfg):
        # NOTE: seems like hydra is changing default behavior.... come back later
        self.work_dir = Path.cwd()
        print(f"workspace: {self.work_dir}")

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.setup()
        self.agent = hydra.utils.instantiate(cfg.agent)
        if cfg.load_checkpoint:
            chkpt = (
                Path(
                    "/private/home/jdchang/state-il/exp/2022.07.05/132235_collect_checkpoints/0"
                )
                / f"{cfg.checkpoint_path}_snapshot.pt"
            )
            self.load_policy(chkpt)

        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0
        self._best_eval_return = -float("inf")

    def setup(self):

        # create logger/video recorder
        self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb)
        self.video = VideoRecorder(self.work_dir if self.cfg.save_video else None)

        # create envs
        self.train_env = hydra.utils.call(self.cfg.suite.task_make_fn_train)
        self.eval_env = hydra.utils.call(self.cfg.suite.task_make_fn_eval)
        obs_spec = self.train_env.observation_spec()
        action_spec = self.train_env.action_spec()

        self.cfg.algo.obs_shape = obs_spec.shape
        n_action = None
        if type(action_spec) is specs.BoundedArray:
            self.cfg.algo.action_dim = action_spec.shape[0]
            n_action = self.cfg.algo.action_dim
        elif type(action_spec) is specs.DiscreteArray:
            self.cfg.algo.num_actions = action_spec.num_values
            n_action = self.cfg.algo.num_actions
        else:
            raise NotImplementedError("Env Action Spec not supported")

        print(
            f"Initialized Environment\nTask: {self.cfg.suite.task}\nObs Shape: {self.cfg.algo.obs_shape}\nAction Dim: {n_action}"
        )

        self.buffer = ReplayBufferMemory(
            specs=self.train_env.specs(),
            max_size=self.cfg.replay_buffer_size,
            batch_size=self.cfg.batch_size,
            nstep=self.cfg.nstep,  # only works for 1 for now.....
            discount=self.cfg.discount,
        )

        # TODO: set flags to turn on and off for pixels/state/rl vs il etc....
        # Map loader rather than iterable since we would want all
        if self.cfg.agent.name == "dac" or self.cfg.agent.name == "boosting":
            demos_path = self.cfg.agent.expert_dir + self.cfg.suite.task + "_10.pkl"
            self.expert_loader = make_expert_replay_loader(
                demos_path, self.cfg.agent.num_demos, self.cfg.agent.batch_size
            )
            self.expert_iter = iter(self.expert_loader)

        if self.cfg.agent.name == "boosting":
            self.disc_buffer = ReplayBufferMemory(
                specs=self.train_env.specs(),
                max_size=self.cfg.replay_buffer_size,
                batch_size=self.cfg.batch_size,
                nstep=self.cfg.nstep,
                discount=self.cfg.discount,
                eta=self.cfg.eta,
                n_samples=self.cfg.n_samples,
            )
            self._disc_replay_iter = None
        self._replay_iter = None

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.suite.action_repeat

    @property
    def best_eval_return(self):
        return self._best_eval_return

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            # self._replay_iter = iter(self.buffer.replay_buffer)
            self._replay_iter = iter(self.buffer)
        return self._replay_iter

    @property
    def disc_replay_iter(self):
        if self._disc_replay_iter is None:
            self._disc_replay_iter = iter(self.disc_buffer)
        return self._disc_replay_iter

    # TODO: Figure out subsampling......
    def collect_samples(self):
        episode = 0
        while episode < self.cfg.agent.n_sample_episodes:
            time_step = self.eval_env.reset()
            time_steps = [time_step]

            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent.policy):
                    action = self.agent.act(
                        time_step.observation, self.global_step, eval_mode=False
                    )
                time_step = self.eval_env.step(action)
                time_steps.append(time_step)

            episode += 1

            for ts in time_steps:
                self.disc_buffer.add(ts)

    def eval(self):
        step, episode, total_reward = 0, 0, 0
        states = list()
        actions = list()
        next_states = list()
        time_list = list()
        while episode < self.cfg.suite.num_eval_episodes:
            time_step = self.eval_env.reset()
            # self.video.init(self.eval_env, enabled=True)  #1 NOTE: for every?
            ep_rew = 0

            # add/////////////
            traj_states = list()
            traj_actions = list()
            traj_next_states = list()
            traj_time = list()
            # add/////////////
            while not time_step.last():
                traj_time.append(time_step)
                traj_states.append(np.array(time_step.observation))
                with torch.no_grad(), utils.eval_mode(self.agent.policy):
                    action = self.agent.act(
                        time_step.observation, self.global_step, eval_mode=True
                    )
                time_step = self.eval_env.step(action)

                # add/////////////

                traj_next_states.append(np.array(time_step.observation))
                traj_actions.append(np.array([action]))
                # add/////////////

                # self.video.record(self.eval_env)
                ep_rew += time_step.reward
                step += 1
            # add/////////////
            states.append(np.array(traj_states))
            time_list.append(traj_time)
            actions.append(np.squeeze(np.array(traj_actions), axis=1))
            next_states.append(np.array(traj_next_states))
            # add/////////////
            total_reward += ep_rew
            episode += 1
            # self.video.save(f"{self.global_frame}_{ep_rew}.mp4")

        with self.logger.log_and_dump_ctx(self.global_frame, ty="eval") as log:
            log("episode_return", total_reward / episode)
            log("episode_length", step * self.cfg.suite.action_repeat / episode)
            log("episode", self.global_episode)
            log("step", self.global_step)
            log("total_time", self.timer.total_time())

        return total_reward / episode, (
            np.array(states),
            np.array(actions),
            np.array(next_states),
        )

    def train(self):
        episode_step, episode_reward = 0, 0
        time_step = self.train_env.reset()
        time_steps = [time_step]
        metrics = None
        eval_counter = 0
        divergence = 0
        # while self.global_step < self.cfg.suite.num_train_steps:
        for _ in range(self.cfg.suite.num_train_steps):
            if time_step.last():
                # Compute IL Rewards
                if repr(self.agent) == "offline_imitation":
                    time_steps, imitation_return = self.agent.compute_offline_rewards(
                        time_steps, self.global_episode
                    )

                # Add to buffer
                for ts in time_steps:
                    self.buffer.add(ts)

                self._global_episode += 1
                if metrics is not None:
                    # log stats
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step * self.cfg.suite.action_repeat
                    with self.logger.log_and_dump_ctx(
                        self.global_frame, ty="train"
                    ) as log:
                        log("fps", episode_frame / elapsed_time)
                        log("total_time", total_time)
                        log("episode_reward", episode_reward)
                        log("episode_length", episode_frame)
                        log("episode", self.global_episode)
                        log("step", self.global_step)
                        # if repr(self.agent) == "offline_imitation":
                        #     log("episode_imitation_return", imitation_return)
                    wandb.log(
                        {
                            "train/fps": episode_frame / elapsed_time,
                            "train/total_time": total_time,
                            "train/episode_reward": episode_reward,
                            "train/episode_length": episode_frame,
                            "train/episode": self.global_episode,
                            "train/global_step": self.global_step,
                        }
                    )

                # reset env
                time_step = self.train_env.reset()
                time_steps = [time_step]

                episode_step, episode_reward = 0, 0

                # Save Snapshot
                if self.cfg.suite.save_snapshot:
                    self.save_snapshot()

            # Eval
            if self.global_step % self.cfg.suite.eval_every_steps == 0:
                eval_return, on_policy_data = self.eval()
                # if eval_return > self.best_eval_return:
                #     ret = int(eval_return)
                #     self.save_snapshot(f"{ret}_snapshot.pt")
                on_policy_data = utils.to_torch(on_policy_data, self.device)
                divergence = self.agent.compute_divergence(
                    self.expert_loader, on_policy_data
                )
                eval_counter += 1

            # ========== BOOSTING ==========
            if (
                self.cfg.agent.name == "boosting"
                and self.global_step % self.cfg.policy_iter
            ):
                # Add Samples
                self.collect_samples()
                # Update Disc
                disc_metrics = self.agent.update_discriminator(
                    self.disc_replay_iter, self.expert_iter
                )

                wandb.log(disc_metrics)

                # Reset Policy
                if self.cfg.agent.reset_policy:
                    self.agent.reset_policy()

                # Reset Buffer
                self.buffer.load_buffer(*self.disc_buffer.get_buffer())
                self._replay_iter = None

            with torch.no_grad(), utils.eval_mode(self.agent.policy):
                action = self.agent.act(
                    time_step.observation, self.global_step, eval_mode=False
                )

            # Update Agent
            if self.global_step >= self.cfg.suite.num_seed_steps:

                if self.cfg.agent.name == "dac":
                    metrics = self.agent.update(
                        self.eval_env,
                        self.buffer,
                        self.replay_iter,
                        self.expert_loader,
                        self.expert_iter,
                        self.global_step,
                    )
                elif self.cfg.agent.name == "boosting":
                    metrics = self.agent.update(
                        self.replay_iter, self.expert_iter, self.global_step
                    )
                else:
                    metrics = self.agent.update(self.replay_iter, self.global_step)

                metrics["eval/custom_step"] = eval_counter
                metrics["eval/divergence"] = divergence
                self.logger.log_metrics(metrics, self.global_frame, ty="train")
                wandb.log(metrics)

            # Env Step
            time_step = self.train_env.step(action)
            time_steps.append(time_step)
            episode_reward += time_step.reward
            episode_step += 1
            self._global_step += 1

    def save_snapshot(self, name="snapshot.pt"):
        snapshot = self.work_dir / name
        keys_to_save = [
            "agent",
            "timer",
            "_global_step",
            "_global_episode",
            "_best_eval_return",
        ]
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open("wb") as f:
            torch.save(payload, f)

    def load_snapshot(self):
        snapshot = self.work_dir / "snapshot.pt"
        with snapshot.open("rb") as f:
            payload = torch.load(f)
        for k, v in payload.items():
            self.__dict__[k] = v

    def load_policy(self, snapshot):
        with snapshot.open("rb") as f:
            payload = torch.load(f)
        policy = payload["agent"].policy
        self.agent.policy = policy


@hydra.main(version_base=None, config_path="./cfgs", config_name="config")
def main(cfg):
    w = Workspace(cfg)
    project_name = "Ranking-IL"
    entity = "kaiwenw_rep_offline_rl"
    ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    name = f"{ts}"
    snapshot = w.work_dir / "snapshot.pt"
    if snapshot.exists():
        print(f"resuming: {snapshot}")
        w.load_snapshot()
    if cfg.wandb:
        with wandb.init(project=project_name, entity=entity, name=name) as run:
            wandb.define_metric("eval/custom_step")
            wandb.define_metric("eval/*", step_metric="eval/custom_step")
            wandb.define_metric("train/global_step")
            wandb.define_metric("train/*", step_metric="train/global_step")
            w.train()
    else:
        w.train()


if __name__ == "__main__":
    main()
