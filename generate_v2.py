import warnings
import os
import hydra
import numpy as np
import torch
import utils
import pickle as pkl
from pathlib import Path
from video import VideoRecorder
from snapshot_paths import snapshots
from collections import defaultdict

import yaml

warnings.filterwarnings("ignore", category=DeprecationWarning)
torch.backends.cudnn.benchmark = True

os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MUJOCO_GL"] = "egl"


@hydra.main(config_path="cfgs", config_name="config_gen")
def main(cfg):
    work_dir = Path.cwd()
    print(f"workspace: {work_dir}")

    utils.set_seed_everywhere(cfg.seed)
    device = torch.device(cfg.device)

    # create envs
    env = hydra.utils.call(cfg.suite.task_make_fn_eval)

    save_dir = Path(cfg.save_generation_dir)

    env_dir = save_dir / cfg.suite.task
    env_dir.mkdir(parents=True, exist_ok=True)
    video = VideoRecorder(env_dir if cfg.save_video else None)

    snapshot = snapshots[cfg.suite.task] + "/snapshot.pt"
    print(snapshot)

    # with open(snapshot, "rb") as f:
    # payload = torch.load(f)
    payload = torch.load(snapshot)

    for k, v in payload.items():
        if k == "agent":
            agent = v

    traj = defaultdict(list)
    eval_returns_list = list()
    print(agent)

    for i in range(cfg.num_gen_episodes):
        eval_return = 0
        states = list()
        actions = list()
        next_states = list()
        rewards = list()
        dones = list()

        # Start episode
        time_step = env.reset()
        video.init(env, enabled=True)
        while not time_step.last():

            states.append(time_step.observation)

            with torch.no_grad(), utils.eval_mode(agent.policy):
                action = agent.act(
                    time_step.observation, 1000000, eval_mode=True
                )
            time_step = env.step(action)

            next_states.append(time_step.observation)
            rewards.append(time_step.reward)
            actions.append(
                np.array([time_step.action])
            )  # For shaping reasons adding dim
            dones.append(bool(time_step.last() == True))

            video.record(env)
            eval_return += time_step.reward

        traj["states"].append(tuple(states))
        traj["actions"].append(tuple(actions))
        traj["next_states"].append(tuple(next_states))
        traj["rewards"].append(tuple(rewards))
        traj["dones"].append(tuple(dones))
        traj["lengths"].append(len(states))

        eval_returns_list.append(eval_return)

        print(f"Eval Return: {eval_return} | Traj Len: {len(states)}")

    rew = np.array(eval_returns_list)
    metadata = {
        "entire_db": {
            "mean": float(rew.mean()),
            "std": float(rew.std()),
            "min": float(rew.min()),
            "max": float(rew.max()),
        },
        "10_db": {
            "mean": float(rew[:10].mean()),
            "std": float(rew[:10].std()),
            "min": float(rew[:10].min()),
            "max": float(rew[:10].max()),
        },
    }
    print(
        f"all: Mean | STD | Min | Max: {rew.mean()} | {rew.std()} | {rew.min()} | {rew.max()}"
    )
    print(
        f"10: Mean | STD | Min | Max: {rew[:10].mean()} | {rew[:10].std()} | {rew[:10].min()} | {rew[:10].max()}"
    )

    # Save Metadata
    metadata_path = env_dir / "metadata.yml"
    with open(metadata_path, "w") as f:
        yaml.dump(metadata, f)

    # Save DB
    snapshot_path = save_dir / f"{cfg.suite.task}_{cfg.num_gen_episodes}.pkl"
    with open(str(snapshot_path), "wb") as f:
        pkl.dump(traj, f)


if __name__ == "__main__":
    main()
