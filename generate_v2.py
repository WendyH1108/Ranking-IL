import warnings
import os
import hydra
import numpy as np
import torch
import utils
import pickle as pkl
from pathlib import Path
from video import VideoRecorder
#import argparse

# from snapshot_paths import snapshots
from collections import defaultdict

import yaml

snapshots={"cheetah_run": "/home/yh374/Ranking-IL/exp_local/2022.11.29/ddpg_cheetah_run",
"walker_walk": "/home/yh374/Ranking-IL/exp_local/2022.11.29/ddpg_walker_walk2",
"quadruped_walk": "/home/yh374/Ranking-IL/exp_local/2022.11.29/ddpg_quadruped_walk",
"humanoid_stand": "/home/yh374/Ranking-IL/exp/2022.11.27/1406_humanoid_stand/0"}

warnings.filterwarnings("ignore", category=DeprecationWarning)
torch.backends.cudnn.benchmark = True

os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MUJOCO_GL"] = "egl"


@hydra.main(version_base=None, config_path="cfgs", config_name="config_gen")
def main(cfg):
    #parser = argparse.ArgumentParser()
    #args = parser.parse_args()

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
    
    # need to change !!!!
    print(cfg.suite.task)
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
        pixels = list()
        next_pixels = list()
        actions = list()
        next_states = list()
        rewards = list()
        dones = list()

        # Start episode
        time_step = env.reset()
        video.init(env, enabled=True)
        while not time_step.last():

            states.append(time_step.observation["state"])
            pixels.append(time_step.observation["pixels"])
            with torch.no_grad(), utils.eval_mode(agent.policy):
                action = agent.act(
                    time_step.observation["state"].astype(np.float32), 1000000, eval_mode=True
                )
            time_step = env.step(action)

            next_states.append(time_step.observation["state"])
            next_pixels.append(time_step.observation["pixels"])
            rewards.append(time_step.reward)
            actions.append(
                np.array([time_step.action])
            )  # For shaping reasons adding dim
            dones.append(bool(time_step.last() == True))

            video.record(env)
            eval_return += time_step.reward

        traj["states"].append(np.stack(states, 0))
        traj["pixels"].append(np.stack(pixels, 0))
        traj["next_pixels"].append(np.stack(next_pixels, 0))
        traj["actions"].append(np.stack(actions, 0))
        traj["next_states"].append(np.stack(next_states, 0))
        traj["rewards"].append(np.stack(rewards, 0))
        traj["dones"].append(np.stack(dones, 0))
        traj["lengths"].append(len(states))

        eval_returns_list.append(eval_return)

        print(f"Eval Return: {eval_return} | Traj Len: {len(states)}")
    video.save(f"{cfg.suite.task}.mp4")
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
