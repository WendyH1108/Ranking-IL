#!/usr/bin/env python3
import os
import argparse
import pathlib
import shutil
from datetime import datetime
from subprocess import Popen


class Overrides(object):
    def __init__(self):
        self.kvs = dict()

    def add(self, key, values):
        value = ",".join(str(v) for v in values)
        assert key not in self.kvs
        self.kvs[key] = value

    def cmd(self):
        cmd = []
        for k, v in self.kvs.items():
            cmd.append(f"{k}={v}")
        return cmd


def make_code_snap(experiment, slurm_dir="exp"):
    now = datetime.now()
    #snap_dir = pathlib.Path.cwd() / slurm_dir
    # TODO: Edit
    snap_dir = pathlib.Path("/home/yh374/Ranking-IL/exp")
    snap_dir /= now.strftime("%Y.%m.%d")
    snap_dir /= now.strftime("%H%M%S") + f"_{experiment}"
    snap_dir.mkdir(exist_ok=True, parents=True)

    def copy_dir(dir, pat):
        dst_dir = snap_dir / "code" /dir
        dst_dir.mkdir(exist_ok=True, parents=True)
        for f in (src_dir / dir).glob(pat):
            shutil.copy(f, dst_dir / f.name)

    dirs_to_copy = [".", "agent", "algo", "suite", "cfgs", "cfgs/agent", "cfgs/suite", "cfgs/algo"]
    # TODO: Edit
    src_dir = pathlib.Path("/home/yh374/Ranking-IL")
    for dir in dirs_to_copy:
        copy_dir(dir, "*.py")
        copy_dir(dir, "*.yaml")

    return snap_dir


# TODO: Change the parameters below to override configs
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=str)
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args()

    snap_dir = make_code_snap(args.experiment)
    print(str(snap_dir))

    overrides = Overrides()
    #overrides.add("hydra/launcher", ["submitit_slurm"])
    # TODO: Edit
    #overrides.add("hydra.launcher.partition", ["default_partition"])
    #overrides.add("hydra.sweep.dir", [str(snap_dir)])
    #overrides.add("hydra.launcher.submitit_folder", [str(snap_dir / "slurm")])
    overrides.add("experiment", [args.experiment])

    # env
    overrides.add('suite.task', values=['walker_walk'])
    overrides.add('suite.obs_type', values=['states'])

    # Base cfg
    overrides.add(key="seed", values=[1])
    overrides.add(key="replay_buffer_size", values=[1000000])
    overrides.add("agent", ["rl"])
    overrides.add("algo", ["ddpg"])
    overrides.add("suite", ["dmc"])
    overrides.add("save_video", [False])
    overrides.add("save_train_video", [False])
    overrides.add("buffer_local", [False])

    # Suite cfg
    # overrides.add("suite.num_seed_steps", [50000])
    overrides.add("suite.num_eval_episodes", [5])
    # overrides.add("suite.noop_max", [30])
    overrides.add(key='suite.num_train_steps', values=[2000000])
    # overrides.add(key="suite.term_death", values=[False])

    #cmd = ["python", str(snap_dir / "code" / "train.py"), "-m"]
    cmd = ["python", str("train.py"), "-m"]
    cmd += overrides.cmd()

    if args.dry:
        print(" ".join(cmd))
    else:
        print(cmd)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(snap_dir / "code")
        p = Popen(cmd, env=env)
        p.communicate()


if __name__ == "__main__":
    main()
