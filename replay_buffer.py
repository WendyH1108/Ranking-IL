import datetime
import io
import random
import traceback
from collections import defaultdict, deque

import numpy as np
import torch
from torch.utils.data import IterableDataset, Dataset
from dm_env import StepType
from suite.dmc import ExtendedTimeStep
import pickle as pkl
from typing import Optional


def episode_len(episode):
    # subtract -1 because the dummy first transition
    return next(iter(episode.values())).shape[0] - 1


def save_episode(episode, fn):
    with io.BytesIO() as bs:
        np.savez_compressed(bs, **episode)
        bs.seek(0)
        with fn.open("wb") as f:
            f.write(bs.read())


def load_episode(fn):
    with fn.open("rb") as f:
        episode = np.load(f)
        episode = {k: episode[k] for k in episode.keys()}
        return episode


def _worker_init_fn(worker_id):
    seed = np.random.get_state()[1][0] + worker_id
    np.random.seed(seed)
    random.seed(seed)


class ReplayBufferStorage:
    def __init__(self, data_specs, replay_dir):
        self._data_specs = data_specs
        self._replay_dir = replay_dir
        replay_dir.mkdir(exist_ok=True)
        self._current_episode = defaultdict(list)
        self._preload()

    def __len__(self):
        return self._num_transitions

    def add(self, time_step):
        for spec in self._data_specs:
            value = time_step[spec.name]
            if np.isscalar(value):
                value = np.full(spec.shape, value, spec.dtype)
            assert spec.shape == value.shape and spec.dtype == value.dtype
            self._current_episode[spec.name].append(value)
        if time_step.last():
            episode = dict()
            for spec in self._data_specs:
                value = self._current_episode[spec.name]
                episode[spec.name] = np.array(value, spec.dtype)
            self._current_episode = defaultdict(list)
            self._store_episode(episode)

    def _preload(self):
        self._num_episodes = 0
        self._num_transitions = 0
        for fn in self._replay_dir.glob("*.npz"):
            _, _, eps_len = fn.stem.split("_")
            self._num_episodes += 1
            self._num_transitions += int(eps_len)

    def _store_episode(self, episode):
        eps_idx = self._num_episodes
        eps_len = episode_len(episode)
        self._num_episodes += 1
        self._num_transitions += eps_len
        ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        eps_fn = f"{ts}_{eps_idx}_{eps_len}.npz"
        save_episode(episode, self._replay_dir / eps_fn)


class ReplayBuffer(IterableDataset):
    def __init__(
        self,
        replay_dir,
        max_size,
        num_workers,
        nstep,
        discount,
        fetch_every,
        save_snapshot,
        return_one_step,
    ):
        self._replay_dir = replay_dir
        self._size = 0
        self._max_size = max_size
        self._num_workers = max(1, num_workers)
        self._episode_fns = []
        self._episodes = dict()
        self._nstep = nstep
        self._discount = discount
        self._fetch_every = fetch_every
        self._samples_since_last_fetch = fetch_every
        self._save_snapshot = save_snapshot
        self._return_one_step = return_one_step

    def _sample_episode(self):
        eps_fn = random.choice(self._episode_fns)
        return self._episodes[eps_fn]

    def _store_episode(self, eps_fn):
        try:
            episode = load_episode(eps_fn)
        except:
            return False
        eps_len = episode_len(episode)
        while eps_len + self._size > self._max_size:
            early_eps_fn = self._episode_fns.pop(0)
            early_eps = self._episodes.pop(early_eps_fn)
            self._size -= episode_len(early_eps)
            early_eps_fn.unlink(missing_ok=True)
        self._episode_fns.append(eps_fn)
        self._episode_fns.sort()
        self._episodes[eps_fn] = episode
        self._size += eps_len

        if not self._save_snapshot:
            eps_fn.unlink(missing_ok=True)
        return True

    def _try_fetch(self):
        if self._samples_since_last_fetch < self._fetch_every:
            return
        self._samples_since_last_fetch = 0
        try:
            worker_id = torch.utils.data.get_worker_info().id
        except:
            worker_id = 0
        eps_fns = sorted(self._replay_dir.glob("*.npz"), reverse=True)
        fetched_size = 0
        for eps_fn in eps_fns:
            eps_idx, eps_len = [int(x) for x in eps_fn.stem.split("_")[1:]]
            if eps_idx % self._num_workers != worker_id:
                continue
            if eps_fn in self._episodes.keys():
                break
            if fetched_size + eps_len > self._max_size:
                break
            fetched_size += eps_len
            if not self._store_episode(eps_fn):
                break

    def _sample(self):
        try:
            self._try_fetch()
        except:
            traceback.print_exc()
        self._samples_since_last_fetch += 1
        episode = self._sample_episode()
        # add +1 for the first dummy transition
        idx = np.random.randint(0, episode_len(episode) - self._nstep + 1) + 1
        obs = episode["observation"][idx - 1]
        action = episode["action"][idx]
        next_obs = episode["observation"][idx + self._nstep - 1]
        reward = np.zeros_like(episode["reward"][idx])
        discount = np.ones_like(episode["discount"][idx])

        # 1 step next obs
        for i in range(self._nstep):
            step_reward = episode["reward"][idx + i]
            reward += discount * step_reward
            discount *= episode["discount"][idx + i] * self._discount
        if self._return_one_step:
            # This is for r(s, s') for imitation
            one_step_obs = episode["observation"][idx]
            return (obs, action, reward, discount, next_obs, one_step_obs)
        return (obs, action, reward, discount, next_obs)

    def __iter__(self):
        while True:
            yield self._sample()


class ReplayDataset(Dataset):
    def __init__(
        self,
        ep_storage: str,
        ensemble_size: int = 5,
        bootstrap: bool = False,
        with_replacement: bool = True,
        rng: Optional[int] = None,
    ):
        ep_fns = ep_storage.glob("*.npz")
        eps = [load_episode(fn) for fn in ep_fns]

        self.bootstrap = bootstrap
        self.with_replacement = with_replacement
        self.ensemble_size = ensemble_size

        self.obs = np.concatenate([e["observation"][:-1] for e in eps], axis=0)
        self.next_obs = np.concatenate([e["observation"][1:] for e in eps], axis=0)
        self.act = np.concatenate([e["action"][1:] for e in eps], axis=0)

        self.ep_len = [episode_len(e) for e in eps]
        self.len = np.sum(self.ep_len)

        if self.ensemble_size > 1:
            assert rng is not None
            self.rng = np.random.default_rng(seed=rng)
            if self.bootstrap:
                self.idxs = self.rng.choice(
                    self.len,
                    size=(ensemble_size, self.len),
                    replace=self.with_replacement,
                )
            else:
                self.idxs = np.empty((ensemble_size, self.len))
                for i in range(ensemble_size):
                    self.idxs[i] = self.rng.permutation(np.arange(self.len))
                self.idxs = self.idxs.astype("int32")

    def __len__(self):
        return self.len

    def __getitem__(self, idx):

        if self.ensemble_size > 1:
            idxs = self.idxs[:, idx]
            return self.obs[idxs], self.act[idxs], self.next_obs[idxs]

        return self.obs[idx], self.act[idx], self.next_obs[idx]


class ReplayBufferLocal(IterableDataset):
    def __init__(
        self,
        storage,
        max_size,
        num_workers,
        nstep,
        discount,
        fetch_every,
        save_snapshot,
        return_one_step,
    ):
        self._storage = storage
        self._size = 0
        self._max_size = max_size
        self._num_workers = max(1, num_workers)
        self._episode_fns = []
        self._episodes = dict()
        self._nstep = nstep
        self._discount = discount
        self._fetch_every = fetch_every
        self._samples_since_last_fetch = fetch_every
        self._save_snapshot = save_snapshot
        self._return_one_step = return_one_step

    def _sample_episode(self):
        eps_fn = random.choice(self._episode_fns)
        return self._episodes[eps_fn]

    def _store_episode(self, eps_fn):
        try:
            episode = load_episode(eps_fn)
        except:
            return False
        eps_len = episode_len(episode)
        while eps_len + self._size > self._max_size:
            early_eps_fn = self._episode_fns.pop(0)
            early_eps = self._episodes.pop(early_eps_fn)
            self._size -= episode_len(early_eps)
            early_eps_fn.unlink(missing_ok=True)
        self._episode_fns.append(eps_fn)
        self._episode_fns.sort()
        self._episodes[eps_fn] = episode
        self._size += eps_len

        if not self._save_snapshot:
            eps_fn.unlink(missing_ok=True)
        return True

    def _try_fetch(self):
        if self._samples_since_last_fetch < self._fetch_every:
            return
        self._samples_since_last_fetch = 0
        try:
            worker_id = torch.utils.data.get_worker_info().id
        except:
            worker_id = 0
        eps_fns = sorted(self._storage._replay_dir.glob("*.npz"), reverse=True)
        fetched_size = 0
        for eps_fn in eps_fns:
            eps_idx, eps_len = [int(x) for x in eps_fn.stem.split("_")[1:]]
            if eps_idx % self._num_workers != worker_id:
                continue
            if eps_fn in self._episodes.keys():
                break
            if fetched_size + eps_len > self._max_size:
                break
            fetched_size += eps_len
            if not self._store_episode(eps_fn):
                break

    def _sample(self):
        try:
            self._try_fetch()
        except:
            traceback.print_exc()
        self._samples_since_last_fetch += 1
        episode = self._sample_episode()
        # add +1 for the first dummy transition
        idx = np.random.randint(0, episode_len(episode) - self._nstep + 1) + 1
        obs = episode["observation"][idx - 1]
        action = episode["action"][idx]
        next_obs = episode["observation"][idx + self._nstep - 1]
        reward = np.zeros_like(episode["reward"][idx])
        discount = np.ones_like(episode["discount"][idx])
        for i in range(self._nstep):
            step_reward = episode["reward"][idx + i]
            reward += discount * step_reward
            discount *= episode["discount"][idx + i] * self._discount
        if self._return_one_step:
            one_step_obs = episode["observation"][idx]
            return (obs, action, reward, discount, next_obs, one_step_obs)
        return (obs, action, reward, discount, next_obs)

    def __iter__(self):
        while True:
            yield self._sample()


class ReplayBufferMemory:
    def __init__(
        self, specs, max_size, batch_size, nstep, discount, eta=None, n_samples=None
    ):
        self._specs = specs
        self._max_size = max_size
        self._batch_size = batch_size
        self._nstep = nstep
        self._discount = discount
        self._idx = 0
        self._full = False
        self._items = dict()
        self._queue = deque([], maxlen=nstep + 1)

        for spec in specs:
            self._items[spec.name] = np.empty((max_size, *spec.shape), dtype=spec.dtype)

        self._eta = eta
        self._n_samples = n_samples

        if self._eta is not None:
            assert self._n_samples is not None

            self._weights = None

    def __len__(self):
        return self._max_size if self._full else self._idx

    def get_weights(self):
        n_learners = len(self) // self._n_samples

        # Uniform Sampling
        if n_learners < 2:
            self._weights = None
            return

        # This is the case when n_learners = 2
        if self._weights is None:
            uniform_weights = np.full(
                (len(self)), 1 / self._n_samples, dtype=np.float32
            )
            uniform_weights[: self._n_samples] *= 1 - self._eta
            uniform_weights[self._n_samples : self._idx] *= self._eta
            self._weights = uniform_weights
            return

        # Polyak Averaging for every weak_learner added
        self._weights *= 1 - self._eta
        new_weights = np.full(
            (self._n_samples), self._eta / self._n_samples, dtype=np.float32
        )
        if not self._full:
            self._weights = np.concatenate([self._weights, new_weights])
            return

        self._weights[self._idx - self._n_samples : self._idx] = new_weights
        self._weights[self._idx : self._idx + self._n_samples] /= self._eta

    def get_buffer(self):
        return self._items, self._full, self._idx

    def load_buffer(self, items, full, idx):
        self._items = items
        self._full = full
        self._idx = idx
        self._queue = deque([], maxlen=self._nstep + 1)

    def add(self, time_step):
        for spec in self._specs:
            if spec.name == "next_observation":
                continue
            value = time_step[spec.name]
            if np.isscalar(value):
                value = np.full(spec.shape, value, spec.dtype)
            # assert spec.shape == value.shape and spec.dtype == value.dtype

        self._queue.append(time_step)
        if len(self._queue) == self._nstep + 1:
            np.copyto(self._items["observation"][self._idx], self._queue[0].observation)
            np.copyto(self._items["action"][self._idx], self._queue[1].action)
            np.copyto(
                self._items["next_observation"][self._idx], self._queue[-1].observation
            )
            reward, discount = 0.0, 1.0
            self._queue.popleft()
            for ts in self._queue:
                reward += discount * ts.reward
                discount *= ts.discount * self._discount
            np.copyto(self._items["reward"][self._idx], reward)
            np.copyto(self._items["discount"][self._idx], discount)

            self._idx = (self._idx + 1) % self._max_size
            self._full = self._full or self._idx == 0

        if time_step.last():
            self._queue.clear()

    def _sample(self):
        if not self._eta:
            idxs = np.random.randint(0, len(self), size=self._batch_size)
        else:
            idxs = np.random.choice(
                np.arange(len(self)), size=self._batch_size, p=self._weights
            )
        batch = tuple(self._items[spec.name][idxs] for spec in self._specs)
        return batch

    def __iter__(self):
        while True:
            yield self._sample()


class ReplayWrapper:
    def __init__(
        self,
        train_env,
        data_specs,
        work_dir,
        cfg,
        buffer_name="buffer",
        return_one_step=True,
    ):
        self.cfg = cfg
        if cfg.buffer_local:
            self.replay_storage = ReplayBufferStorage(
                data_specs, work_dir / buffer_name
            )

            max_size_per_worker = cfg.replay_buffer_size // max(
                1, cfg.replay_buffer_num_workers
            )

            iterable = ReplayBufferLocal(
                self.replay_storage,
                max_size_per_worker,
                cfg.replay_buffer_num_workers,
                cfg.nstep,
                cfg.suite.discount,
                fetch_every=1000,
                save_snapshot=True,
                return_one_step=return_one_step,
            )

            self.replay_buffer = torch.utils.data.DataLoader(
                iterable,
                batch_size=cfg.batch_size,
                num_workers=cfg.replay_buffer_num_workers,
                pin_memory=True,
                worker_init_fn=_worker_init_fn,
            )
        else:
            self.replay_buffer = ReplayBufferMemory(
                specs=train_env.specs(),
                max_size=cfg.replay_buffer_size,
                batch_size=cfg.batch_size,
                nstep=cfg.nstep,
                discount=cfg.suite.discount,
            )

    def add(self, elt):
        if self.cfg.buffer_local:
            self.replay_storage.add(elt)
        else:
            self.replay_buffer.add(elt)


class ExpertReplayBuffer(IterableDataset):
    """
    Used for Adversarial IL type algorithms
    """

    def __init__(self, dataset_path, num_demos):
        # Load Expert Demos
        with open(dataset_path, "rb") as f:
            data = pkl.load(f)
            # obs, act = data[0], data[-2]
            obs, act = np.array(data["states"]), np.array(data["actions"])
        self.obs = obs
        self.act = act
        self._episodes = []
        for i in range(num_demos):
            episode = dict(observation=obs[i], action=act[i])
            self._episodes.append(episode)
        self.num_demos = num_demos

    def _sample_episode(self):
        idx = np.random.randint(0, self.num_demos)
        return self._episodes[idx]

    def _sample(self):
        episode = self._sample_episode()
        idx = np.random.randint(0, episode_len(episode)) + 1
        obs = episode["observation"][idx]
        action = episode["action"][idx]
        # if len(action.shape) == 3:
        #     action = np.squeeze(action, axis=1)
        if idx == len(episode["observation"]) - 1:
            next_obs = np.zeros_like(episode["observation"][0])
        else:
            next_obs = episode["observation"][idx + 1]
        return (obs, action, next_obs)

    def __iter__(self):
        while True:
            yield self._sample()


def make_expert_replay_loader(dataset_path, num_demos, batch_size):
    iterable = ExpertReplayBuffer(dataset_path, num_demos)
    loader = torch.utils.data.DataLoader(
        iterable,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=True,
        worker_init_fn=_worker_init_fn,
    )

    return loader


def make_replay_loader(
    replay_dir,
    max_size,
    batch_size,
    num_workers,
    save_snapshot,
    nstep,
    discount,
    return_one_step,
):
    max_size_per_worker = max_size // max(1, num_workers)

    iterable = ReplayBuffer(
        replay_dir,
        max_size_per_worker,
        num_workers,
        nstep,
        discount,
        fetch_every=1000,
        save_snapshot=save_snapshot,
        return_one_step=return_one_step,
    )

    loader = torch.utils.data.DataLoader(
        iterable,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=_worker_init_fn,
    )
    return loader


# if __name__ == '__main__':
# pass
# demos_path = "/home/yh374/Ranking-IL/expert_v2/cheetah_run_10.pkl"
# num_demos = 10
# batch_size = 2
# expert_loader = make_expert_replay_loader(
#             demos_path, num_demos,batch_size)
# print("pass loader")

# expert_iter = iter(expert_loader)

# # print(next(expert_iter))
# expert_obs, actions, expert_obs_next = next(expert_iter)
# print("expert_obs", expert_obs.shape)
# print("actions",actions.shape)
# print("expert_obs_next",expert_obs_next.shape)
