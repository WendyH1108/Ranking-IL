import numpy as np
from collections import deque
from typing import Any, NamedTuple

import dm_env
import cv2
from gym.envs.atari import AtariEnv
from dm_env import specs, StepType


# TODO: make sure not to use this for other experiments than breakout
class BreakoutTimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    handcraft: Any

    def first(self) -> bool:
        return self.step_type == StepType.FIRST

    def mid(self) -> bool:
        return self.step_type == StepType.MID

    def last(self) -> bool:
        return self.step_type == StepType.LAST


def getBreakoutFeatures(state):
    """
    Returns features for ball's x position, ball's y position, and paddle's x position.
    """

    BOTTOM_BLOCK_ROW = 93
    TOP_PADDLE_ROW = 189
    SCREEN_L = 8
    SCREEN_R = 152
    MIDDLE_X = (SCREEN_L + SCREEN_R) / 2
    MIDDLE_Y = (BOTTOM_BLOCK_ROW + TOP_PADDLE_ROW) / 2

    features = dict()
    # get possible paddle positions (get from pixel image by color and location)
    #paddle_xpos = state[TOP_PADDLE_ROW, SCREEN_L:SCREEN_R, 0]
    paddle_xpos = state[TOP_PADDLE_ROW, SCREEN_L:SCREEN_R]
    # find the first non-zero value in the list (i.e. the first non-black pixel in that row)
    # that is the leftmost position of the paddle
    # else, give the paddle the position of the middle of the screen
    features["paddlex"] = next(
        (i for i, x in enumerate(paddle_xpos) if x != 0), MIDDLE_X
    )

    # get possible ball x positions between the bottom block row and top paddle row
    #ball_xpos = np.sum(
    #    state[BOTTOM_BLOCK_ROW:TOP_PADDLE_ROW, SCREEN_L:SCREEN_R, 0], axis=0
    #)
    ball_xpos = np.sum(
        state[BOTTOM_BLOCK_ROW:TOP_PADDLE_ROW, SCREEN_L:SCREEN_R], axis=0
    )
    # find the first non-zero value in the list (i.e. the first non-black pixel in that row)
    # that is the leftmost position of the ball
    # else, give the ball the position of the middle of the screen in the x-direction
    features["ballx"] = next((i for i, x in enumerate(ball_xpos) if x != 0), MIDDLE_X)

    # get the possible y positions of the ball given where we know the x position to be
    #ball_ypos = np.sum(
    #    state[BOTTOM_BLOCK_ROW:TOP_PADDLE_ROW, SCREEN_L:SCREEN_R, 0], axis=1
    #)
    ball_ypos = np.sum(
        state[BOTTOM_BLOCK_ROW:TOP_PADDLE_ROW, SCREEN_L:SCREEN_R], axis=1
    )
    # find the first non-zero value in the list (i.e. the first non-black pixel in that row)
    # that is the topmost position of the ball
    # else, give the ball the position of the middle of the screen in the y-direction
    features["bally"] = next((i for i, x in enumerate(ball_ypos) if x != 0), MIDDLE_Y)

    # discretize the feature space
    # tested for various discretizations
    features["paddlex"] = features["paddlex"] / 32
    features["ballx"] = features["ballx"] / 32
    features["bally"] = features["bally"] / 50

    rep = np.array([features["paddlex"], features["ballx"], features["bally"]])
    return rep


class Atari(dm_env.Environment):
    def __init__(
        self,
        game,
        frame_skip,
        seed,
        sticky_actions=False,
        terminal_on_life_loss=False,
        noop_max=30,
        screen_size=84,
    ):

        self._env = AtariEnv(
            game=game,
            obs_type="image",
            frameskip=1,
            repeat_action_probability=0.25 if sticky_actions else 0.0,
        )

        self._frame_skip = frame_skip
        self._terminal_on_life_loss = terminal_on_life_loss
        self._screen_size = screen_size

        obs_space = self._env.observation_space
        action_space = self._env.action_space
        self._obs_spec = specs.BoundedArray(
            shape=(1, screen_size, screen_size),
            dtype=np.uint8,
            minimum=0,
            maximum=255,
            name="observation",
        )

        self._action_spec = specs.DiscreteArray(
            num_values=action_space.n, dtype=np.int64, name="action"
        )

        self._screen_buffer = [
            np.empty(obs_space.shape[:2], dtype=np.uint8),
            np.empty(obs_space.shape[:2], dtype=np.uint8),
        ]
        self._lives = 0
        self._rng = np.random.default_rng(seed)
        self.noop_max = noop_max
        assert self._env.unwrapped.get_action_meanings()[0] == "NOOP"

    def _fetch_grayscale_observation(self, buffer):
        self._env.ale.getScreenGrayscale(buffer)
        return buffer

    def _pool_and_resize(self):
        # pool if there are enough screens to do so.
        if self._frame_skip > 1:
            np.maximum(
                self._screen_buffer[0],
                self._screen_buffer[1],
                out=self._screen_buffer[0],
            )

        image = cv2.resize(
            self._screen_buffer[0],
            (self._screen_size, self._screen_size),
            interpolation=cv2.INTER_LINEAR,
        )
        image = np.asarray(image, dtype=np.uint8)
        return np.expand_dims(image, axis=0)

    def reset(self):
        self._env.reset()

        # add random number of noops to produce non-deterministic start state
        noops = self._rng.integers(0, self.noop_max + 1)
        if self.noop_max == 0:
            assert noops == 0
        for t in range(noops):
            obs, _, _, _ = self._env.step(0)
            if t >= noops - 2:
                nt = t - (noops - 2)
                self._fetch_grayscale_observation(self._screen_buffer[nt])

        self._lives = self._env.ale.lives()
        obs = self._pool_and_resize()

        # NOTE: ONLY FOR BREAKOUT
        # handcraft = getBreakoutFeatures(self._screen_buffer[0])
        # return BreakoutTimeStep(StepType.FIRST, 0.0, 1.0, obs, handcraft)
        return dm_env.TimeStep(StepType.FIRST, 0.0, 1.0, obs)

    def render(self, mode="rgb_array"):
        return self._env.render(mode)

    def step(self, action):
        total_reward = 0.0
        for t in range(self._frame_skip):
            _, reward, game_over, _ = self._env.step(action)
            total_reward += reward

            if self._terminal_on_life_loss:
                new_lives = self._env.ale.lives()
                done = game_over or new_lives < self._lives
                self._lives = new_lives
            else:
                done = game_over

            if done:
                break
            elif t >= self._frame_skip - 2:
                nt = t - (self._frame_skip - 2)
                self._fetch_grayscale_observation(self._screen_buffer[nt])

        obs = self._pool_and_resize()
        if done:
            return dm_env.termination(total_reward, obs)
        return dm_env.transition(total_reward, obs)

        #handcraft = getBreakoutFeatures(self._screen_buffer[0])
        #if done:
        #    return BreakoutTimeStep(StepType.LAST, total_reward, 0.0, obs, handcraft)
        #return BreakoutTimeStep(StepType.MID, total_reward, 1.0, obs, handcraft)

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._action_spec

    def reward_spec(self):
        return specs.Array(shape=(), name="reward", dtype=np.float32)

    def discount_spec(self):
        return specs.Array(shape=(), name="discount", dtype=np.float32)


class FrameStack(dm_env.Environment):
    def __init__(self, env, k):
        self._env = env
        self._k = k
        self._frames = deque([], maxlen=k)

        obs_shape = env.observation_spec().shape
        self._obs_spec = specs.BoundedArray(
            shape=np.concatenate([[obs_shape[0] * k], obs_shape[1:]], axis=0),
            dtype=np.uint8,
            minimum=0,
            maximum=255,
            name="observation",
        )

    def _transform_observation(self, time_step):
        assert len(self._frames) == self._k
        obs = np.concatenate(list(self._frames), axis=0)
        return time_step._replace(observation=obs)

    def reset(self):
        time_step = self._env.reset()
        pixels = time_step.observation
        for _ in range(self._k):
            self._frames.append(pixels)
        return self._transform_observation(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        pixels = time_step.observation
        self._frames.append(pixels)
        return self._transform_observation(time_step)

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._env.action_spec()

    def reward_spec(self):
        return self._env.reward_spec()

    def discount_spec(self):
        return self._env.discount_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)

    def specs(self):
        obs_spec = self.observation_spec()
        action_spec = self.action_spec()
        next_obs_spec = specs.Array(obs_spec.shape, obs_spec.dtype, "next_observation")
        reward_spec = specs.Array((), action_spec.dtype, "reward")
        discount_spec = specs.Array((), action_spec.dtype, "discount")
        return (obs_spec, action_spec, reward_spec, discount_spec, next_obs_spec)


class TimeLimit(dm_env.Environment):
    def __init__(self, env, max_steps):
        self._env = env
        self._max_steps = max_steps
        self._elapsed_steps = 0

    def step(self, action):
        time_step = self._env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_steps:
            # truncate episode, but preserve original discount
            return time_step._replace(step_type=dm_env.StepType.LAST)
        return time_step

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def reset(self):
        self._elapsed_steps = 0
        return self._env.reset()

    def reward_spec(self):
        return self._env.reward_spec()

    def discount_spec(self):
        return self._env.discount_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ExtendedTimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    action: Any
    #handcraft: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        return getattr(self, attr)


class ExtendedTimeStepWrapper(dm_env.Environment):
    def __init__(self, env):
        self._env = env

    def reset(self):
        time_step = self._env.reset()
        return self._augment_time_step(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._augment_time_step(time_step, action)

    def _augment_time_step(self, time_step, action=None):
        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
        return ExtendedTimeStep(
            observation=time_step.observation,
            step_type=time_step.step_type,
            action=action,
            reward=time_step.reward,
            discount=time_step.discount,
            #handcraft=time_step.handcraft,
        )

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def reward_spec(self):
        return self._env.reward_spec()

    def discount_spec(self):
        return self._env.discount_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


def make(
    name,
    frame_stack=4,
    frame_skip=4,
    action_repeat=4,
    seed=1,
    term_death=False,
    noop_max=30,
):
    # we only include action_repeat arg for compatibility with dmc.make function
    assert action_repeat == frame_skip
    env = Atari(
        name, frame_skip, seed, terminal_on_life_loss=term_death, noop_max=noop_max
    )
    env = TimeLimit(env, max_steps=27000)
    env = FrameStack(env, k=frame_stack)
    env = ExtendedTimeStepWrapper(env)
    return env
