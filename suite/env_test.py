import gym
import envpool
import numpy as np
from time import time
from tqdm import tqdm

#from gym.envs.atari import AtariEnv
#from gym.wrappers import FrameStack
from atari import make


gym_env = make('breakout')
def run(env, n_env):
    state = env.reset()
    #n = env.action_space.n
    #n = env.action_space.n
    n = 4
    start = time()
    steps = int(100000/n_env)
    for _ in tqdm(range(steps)):
        act = np.random.randint(0, n, size=(n_env))
        #state, reward, done, info = env.step(act)
        timestep = env.step(act)
        print(timestep.last())
    end = time()
    print(f'Time : {end - start}')

def run_v2(env, n_env):
    timestep = env.reset()
    #n = env.action_space.n
    #n = env.action_space.n
    n = 4
    start = time()
    steps = int(100000/n_env)
    for _ in tqdm(range(steps)):
        if timestep.last():
            timestep = env.reset()
        act = np.random.randint(0, n, size=(n_env))
        timestep = env.step(act[0])
    end = time()
    print(f'Time : {end - start}')

#run_v2(gym_env, 1)
pool_env = envpool.make_dm("Breakout-v5", num_envs=10, episodic_life=True, max_episode_steps=27000)
run(pool_env, 10)
