import gym
import pybullet_envs
import multiprocessing as mp


def worker(env):
    env.reset()
    env.step(env.action_space.sample())

if __name__ == "__main__":
    env1 = gym.make('HalfCheetahBulletEnv-v0')
    env2 = gym.make('HalfCheetahBulletEnv-v0')
