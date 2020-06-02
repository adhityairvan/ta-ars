import ray
import gym
import pybullet_envs

@ray.remote
class Worker:
    def __init__(self,env, hp):
        self.env = env
        self.hp = hp
        pass
    def explore(self,normalizer, policy, direction = None, delta = None):
        """explore environment to one specific pertubation value"""
        state = self.env.reset()
        done = False
        num_plays = 0.
        sum_rewards = 0
        while not done and num_plays < self.hp.episode_length:
            state = ray.get(normalizer.observe.remote(state))
            action = policy.evaluate(state, delta, direction)
            state, reward, done, _ = self.env.step(action)
            
            sum_rewards += (reward - self.hp.shift)
            num_plays += 1
        return sum_rewards