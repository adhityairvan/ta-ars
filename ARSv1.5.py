# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 03:30:25 2020

@author: Adhitya Irvansyah
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 17:38:56 2019

@author: Adhitya Irvansyah
"""

import numpy as np
from gym import wrappers
import gym
import pybullet_envs
import os
import time

import multiprocessing as mp
from multiprocessing.managers import BaseManager

class Hp():
    """ Class for saving all hyperparameter values"""
    def __init__(self, nb_steps = 5, episode_length = 1000, learning_rate= 0.02, nb_directions = 16, nb_best_directions = 16, noise = 0.03, seed = 1, env_name = "HalfCheetahBulletEnv-v0", shift = 0):
        self.nb_steps = nb_steps
        self.episode_length = episode_length
        self.learning_rate = learning_rate
        self.nb_directions = nb_directions
        self.nb_best_directions = nb_best_directions
        assert self.nb_directions >= self.nb_best_directions
        self.noise = noise
        self.seed = seed
        self.env_name = env_name
        self.shift = shift

class Policy():
    """ Policy class handles all policy related task liked giving output to some input, returning random samples and updating policy value"""
    def __init__(self, hp, input_size, output_size):
        self.tetha = np.zeros((output_size, input_size))
        self.hp = hp
    def evaluate(self, input, delta = None, direction = None):
        """ Evaluate input value against theta/weight matrix and return output matrix"""
        if direction is None:
            return self.tetha.dot(input)
        elif direction == 'positive':
            return (self.tetha + self.hp.noise * delta).dot(input)
        else:
            return (self.tetha - self.hp.noise * delta).dot(input)
        
    def sample_deltas(self):
        """ generate random samples"""
        return [ np.random.randn(*self.tetha.shape) for _ in range(self.hp.nb_directions)]
    
    def update(self, rollouts, sigma_r):
        """updating theta or weight matrix value"""
        step = np.zeros(self.tetha.shape)
        
        for r_pos, r_neg, d in rollouts:
            step +=   (r_pos - r_neg) * d
        
        self.tetha += self.hp.learning_rate / (self.hp.nb_best_directions * sigma_r) * step

#Normalizer Class
class Normalizer(object):
    """class for normalizer object to normalize input/observation value"""
    def __init__(self, nb_inputs):
        self.n = np.zeros(nb_inputs)
        self.mean = np.zeros(nb_inputs)
        self.mean_diff = np.zeros(nb_inputs)
        self.var = np.zeros(nb_inputs)
    
    def observe(self, x):
        """ observe new input value and calculate appropriate mean and variance"""
        self.n += 1.
        last_mean = self.mean.copy()
        self.mean += (x - self.mean) /self.n
        self.mean_diff += (x - last_mean) * (x - self.mean)
        self.var = (self.mean_diff / self.n).clip(min = 1e-2)
    
    def normalize(self, inputs):
        """ normalize input"""
        obs_mean = self.mean
        obs_std = np.sqrt(self.var)
        return (inputs - obs_mean) / obs_std

class ARS():
    """ ars class. Main class to ars algorithm"""
    def __init__(self, hp, monitor_dir):
        self.env = gym.make(hp.env_name)        
        self.env = wrappers.Monitor(self.env, monitor_dir, force = True, video_callable=capped_cubic_video_schedule)
        self.hp = hp
        nb_inputs = self.env.observation_space.shape[0]
        nb_outputs = self.env.action_space.shape[0]
        self.policy = Policy(hp, nb_inputs, nb_outputs)
        
        # create lock manager to share normalizer object to all multiprocess task
        BaseManager.register('Normalizer',Normalizer)
        manager = BaseManager()
        manager.start()
        self.normalizer = manager.Normalizer(nb_inputs)
        
    def train(self):
        file = open('log_reward', 'w')
        """ train agent"""
        envs = [gym.make(self.hp.env_name) for i in range(self.hp.nb_directions)]
        for step in range(self.hp.nb_steps):
            self.pool = mp.Pool()
            #generate random pertrutbation
            deltas = self.policy.sample_deltas()
            #create variable to save the rewards
            positive_rewards = [0] * self.hp.nb_directions
            negative_rewards = [0] * self.hp.nb_directions
            
            positive_rewards = self.pool.starmap(explore,[(envs[i], self.hp, self.normalizer, self.policy, "positive", deltas[i]) for i in range(self.hp.nb_directions)])
            negative_rewards = self.pool.starmap(explore,[(envs[i], self.hp, self.normalizer, self.policy, "negative", deltas[i]) for i in range(self.hp.nb_directions)])
            
            self.pool.close()
            self.pool.join()
            
            #gathering the rewards
            all_rewards = np.array(positive_rewards + negative_rewards)
            
            #get the standard deviation of all rewards
            sigma_r = all_rewards.std()
            
            #soring the rewards to generate rollouts for updating weight
            scores = { k:max(r_pos, r_neg) for k,(r_pos,r_neg) in enumerate(zip(positive_rewards, negative_rewards))} 
            order = sorted(scores.keys(), key = lambda x: scores[x], reverse = True)[:self.hp.nb_best_directions]
            rollouts = [(positive_rewards[k], negative_rewards[k], deltas[k]) for k in order]
            
            #update the policy with new weight
            self.policy.update(rollouts, sigma_r)
            #print result 
            reward_evaluation = explore(self.env, self.hp, self.normalizer, self.policy)
            print("Step ", step, "=> Reward: ", reward_evaluation)
            file.write(str(reward_evaluation)+'\n')
        return self.policy.tetha

# function to explore the env in one way, this can be called in paralel by specifying lock
def explore(env_name, hp, normalizer, policy, direction = None, delta = None):
    """explore environment to one specific theta value"""
    # if(isinstance(env_name, str)):
    #     env = gym.make(env_name)
    # else:
    env = env_name
    state = env.reset()
    done = False
    num_plays = 0.
    sum_rewards = 0
    while not done and num_plays < hp.episode_length:
        # lock the shared normalizer object before doing calculation
        lock.acquire()
        normalizer.observe(state)
        state = normalizer.normalize(state)
        lock.release()
        action = policy.evaluate(state, delta, direction)
        state, reward, done, _ = env.step(action)
        
        sum_rewards += (reward - hp.shift)
        num_plays += 1
    if isinstance(env_name,str):
        env.close()
    return sum_rewards
        
# utillity function to create base dir
def mkdir(base, name):
    """ helper function to create result directory"""
    path = os.path.join(base,name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path

# utility function, overriding video record scheduling
def capped_cubic_video_schedule(episode_id):
    if episode_id < 1000:
        return int(round(episode_id ** (1. / 3))) ** 3 == episode_id
    else:
        return episode_id % 1000 == 0

#main code to run all the training

def main():
    work_dir = mkdir('exp', 'brs')
    monitor_dir = mkdir(work_dir, 'monitor')
    hp = Hp(800, 1000, nb_directions = 20, nb_best_directions = 5, env_name = "HopperBulletEnv-v0", shift = 1)
    ars = ARS(hp, monitor_dir)
    start = time.time()
    ars.train()
    print(time.time() - start)
    
lock = mp.Lock()
if __name__ == "__main__":
    main()