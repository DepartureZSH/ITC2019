import os
import torch
import random
import torch.nn.functional as F
import numpy as np

class Scheduler:
    def __init__(self, obs_dim, action_dim, config):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.output_dir = config['config']['output']
        self.gamma = config['train']['sched']['gamma']
        self.lr = config['train']['sched']['learning_rate']
        self.max_epsilon = config['train']['sched']['max_epsilon']
        self.min_epsilon = config['train']['sched']['min_epsilon']
        self.decay_rate = config['train']['sched']['decay_rate']
        self.Qtable = np.zeros((self.obs_dim, self.action_dim)) # ind/state-position
        
    def take_action(self, positions, masks):
        # epsilon_greedy_policy
        actions = [pos for ind, pos in enumerate(positions)]
        p2i = {pos:ind for ind, pos in enumerate(positions)}
        poses = [pos for pos, mask in zip(positions, masks) if mask==0]
        for ind, (pos, mask) in enumerate(zip(positions, masks)):
            if mask == 1: continue
            exploration_exploitation_tradeoff = random.uniform(0, 1)
            if exploration_exploitation_tradeoff > self.epsilon:
                action = np.argmax(self.Qtable[ind])
            else:
                action = random.randint(0, pos - 1)
            if action in poses:
                continue
            poses.append(action)
            actions[ind] = action
            actions[p2i[action]] = pos
        return actions

    def update(self, buffers):
        for ind in range(self.obs_dim):
            state = buffers["states"][ind]
            action = buffers["actions"][ind]
            reward = buffers["rewards"][ind]
            new_state = buffers["next_states"][ind]
            self.Qtable[ind, action] += self.lr * (reward + self.gamma * np.max(self.Qtable[:, action]) - self.Qtable[ind, action]) 

    def set_epsilon(self, episode):
        self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-self.decay_rate * episode)
        return self.epsilon
    
    def save(self, pname):
        np.save(f'{self.output_dir}/{pname}/{pname}.npy', self.Qtable)
    
    def load(self, pname):
        self.Qtable = np.load(f'{self.output_dir}/{pname}/{pname}.npy')
