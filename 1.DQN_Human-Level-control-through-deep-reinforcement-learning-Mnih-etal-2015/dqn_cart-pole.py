import gym
from torch import nn, optim
import torch.nn.functional as F
import torch
import numpy as np
import math, random
import pdb
import logging
logger = logging.getLogger(__name__)
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

class ReplayMemory(object):
    def __init__(self, size=1000000, agent_history_length=4, batch_size=32):
        """
        Simple Replay Memory implementation as described in Minh et al. 2015
        Stores transitions (curr_obs, action_t, reward_t, next_obs, done)
        Hyperparameters in Appendix for more info: https://www.nature.com/articles/nature14236.pdf

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer
        agent_history_length: int
            Number of transitions observations to stack together for a state
        batch_size: int
            Number of states returned in a minibatch
        """
        self.size, self.batch_size, self.agent_history_length = size, batch_size, agent_history_length

def main():
    #Create replay memory 
    replay_memory = ReplayMemory()

#%%
if __name__ == "__main__":
    main()

