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

class Simple_ReplayMemory(object):
    def __init__(self, size=10000, batch_size=32, obs_space_shape = 4):
        """
        Simple Replay Memory implementation for low-dimensional data observations (eg. Classic Controle envs)

        NOTE: This implementation does not "stack frames" together as described by <agent_history_length> hyperparamter (https://www.nature.com/articles/nature14236.pdf), Mihn et al. 2015
        Stacking frames is unneccesary for low dimensional observations, like CartPole which is simply of type Box(4,)

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer
        batch_size: int
            Number of states returned in a minibatch
        obs_space_shape: int
            The size of the vector representing the observation space 
        """
        self.size, self.batch_size, self.obs_space_shape = size, batch_size,obs_space_shape

        self.memory = [] #stores the observations
        self.count = 0 #keep track of how many experiences you have
        self.idx = 0 #and which experience you're on

        #Pre-allocate arrays that return samples
        self.obs_minibatch = np.empty((self.batch_size, self.obs_space_shape))
        self.actions_minibatch = np.empty((self.batch_size,))
        self.rewards_minibatch = np.empty((self.batch_size, ))
        self.next_obs_minibatch = self.obs_minibatch.copy()
        self.done_minibatch = np.empty((self.batch_size, ))


    def add(self, obs, action, reward, next_obs, done):
        """
        Add an experience to replay memory
        Parameters
        ----------
        obs: np.array 
            An observation from the Gym environment of shape (<self.obs_space_shape>,) (For Cart-Pole this is (4, ))
        action: np.array
            An action from Gym environment of shape (<env.action_space.n>, )
        reward: float
            Reward for calling env.step(action) on obs
        next_obs: np.array
            Next observation from calling env.step(action) on obs
        done: boolean
            Boolean stating whether the episode terminated (also from calling env.step(action) on obs)
        """
        data = (obs, action, reward, next_obs, done)

        #Checks if current index is initialized in memory (if it has been initialized, overwrite)
        if self.idx >= len(self.memory):
            self.memory.append(data)
        else:
            self.memory[self.idx] = data

        self.count = max(self.count, self.idx+1) 
        self.idx = (self.idx+1) % self.size #if self.idx == self.size, resets to zero
        

    def sample(self):
        """ Sample a random minibatch of states
        (default implementation is batch_size=32 & env=CartPole-v1)
        Returns:
        -------
        obs_minibatch: np.array
            Shape (<self.batch_size>, <agent_history_length>, <self.obs_space_shape>) - default implementation : (32, 4, 1) 
        actions_minibatch: np.array
            Shape (<self.batch_size>, <env.action_space.n>) - default implementation : (32,1)
        rewards_minibatch: np.array
            Shape (<self.batch_size>, ) - default implementation: (32,1)
        next_obs_minibatch: np.array
            Same as obs_minibatch
        done_minibatch: np.array
            Shape (<self.batch_size>, ) - default implementation: (32,1)
        """

        #Sample array of random indicies [agent_history_length-1, agent_history_length, ...., self.count] without repetition
        sampling_idxs = random.sample(range(self.agent_history_length-1, self.count), self.batch_size) 

        for i, sample_idx in enumerate(sampling_idxs):

            #Stack current sampled observation (sample_idx) + the past <self.agent_history_length> sampled observations into the minibatch
            start_idx = sample_idx + 1 - self.agent_history_length
            end_idx = sample_idx + 1
            history_samples = self.memory[start_idx:end_idx] #returns a list of tuples (obs, action, reward, next_obs, done) of length <self.agent_history_length>

            #Iterate through list and update minibatch with history samples
            for j, data_sample in enumerate(history_samples):
                self.obs_minibatch[i,j,...] = data_sample[0]
                self.next_obs_minibatch[i,j,...] = data_sample[3]

            #The action, rewards, and done minibatches don't care about stacking history samples. We can directly index into the current sample
            curr_sample = self.memory[sample_idx]
            self.actions_minibatch[i] = curr_sample[1]
            self.rewards_minibatch[i] = curr_sample[2]
            self.done_minibatch[i] = curr_sample[4]

            return self.obs_minibatch, self.next_obs_minibatch, self.actions_minibatch[None], self.rewards_minibatch[None], self.done_minibatch[None]
            
def main():
    #Make OpenAI gym environment
    env = gym.make('CartPole-v1')
    obs_space_shape = env.observation_space.shape[0]
    #We alter the default hyperparameters presented in Mihn et al. due to simplicity of the Cart-Pole environment
    replay_memory = ReplayMemory(size=10000, agent_history_length=1, batch_size=32, obs_space_shape=obs_space_shape)
    

if __name__ == "__main__":
    main()

