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
    def __init__(self, size=1000000, agent_history_length=4, batch_size=32, obs_space_shape = 4):
        """
        Memory Efficient Replay Memory implementation as described in Minh et al. 2015

        Default Hyperparameters in Appendix for more info: https://www.nature.com/articles/nature14236.pdf
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer
        agent_history_length: int
            Number of transitions observations to stack together for a state
        batch_size: int
            Number of states returned in a minibatch
        obs_space_shape: int
            The size of the vector representing the observation space 
        """
        self.size, self.batch_size, self.agent_history_length, self.obs_space_shape = size, batch_size, agent_history_length, obs_space_shape
        
        #Pre-allocate arrays that represent the state of the Replay Memory
        self.observations = np.empty((self.size, obs_space_shape), dtype=np.float32)
        self.actions = np.empty(self.size, dtype=np.int32)
        self.rewards = np.empty(self.size, dtype=np.float32)
        self.done_flags = np.empty(self.size, dtype=np.bool)

        self.count = 0 #keep track of how many experiences you have
        self.idx = 0 #and which experience you're on

        #Pre-allocate arrays that return samples
        self.obs_minibatch = np.empty((self.batch_size, self.agent_history_length, self.obs_space_shape))
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

        #Update state of Replay Memory to include new information
        self.observations[self.count, ...] = obs
        self.actions[self.count] = action
        self.rewards[self.count] = reward
        self.done_flags[self.count] = done
        self.count = max(self.count, self.idx+1) 
        self.idx = (self.idx+1) % self.size #if self.idx == self.size, resets to zero
        
    def sample(self):
        """ Sample a random minibatch of states
        (default implementation is batch_size=32 & env=CartPole-v1)
        Returns:
        -------
        obs_minibatch: np.array
            Shape (<self.batch_size>, <agent_history_length>, <self.obs_space_shape>) - default implementation : (32, 1, 4) 
        actions_minibatch: np.array
            Shape (<self.batch_size>, <env.action_space.n>) - default implementation : (32,)
        rewards_minibatch: np.array
            Shape (<self.batch_size>, ) - default implementation: (32,)
        next_obs_minibatch: np.array
            Same as obs_minibatch
        done_minibatch: np.array
            Shape (<self.batch_size>, ) - default implementation: (32,)
        """

        #Sample array of random indicies [agent_history_length-1, agent_history_length, ...., self.count] without repetition
        sampling_idxs = random.sample(range(self.agent_history_length-1, self.count), self.batch_size) 
        
        zero_array = np.zeros((self.agent_history_length, self.obs_space_shape))

        for i, sample_idx in enumerate(sampling_idxs):
            #Add current sampled observation (sample_idx) and the past <self.agent_history_length> sampled observations as a sample in the minibatch
            start_idx = sample_idx + 1 - self.agent_history_length
            end_idx = sample_idx + 1

            self.obs_minibatch[i, ...] = self.observations[start_idx:end_idx, ...]
            self.next_obs_minibatch[i, ...] = self.observations[start_idx + 1:end_idx + 1]
            self.actions_minibatch[i] = self.actions[sample_idx]
            self.rewards_minibatch[i] = self.rewards[sample_idx]
            self.done_minibatch[i] = self.done_flags[sample_idx]
            #in this case, there is no next_obs -> just append zeros
            
def main():
    #Make OpenAI gym environment
    env = gym.make('CartPole-v1')
    obs_space_shape = env.observation_space.shape[0]
    #We alter the default hyperparameters presented in Mihn et al. due to simplicity of the Cart-Pole environment
    replay_memory = ReplayMemory(size=10000, agent_history_length=1, batch_size=32, obs_space_shape=obs_space_shape)
    

if __name__ == "__main__":
    main()

