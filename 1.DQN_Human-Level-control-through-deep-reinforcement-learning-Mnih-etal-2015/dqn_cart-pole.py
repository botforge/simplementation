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
        Simple Replay Memory implementation for low-dimensional data observations (eg. Classic Control envs) 

        NOTE: This implementation does not "stack frames" together as described by <agent_history_length> hyperparamter (https://www.nature.com/articles/nature14236.pdf), Mnih et al. 2015
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
        super(Simple_ReplayMemory, self).__init__()

        self.max_size, self.batch_size, self.obs_space_shape = size, batch_size,obs_space_shape

        #Pre-allocate arrays that represent the state of the Replay Memory
        self.observations = np.empty((self.max_size, obs_space_shape), dtype=np.float32)
        self.actions = np.empty((self.max_size, 1), dtype=np.int32)
        self.rewards = np.empty((self.max_size, 1), dtype=np.float32)
        self.done_flags = np.empty((self.max_size, 1), dtype=np.bool)

        self.count = 0 
        self.idx = 0 

    def add(self, obs, action, reward, done):
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
        done: boolean
            Boolean stating whether the episode terminated (also from calling env.step(action) on obs)
        """

        #Update state of Replay Memory to include new information
        self.observations[self.count, ...] = obs
        self.actions[self.count, 0] = action
        self.rewards[self.count, 0] = reward
        self.done_flags[self.count, 0] = done
        self.count = max(self.count, self.idx+1) 
        self.idx = (self.idx+1) % self.max_size #if self.idx == self.max_size, resets to zero

    def sample(self):
        """ Sample a random minibatch of states
        (default implementation is batch_size=32 & env=CartPole-v1)

        Returns:
        -------
        obs_minibatch: np.array
            Shape (<self.batch_size>, <self.obs_space_shape>) - default implementation : (32, 4)
        actions_minibatch: np.array
            Shape (<self.batch_size>, <env.action_space.n>) - default implementation : (32,1)
        rewards_minibatch: np.array
            Shape (<self.batch_size>, 1) - default implementation: (32,1)
        next_obs_minibatch: np.array
            Same as obs_minibatch
        done_minibatch: np.array
            Shape (<self.batch_size>, 1) - default implementation: (32,1)
        """

        #Sample array of random indicies [0, 1, 2,...., self.count-1] without repetition
        sampling_idxs = np.array(random.sample(range(0, self.count), self.batch_size))
        next_idxs = sampling_idxs + 1
        invalid_next_idxs = np.nonzero(next_idxs >= self.count)[0]
        next_idxs[invalid_next_idxs] = -1

        obs_minibatch = self.observations[sampling_idxs, ...]

        #Fix next_obs_minibatch by zeroing out invalid observations
        next_obs_minibatch = self.observations[next_idxs, ...]
        next_obs_minibatch[invalid_next_idxs, ...] = np.zeros(self.obs_space_shape)

        actions_minibatch = self.actions[sampling_idxs, ...]
        rewards_minibatch = self.rewards[sampling_idxs, ...]
        done_minibatch = self.done_flags[sampling_idxs, ...]
        return obs_minibatch, next_obs_minibatch, actions_minibatch, rewards_minibatch, done_minibatch

def _weights_init(m):
    if hasattr(m, 'weight'):
        nn.init.kaiming_uniform_(m.weight, a=0, nonlinearity='relu')
    if hasattr(m, 'bias'):
        nn.init.constant(m.bias, 0)

class Simple_DQN(nn.Module):
    def __init__(self, obs_space_shape, action_space_shape):
        """
        Neural Network that predicts the Q-value for all actions "a_t" given an input state "s_t" for low dimensional action space
        The Q-Value is the output function Q(s_t, a_t) that estimates the expected future reward of taking an action "a_t" in state "s_t"
        For the CartPole problem, this function will predict the expected future reward of "moving left" and "moving right" given 4 joint states.
        Visit: http://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-7.pdf for more info

        Parameters
        ----------
        obs_space_shape:int 
            The size of the vector representing the observation space (for Cartpole = 4)
        action_space_shape:int
            The size of the vector representing the action space (for Cartpole = 2)
        """
        super(Simple_DQN, self).__init__()

        #Build a simple Linear modell
        self.model = nn.Sequential(
            nn.Linear(obs_space_shape, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_space_shape)
        )

        #He Initialization
        self.apply(_weights_init)

    def forward(self, obs):
        """
        Run a forward pass through Neural Network
        
        Parameters
        ----------
        obs: np.array
            Observations of shape (batch_size, <obs_space_shape>)
        
        Returns
        -------
        act: np.array
            Actions of shape (batch_size, <action_space_shape>)
        """
        return self.model(obs)
    
def main():
    #Make OpenAI gym environment (we only consider discrete binary action spaces)
    env = gym.make('CartPole-v1')
    obs_space_shape = env.observation_space.shape[0]
    action_space_shape = env.action_space.n

    #Make Q-network and Target Q-network
    q_net = 

    #Set random seeds
    seed = random.randint(0, 9999)
    torch.manual_seed(seed)
    if torch.cuda.is_available:
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    env.seed(seed)

    #We alter the default hyperparameters presented in Mihn et al. due to low-dimensionality of the Cart-Pole environment
    replay_memory = Simple_ReplayMemory(size=10000, batch_size=32, obs_space_shape=obs_space_shape)
    
    
if __name__ == "__main__":
    main()

