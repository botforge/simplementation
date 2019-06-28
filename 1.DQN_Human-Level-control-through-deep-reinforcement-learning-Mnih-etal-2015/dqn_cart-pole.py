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
        self.observations[self.idx, ...] = obs
        self.actions[self.idx, 0] = action
        self.rewards[self.idx, 0] = reward
        self.done_flags[self.idx, 0] = done
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
        done_minibatch: np.array (uint8)
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
        return obs_minibatch, next_obs_minibatch, actions_minibatch, rewards_minibatch, done_minibatch.astype(np.float32)

def _weights_init(m):
    if hasattr(m, 'weight'):
        nn.init.kaiming_uniform_(m.weight, a=0, nonlinearity='relu')
    if hasattr(m, 'bias'):
        nn.init.constant_(m.bias, 0)

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
        obs: tensor
            Observations of shape (batch_size, <obs_space_shape>)
        
        Returns
        -------
        act: tensor
            Actions of shape (batch_size, <action_space_shape>)
        """
        return self.model(obs)
    
def epsilon_at_t(t):
    """
    Defines "epsilon" for frame "t" for epsilon-greedy exploration strategy that follows a piecewise function
    W/ probability "epsilon", we choose a random action at - otherwise we choose at=argmax_at[Q(st, at)] (Read Mnih et al. 2015)

    Parameters
    ----------
    t: int
        Frame number (Frames encountered later in training have higher frame nums.)

    Returns
    -------
    epsilon: float
        Defines the parameter for epsilon-greedy exploration
    """
    epsilon = 0
    function_type = 'exp'
    if function_type == 'lin':
        lt = 700
        rt = 4000
        #Start off always exploring
        if t < lt:
            epsilon = 1
        #Linearly decrease exploration param
        elif t >= lt and t < rt:
            alpha = float(t - lt) / (rt - lt)
            epsilon = 1 + alpha * (0.01 - 1)
        #Fix a very low exploration param for t > 4000
        else:
            epsilon = 0.01
    elif function_type == 'exp':
        factor = 500
        epsilon = 0.01 + (1 - 0.01) * math.exp(-1. * t / factor)
    elif function_type == 'decay':
        decay = 0.4
        epsilon = 0.01 + (1 - 0.01) * (decay ** t)
    return epsilon

def main():
    #Make OpenAI gym environment (we only consider discrete binary action spaces)
    env = gym.make('CartPole-v0')
    env = gym.wrappers.Monitor(env, './data/ting.pkl', video_callable=False, force=True)
    obs_space_shape = env.observation_space.shape[0]
    action_space_shape = env.action_space.n
    
    #Set random seeds
    seed = 6582
    torch.manual_seed(seed)
    if torch.cuda.is_available:
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    env.seed(seed)

    #Initialize Replay Memory (Line 1)
    replay_memory = Simple_ReplayMemory(size=1000, batch_size=32, obs_space_shape=obs_space_shape)

    #Make Q-network and Target Q-network (Lines 2 & 3) 
    qnet = Simple_DQN(obs_space_shape, action_space_shape).to(device)
    target_qnet = Simple_DQN(obs_space_shape, action_space_shape).to(device)
    target_qnet.load_state_dict(qnet.state_dict())

    #Training Parameters (Changes from Mnih et al. outlined in README.md)
    optimizer = optim.Adam(qnet.parameters(), lr=1e-3)
    num_frames = 100000
    gamma = 0.99
    replay_start_size = 33
    target_network_update_freq = 10
    
    #Train
    obs = env.reset()
    for t in range(num_frames):
        epsilon = epsilon_at_t(t)
        #-------------------------------------------------------------------
        #Take one step in the environment & add to Replay Memory (Line 7-11)
        #-------------------------------------------------------------------
        torch.set_grad_enabled(False)
        #Select action with epsilon-greedy exploration (Line 7,8)
        if random.random() > epsilon:
            ts_obs = torch.from_numpy(obs.astype(np.float32)).view(1, -1).to(device)
            ts_qvals = qnet(ts_obs)
            action = ts_qvals.max(-1)[1].item()
        else:
            action = random.randint(0, action_space_shape-1)
        torch.set_grad_enabled(True)
        #Execute action and get reward + next_obs (Line 9, 10)
        next_obs, reward, done, _ = env.step(action)

        #Store transition in Replay Memory
        replay_memory.add(obs, action, reward, done)
        obs = next_obs
        if done:
            obs = env.reset()

        #Populate Replay Memory with <replay_start_size> experiences before learning
        if t > replay_start_size:
            #---------------------------------------------
            #Sample batch & compute loss & update network (Lines 12 - 15)
            #---------------------------------------------
            obs_minibatch, next_obs_minibatch, actions_minibatch, rewards_minibatch, done_minibatch = replay_memory.sample()

            ts_obs, ts_actions, ts_rewards, ts_next_obs, ts_done = map(lambda x: torch.from_numpy(x).to(device), [obs_minibatch, actions_minibatch, rewards_minibatch, next_obs_minibatch, done_minibatch])

            ts_actions = ts_actions.long() 

            torch.set_grad_enabled(False)
            #Compute Target Values 
            ts_next_qvals = target_qnet(ts_next_obs) #(32, 2)
            ts_next_action = ts_next_qvals.argmax(-1, keepdim=True) #(32, 1)
            ts_next_action_qvals = ts_next_qvals.gather(-1, ts_next_action).view(-1) #(32,)
            ts_target_q = ts_rewards.view(-1) + gamma * ts_next_action_qvals * (1 - ts_done)
            torch.set_grad_enabled(True)

            #Compute predicted 
            ts_pred_q = qnet(ts_obs).gather(-1, ts_actions).view(-1) #(32,)

            loss = F.mse_loss(ts_pred_q, ts_target_q)
            #Compute Huber Loss (Also smooth L1 Loss) (Line 14)
            # loss = F.smooth_l1_loss(ts_pred_q, ts_target_q)

            #Perform gradient descent (Line 14)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #Update target network ever <target_network_update_freq> steps (Line 15)
            if t % target_network_update_freq == 0:
                target_qnet.load_state_dict(qnet.state_dict())
        #Log stuff
        episode_rewards = env.get_episode_rewards()
        print('Timesteps', t,'Mean Reward', np.mean(episode_rewards[-100:]))
if __name__ == "__main__":
    main()