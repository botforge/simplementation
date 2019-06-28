import pdb
import gym
from torch import nn, optim
import torch.nn.functional as F
import torch
import numpy as np
import math, random
from collections import deque
import logging
logger = logging.getLogger(__name__)
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

class Simple_ReplayMemory(object):
    def __init__(self, size=10000, batch_size=32, obs_space_shape = 4):
        """
        Simple Replay Memory implementation for low-dimensional data observations (eg. Classic Control envs) 

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
        self.memory = deque(maxlen=self.max_size)
        self.count = 0 #keep track of the number of elements

    def add(self, obs, next_obs, action, reward, done):
            """
            Add an experience to replay memory
            Parameters
            ----------
            obs: np.array 
                An observation from the Gym environment of shape (<self.obs_space_shape>,) (For Cart-Pole this is (4, ))
            next_obs: np.array 
                The next observation from the Gym environment of shape (<self.obs_space_shape>,) (For Cart-Pole this is (4, ))
            action: np.array
                An action from Gym environment of shape (<env.action_space.n>, )
            reward: float
                Reward for calling env.step(action) on obs
            done: boolean
                Boolean stating whether the episode terminated (also from calling env.step(action) on obs)
            """

            obs      = np.expand_dims(obs, 0).astype(np.float32) #turns to (1, 4) for Cart-Pole
            next_obs = np.expand_dims(next_obs, 0).astype(np.float32)
            self.count = min(self.max_size, self.count + 1)
            self.memory.append((obs, action, reward, next_obs, done))

    def sample(self):
        """ Sample a random minibatch of states
        (default implementation is batch_size=32 & env=CartPole-v1)

        Returns:
        -------
        obs_minibatch: np.array
            Shape (<self.batch_size>, <self.obs_space_shape>) - default implementation : (32, 4)
        actions_minibatch: np.array
            Shape (<self.batch_size>, <env.action_space.n>) - default implementation : (32,1)
        rewards_minibatch: tuple
            len = <self.batch_size>- default implementation: len = 32
        next_obs_minibatch: np.array
            Same as obs_minibatch
        done_minibatch: tuple
            len = <self.batch_size>- default implementation: len = 32
        """
        obs_minibatch, actions_minibatch, rewards_minibatch, next_obs_minibatch, done_minibatch = zip(*random.sample(self.memory, self.batch_size))

        actions_minibatch = np.expand_dims(actions_minibatch, 1)
        return np.concatenate(obs_minibatch), np.concatenate(next_obs_minibatch), actions_minibatch, rewards_minibatch, done_minibatch

def _weights_init(m):
    if hasattr(m, 'weight'):
        nn.init.xavier_uniform_(m.weight)
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
        # self.apply(_weights_init)

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
    env = gym.make('CartPole-v1')
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
    optimizer = optim.Adam(qnet.parameters())
    num_frames = 50000
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
        replay_memory.add(obs, next_obs, action, reward, done)

        obs = next_obs
        if done:
            obs = env.reset()

        #Populate Replay Memory with <replay_start_size> experiences before learning
        if t > replay_start_size:
            #---------------------------------------------
            #Sample batch & compute loss & update network (Lines 12 - 15)
            #---------------------------------------------
            obs_minibatch, next_obs_minibatch, actions_minibatch, rewards_minibatch, done_minibatch = replay_memory.sample()

            ts_obs, ts_rewards, ts_next_obs, ts_done = map(lambda x: torch.FloatTensor(x).to(device), [obs_minibatch, rewards_minibatch, next_obs_minibatch, done_minibatch])
            ts_actions = torch.LongTensor(actions_minibatch).to(device)

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