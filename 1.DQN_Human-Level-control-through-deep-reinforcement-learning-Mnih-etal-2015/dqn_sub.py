import pdb
import math, random

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import deque

dhruv_replaymemory = True

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
        nn.init.kaiming_uniform_(m.weight, a=0, nonlinearity='relu')
    if hasattr(m, 'bias'):
        nn.init.constant_(m.bias, 0)

class DQN(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(DQN, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, env.action_space.n)
        )

        # #He Initialization
        # self.apply(_weights_init)        

    def forward(self, x):
        return self.layers(x)
    
    def act(self, state, epsilon):
        if random.random() > epsilon:
            state   = Variable(torch.FloatTensor(state).unsqueeze(0), volatile=True)
            q_value = self.forward(state)
            action  = q_value.max(1)[1].data[0].item()
        else:
            action = random.randrange(env.action_space.n)
        return action

def epsilon_at_t(t):
    epsilon_start = 1.0
    epsilon_final = 0.01
    epsilon_decay = 500

    epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)
    return epsilon_by_frame(t)

def compute_td_loss(batch_size):
    state, next_state, action, reward, done = replay_buffer.sample()

    state      = Variable(torch.FloatTensor(np.float32(state))) #32,4
    next_state = Variable(torch.FloatTensor(np.float32(next_state)), volatile=True) #32,4
    action     = Variable(torch.LongTensor(action))#32, 1
    reward     = Variable(torch.FloatTensor(reward))#32
    done       = Variable(torch.FloatTensor(done))#32
    q_values      = model(state) #32,2
    next_q_values = model(next_state)

    q_value          = q_values.gather(1, action).squeeze(1)
    next_q_value     = next_q_values.max(1)[0]
    expected_q_value = reward + gamma * next_q_value * (1 - done)
    
    loss = (q_value - Variable(expected_q_value.data)).pow(2).mean()
        
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss

seed = 6582
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

env_id = "CartPole-v1"
env = gym.make(env_id)
env = gym.wrappers.Monitor(env, './dddd', video_callable=False, force=True)
env.seed(seed)

model = DQN(env.observation_space.shape[0], env.action_space.n)

if USE_CUDA:
    model = model.cuda()
    
optimizer = optim.Adam(model.parameters())

replay_buffer = Simple_ReplayMemory(size=1000, batch_size=32, obs_space_shape=4)

num_frames = 10000
batch_size = 32
gamma      = 0.99

losses = []
all_rewards = []
episode_reward = 0

state = env.reset()
for frame_idx in range(1, num_frames + 1):
    epsilon = epsilon_at_t(frame_idx)
    action = model.act(state, epsilon)
    next_state, reward, done, _ = env.step(action)
    replay_buffer.add(state, next_state, action, reward, done)

    state = next_state
    episode_reward += reward
    
    if done:
        state = env.reset()
        all_rewards.append(episode_reward)
        episode_reward = 0
        
    if frame_idx > batch_size:
        loss = compute_td_loss(batch_size)
        losses.append(loss.item())
    if frame_idx % 200 == 0:
        pass
        #plot(frame_idx, all_rewards, losses)
    episode_rewards = env.get_episode_rewards()
    print(frame_idx, "  ", np.mean(episode_rewards[-100:]))