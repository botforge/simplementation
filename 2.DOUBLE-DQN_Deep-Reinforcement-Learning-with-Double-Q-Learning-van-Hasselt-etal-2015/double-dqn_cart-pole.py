import gym
from torch import nn, optim
import torch.nn.functional as F
import torch
import numpy as np
import math, random
from datetime import datetime
from collections import deque

now = datetime.now()
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

class Simple_ReplayMemory(object):
    def __init__(self, max_size=10000, batch_size=32):
        """
        Simple Replay Memory implementation for low-dimensional data observations (eg. Classic Control envs) 
        obs_space_shape: Integer, env.observation_space.shape[0] = 4 for CartPole
        """
        super(Simple_ReplayMemory, self).__init__()

        self.max_size, self.batch_size = max_size, batch_size
        self.memory = deque(maxlen=self.max_size)
        self.count = 0 #keep track of the number of elements

    def add(self, obs, next_obs, action, reward, done):
            """
            Add an experience to replay memory
            obs: np.array, shape (obs_space_shape, ) = (4, ) for CartPole
            next_obs: np.array, shape (obs_space_shape, ) = (4, ) for CartPole
            action: np.array, shape (env.action_space.n, ) = (2, ) for CartPole
            reward: float , done: boolean
            """
            obs = np.expand_dims(obs, 0).astype(np.float32) #turns to (1, 4) for Cart-Pole
            next_obs = np.expand_dims(next_obs, 0).astype(np.float32)
            self.count = min(self.max_size, self.count + 1)
            self.memory.append((obs, action, reward, next_obs, done))

    def sample(self):
        """ Sample a random minibatch of states
        (default implementation is batch_size=32 & env=CartPole-v1)

        obs_minibatch: np.array shape (self.batch_size, obs_space_shape) = (32, 4)
        actions_minibatch: np.array shape (self.batch_size, env.action_space.n) = (32,1)
        rewards_minibatch: tuple len = self.batch_size = 32
        next_obs_minibatch: np.array- same as obs_minibatch
        done_minibatch: tuple len = self.batch_size = 32
        """
        obs_minibatch, actions_minibatch, rewards_minibatch, next_obs_minibatch, done_minibatch = zip(*random.sample(self.memory, self.batch_size))
        actions_minibatch = np.expand_dims(actions_minibatch, 1)

        return np.concatenate(obs_minibatch), np.concatenate(next_obs_minibatch), actions_minibatch, rewards_minibatch, done_minibatch

def _weights_init(m):
    if hasattr(m, 'weight'):
        nn.init.kaiming_uniform_(m.weight)
    if hasattr(m, 'bias'):
        nn.init.constant_(m.bias, 0)

class Simple_DQN(nn.Module):
    def __init__(self, obs_space_shape, action_space_shape):
        """
        Neural Network that predicts the Q-value for all actions "a_t" given an input state "s_t" for low dimensional action space
        Visit: http://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-7.pdf 
        obs_space_shape:int, env.observation_space.shape[0] = (for Cartpole = 4)
        action_space_shape:int, env.action_space.n = (for Cartpole = 2)
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

        #He Initialization (Paper applies it, works better without it)
        # self.apply(_weights_init)

    def forward(self, obs):
        """
        obs: tensor, shape (batch_size, <obs_space_shape>)
        Returns tensor shape (batch_size, <action_space_shape>)
        """
        return self.model(obs)
    
def epsilon_at_t(t):
    """
    Defines "epsilon" for frame "t" for epsilon-greedy exploration strategy 
    W/ probability "epsilon", we choose a random action at - otherwise we choose at=argmax_at[Q(st, at)] (Read Mnih et al. 2015)
    t: int, Frame number (Frames encountered later in training have higher frame nums.) 
    Returns epsilon: float 
    """
    epsilon = 0
    function_type = 'exp' #exp works better for CartPole, but the paper uses lin
    if function_type == 'lin':
        lt = 700
        rt = 2000
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
        factor = 400
        epsilon = 0.01 + (1 - 0.01) * math.exp(-1. * t / factor)
    return epsilon

def main():
    #Make OpenAI gym environment
    env = gym.make('CartPole-v0')
    date_time = now.strftime("_%H:%M:%S_%m-%d-%Y")
    env = gym.wrappers.Monitor(env, './data_dqn_cartpole' + date_time)
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
    replay_memory = Simple_ReplayMemory(max_size=10000, batch_size=32)

    #Make Q-network and Target Q-network (Lines 2 & 3) 
    qnet = Simple_DQN(obs_space_shape, action_space_shape).to(device)
    target_qnet = Simple_DQN(obs_space_shape, action_space_shape).to(device)
    target_qnet.load_state_dict(qnet.state_dict())

    #Training Parameters (Changes from Mnih et al. outlined in README.md)
    optimizer = optim.Adam(qnet.parameters())
    num_frames = 100000
    gamma = 0.99
    replay_start_size = 32
    target_network_update_freq = 100
    
    #Train
    obs = env.reset()
    num_episodes = 0
    for t in range(1, num_frames+1):
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
            action = random.randrange(action_space_shape)
        torch.set_grad_enabled(True)

        #Execute action and get reward + next_obs (Line 9, 10)
        next_obs, reward, done, _ = env.step(action)

        #Store transition in Replay Memory
        replay_memory.add(obs, next_obs, action, reward, done)

        obs = next_obs

        if done:
            obs = env.reset()
            num_episodes += 1

        #Populate Replay Memory with <replay_start_size> experiences before learning
        if t > replay_start_size:
            #---------------------------------------------
            #Sample batch & compute loss & update network (Lines 12 - 15)
            #---------------------------------------------
            obs_minibatch, next_obs_minibatch, actions_minibatch, rewards_minibatch, done_minibatch = replay_memory.sample()

            ts_obs, ts_rewards, ts_next_obs, ts_done = map(lambda x: torch.FloatTensor(x).to(device), [obs_minibatch, rewards_minibatch, next_obs_minibatch, done_minibatch])
            ts_actions = torch.LongTensor(actions_minibatch).to(device)

            torch.set_grad_enabled(False)
            # #Compute Target Values 
            ts_next_qvals = target_qnet(ts_next_obs) #(32, 2)
            ts_next_action = ts_next_qvals.argmax(-1, keepdim=True) #(32, 1)
            ts_next_action_qvals = ts_next_qvals.gather(-1, ts_next_action).view(-1) #(32,)
            ts_target_q = ts_rewards + gamma * ts_next_action_qvals * (1 - ts_done)
            torch.set_grad_enabled(True)

            #Compute predicted
            ts_pred_q = qnet(ts_obs).gather(-1, ts_actions).view(-1) #(32,)

            #Calculate Loss & Perform gradient descent (Line 14) Paper uses Huber Loss, but MSE works better for CartPole
            loss = F.mse_loss(ts_pred_q, ts_target_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #Update target network ever <target_network_update_freq> steps (Line 15)
            if t % target_network_update_freq == 0:
                target_qnet.load_state_dict(qnet.state_dict())

        #Log to Terminal
        episode_rewards = env.get_episode_rewards()
        print('Timesteps', t, 'Episode', num_episodes,'Mean Reward', np.mean(episode_rewards[-100:]))
    env.env.close()
if __name__ == "__main__":
    main()
