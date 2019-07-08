import gym
from torch import nn, optim
import torch.nn.functional as F
import torch
import numpy as np
import math, random
from datetime import datetime
from collections import deque
from wrappers import make_atari, wrap_deepmind, wrap_pytorch, NoopResetEnv, MaxAndSkipEnv

now = datetime.now()
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

class ReplayMemory(object):
    def __init__(self, max_size=10000, batch_size=32):
        """
        Replay Memory implementation for Atari games 
        """
        super(ReplayMemory, self).__init__()

        self.max_size, self.batch_size = max_size, batch_size
        self.memory = deque(maxlen=self.max_size)
        self.count = 0 #keep track of the number of elements

    def add(self, obs, next_obs, action, reward, done):
            """
            Add an experience to replay memory
            obs: np.array, shape (obs_space_shape, ) 
            next_obs: np.array, shape (obs_space_shape, ) 
            action: np.array, shape (env.action_space.n, ) 
            reward: float, done: boolean
            """
            obs = np.expand_dims(obs, 0).astype(np.float32) 
            next_obs = np.expand_dims(next_obs, 0).astype(np.float32)
            self.count = min(self.max_size, self.count + 1)
            self.memory.append((obs, action, reward, next_obs, done))

    def sample(self):
        """ Sample a random minibatch of states

        obs_minibatch: np.array shape (self.batch_size, obs_space_shape) 
        actions_minibatch: np.array shape (self.batch_size, env.action_space.n) 
        rewards_minibatch: tuple len = self.batch_size 
        next_obs_minibatch: np.array- same as obs_minibatch
        done_minibatch: tuple len = self.batch_size
        """
        obs_minibatch, actions_minibatch, rewards_minibatch, next_obs_minibatch, done_minibatch = zip(*random.sample(self.memory, self.batch_size))
        actions_minibatch = np.expand_dims(actions_minibatch, 1)

        return np.concatenate(obs_minibatch), np.concatenate(next_obs_minibatch), actions_minibatch, rewards_minibatch, done_minibatch

def _weights_init(m):
    if hasattr(m, 'weight'):
        nn.init.kaiming_uniform_(m.weight)
    if hasattr(m, 'bias'):
        nn.init.constant_(m.bias, 0)

class Atari_DQN(nn.Module):
    def __init__(self, obs_space_shape, action_space_shape):
        """
        Neural Network that predicts the Q-value for all actions "a_t" given an input state "s_t" for Atari Games
        Visit: http://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-7.pdf 
        obs_space_shape:int, env.observation_space.shape[0] 
        action_space_shape:int, env.action_space.n 
        """
        super(Atari_DQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(obs_space_shape, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.fully_connected = nn.Sequential(
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(),
            nn.Linear(512, action_space_shape)
        )

        # He Initialization 
        self.apply(_weights_init)

    def forward(self, obs):
        """
        obs: tensor, shape (batch_size, <obs_space_shape>)
        Returns tensor shape (batch_size, <action_space_shape>)
        """
        out = self.conv(obs)
        out = out.view(out.size(0), -1) #flatten 
        out = self.fully_connected(out)
        return out

def epsilon_at_t(t):
    """
    Defines "epsilon" for frame "t" for epsilon-greedy exploration strategy 
    W/ probability "epsilon", we choose a random action at - otherwise we choose at=argmax_at[Q(st, at)] (Read Mnih et al. 2015)
    t: int, Frame number (Frames encountered later in training have higher frame nums.) 
    Returns epsilon: float 
    """
    epsilon = 0
    function_type = 'lin' #Paper uses lin, but you can try exp too
    if function_type == 'lin':
        lt = 50000
        rt = 1000000
        #Start off always exploring
        if t < lt:
            epsilon = 1
        #Linearly decrease exploration param
        elif t >= lt and t < rt:
            alpha = float(t - lt) / (rt - lt)
            epsilon = 1 + alpha * (0.1 - 1)
        #Fix a very low exploration param for large t
        else:
            epsilon = 0.1
    elif function_type == 'exp':
        factor = 30000
        epsilon = 0.01 + (1 - 0.01) * math.exp(-1. * t / factor)
    return epsilon

def main():
    #Make OpenAI gym environment + wrappers
    date_time = now.strftime("_%H:%M:%S_%m-%d-%Y")
    env = gym.make("PongNoFrameskip-v4")
    env = gym.wrappers.Monitor(env, './data_dqn_ataripong' + date_time)
    assert 'NoFrameskip' in env.spec.id
    env = NoopResetEnv(env, noop_max=30) 
    env = MaxAndSkipEnv(env, skip=4) #skip 4 frames & max over last_obs
    env = wrap_deepmind(env)
    env = wrap_pytorch(env) #obs shape = num_channels x width x height
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
    replay_memory = ReplayMemory(max_size=100000)

    #Make Q-Network and Target Q-Network (Lines 2 & 3)
    qnet = Atari_DQN(obs_space_shape, action_space_shape).to(device)
    target_qnet = Atari_DQN(obs_space_shape, action_space_shape).to(device)
    target_qnet.load_state_dict(qnet.state_dict())

    #Training Parameters (Changes from Mnih et al. outlined in README.md)
    optimizer = optim.Adam(qnet.parameters())
    num_frames = 1400000 
    gamma = 0.99
    replay_start_size = 50000
    target_network_update_freq = 10000

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
            ts_obs = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0).to(device)
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
            # Compute Target Values (as per Double-DQN update rule)
            ts_next_qvals_outer = qnet(ts_next_obs) #(32, 2) (outer Qnet, evaluates value)
            ts_next_qvals_inner = target_qnet(ts_next_obs) #(32, 2) (inner Qnet, evaluates action)
            ts_next_action_inner = ts_next_qvals_inner.argmax(-1, keepdim=True) #(32, 1)
            ts_next_action_qvals_outer = ts_next_qvals_outer.gather(-1, ts_next_action_inner).view(-1) #(32, ) (use inner actions to evaluate outer Q values)
            ts_target_q = ts_rewards + gamma * ts_next_action_qvals_outer * (1 - ts_done)
            torch.set_grad_enabled(True)

            #Compute predicted
            ts_pred_q = qnet(ts_obs).gather(-1, ts_actions).view(-1) #(32,)

            #Calculate Loss & Perform gradient descent (Line 14) 
            loss = F.smooth_l1_loss(ts_pred_q, ts_target_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #Update target network ever <target_network_update_freq> steps (Line 15)
            if t % target_network_update_freq == 0:
                target_qnet.load_state_dict(qnet.state_dict())

        #Log to Terminal
        episode_rewards = env.env.env.env.env.env.env.env.get_episode_rewards()
        print('Timesteps', t, 'Episode', num_episodes,'Mean Reward', np.mean(episode_rewards[-100:]))
    env.env.close()
if __name__ == "__main__":
    main()
