import gym
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = {
    "buffer_size": 10000,   # 经验池容量
    "episodes": 1000,       # 最大训练次数
    "gamma": 0.95,          # 折扣因子
    "batch_size": 128,      # 批次数据量
    "noise_theta": 0.2,     # 噪声衰减因子
    "noise_sigma": 0.1,     # 噪声变化阈值
    "tar_rate": 0.005,      # 目标网络学习率
}

# 经验回放缓冲区
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        self.Experience = namedtuple('Experience',['state', 'action', 'reward', 'next_state', 'done'])

    def __len__(self):
        return len(self.buffer)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append(self.Experience(state, action, reward, next_state, done))

    def sample(self, batch_size):
        experiences = random.sample(self.buffer, k=batch_size)
        states = np.vstack([e.state for e in experiences])
        actions = np.vstack([e.action for e in experiences])
        rewards = np.vstack([e.reward for e in experiences])
        next_states = np.vstack([e.next_state for e in experiences])
        dones = np.vstack([e.done for e in experiences])
        return (torch.tensor(states, dtype=torch.float32).to(device),
                torch.tensor(actions, dtype=torch.float32).to(device),
                torch.tensor(rewards, dtype=torch.float32).to(device),
                torch.tensor(next_states, dtype=torch.float32).to(device),
                torch.tensor(dones, dtype=torch.float32).to(device))

# Actor策略梯度网络
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))

# Critic值函数评估网络
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# DDPG智能体
class DDPGAgent:
    def __init__(self, env, buffer_capacity=config["buffer_size"]):
        # 场景信息初始化
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.action_high = torch.tensor(env.action_space.high).to(device)
        self.action_low = torch.tensor(env.action_space.low).to(device)
        # 网络初始化
        self.actor = Actor(self.state_dim, self.action_dim).to(device)
        self.critic = Critic(self.state_dim, self.action_dim).to(device)
        self.actor_target = Actor(self.state_dim, self.action_dim).to(device)
        self.critic_target = Critic(self.state_dim, self.action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-3)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)
        # 经验池初始化
        self.buffer = ReplayBuffer(buffer_capacity)

    # 策略决策
    def act(self, state, noise=None, noise_theta=config["noise_theta"], noise_sigma=config["noise_sigma"]):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad(): action = self.actor(state).cpu().numpy().squeeze(0)
        if noise is not None: # 添加噪声，提高探索性
            noise = (noise * (1 - noise_theta) + noise_sigma * np.random.randn(self.action_dim))
            return np.clip(action + noise, self.action_low.cpu().numpy(), self.action_high.cpu().numpy()), noise
        else: return action

    # 网络离线更新
    def update(self, gamma=config["gamma"], tar_rate=config["tar_rate"], batch_size=config["batch_size"]):
        if len(self.buffer) < batch_size: return
        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)
        # Critic更新
        next_Q = self.critic_target(next_states, self.actor_target(next_states))
        target_Q = rewards + (1 - dones) * gamma * next_Q
        critic_loss = F.mse_loss(self.critic(states, actions), target_Q.detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        # Actor更新
        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        # 目标网络更新
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(tar_rate * param.data + (1 - tar_rate) * target_param.data)
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(tar_rate * param.data + (1 - tar_rate) * target_param.data)

    # 智能体训练
    def train(self, episodes=config["episodes"]):
        for episode in range(episodes):
            total_reward = 0
            state = self.env.reset()[0]
            noise = np.zeros(self.action_dim)
            while True:
                action, noise = self.act(state, noise=noise)
                next_state, reward, done, truncated, _ = self.env.step(action)
                self.buffer.add(state, action, reward, next_state, done or truncated)
                state = next_state
                total_reward += reward
                self.update()
                if done or truncated: break
            print(f"Episode: {episode + 1}, Total Reward: {total_reward:.2f}")
            if total_reward > -1: break

# 创建环境并训练
if __name__ == "__main__":
    # 训练场景初始化
    env = gym.make('Pendulum-v1')
    # 智能体初始化
    agent = DDPGAgent(env)
    # 智能体训练
    agent.train()
    env.close()
    # 测试场景初始化
    env = gym.make('Pendulum-v1', render_mode="human")
    while True:
        state = env.reset()[0]
        while True:
            action = agent.act(state)
            state, reward, done, truncated, _ = env.step(action)
            if done or truncated: break
    