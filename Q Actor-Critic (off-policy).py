import gym
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from collections import deque
import torch.nn.functional as F

# 训练参数
config = {
    "episodes": 500,                # 最大训练轮次
    "gamma": 0.90,                  # 折扣因子
    "batch_size": 64,               # 批次数据量
    "buffer_size": 1000,            # 经验数据量
    "actor_learning_rate": 0.001,   # actor学习率
    "critic_learning_rate": 0.002,  # critic学习率
}

# Actor策略网络
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=-1)
        return x

# Critic价值网络
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.action_dim = int(action_dim)
        self.fc1 = nn.Linear(state_dim + action_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, state, action):
        # 将状态和动作拼接作为输入
        action_onehot = F.one_hot(action, num_classes=self.action_dim).float()
        x = torch.cat([state, action_onehot], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# 经验回放缓冲区
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.as_tensor(np.array(states), dtype=torch.float32),
            torch.as_tensor(np.array(actions), dtype=torch.long),
            torch.as_tensor(np.array(rewards), dtype=torch.float32).unsqueeze(1),
            torch.as_tensor(np.array(next_states), dtype=torch.float32),
            torch.as_tensor(np.array(dones), dtype=torch.float32).unsqueeze(1)
        )

# QAC算法实现
class QAC_Agent:
    def __init__(self, state_dim, action_dim):
        self.gamma = config["gamma"]
        self.batch_size = config["batch_size"]
        self.buffer_size = config["buffer_size"]
        self.buffer = ReplayBuffer(self.buffer_size)
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim, action_dim)
        self.actor_lr = config["actor_learning_rate"]
        self.critic_lr = config["critic_learning_rate"]
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.critic_lr)

    # 策略决策
    def get_action(self, state):
        with torch.no_grad(): probs = self.actor(torch.FloatTensor(state).unsqueeze(0))
        return torch.multinomial(probs, 1).item()

    # 模型更新
    def update(self):
        if len(self.buffer) < self.batch_size: return
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        # Critic更新
        current_q = self.critic(states, actions)
        with torch.no_grad():
            next_actions = torch.argmax(self.actor(next_states), dim=1)
            next_q = self.critic(next_states, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * next_q
        critic_loss = F.mse_loss(current_q, target_q)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()
        # Actor更新
        dist = torch.distributions.Categorical(self.actor(states))
        sampled_actions = dist.sample()
        log_probs = dist.log_prob(sampled_actions).unsqueeze(1)
        q_values = self.critic(states, sampled_actions)
        actor_loss = -torch.mean(q_values * log_probs)
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

# 训练过程
def train_agent(env, agent, episodes=config["episodes"]):
    for episode in range(episodes):
        state = env.reset()[0]
        episode_reward = 0
        while True:
            action = agent.get_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            agent.buffer.add(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            agent.update()
            if done or truncated: break
        print(f"Episode {episode:4d} | Reward: {episode_reward:4.1f}")
        if episode_reward == 500: break

if __name__ == "__main__":
    # 初始化倒立摆
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    # 训练智能体
    agent = QAC_Agent(state_dim, action_dim)
    train_agent(env, agent)
    env.close()
    # 可视化演示
    env = gym.make('CartPole-v1',render_mode="human")
    while True:
        state = env.reset()[0]
        while True:
            action = agent.get_action(state)
            state, reward, done, truncated, _ = env.step(action)
            if done or truncated: break
