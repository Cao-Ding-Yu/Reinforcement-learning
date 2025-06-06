import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# 训练参数
config = {
    "episodes": 500,                # 最大训练次数
    "gamma": 0.99,                  # 折扣因子
    "actor_learning_rate": 0.001,   # actor学习率
    "critic_learning_rate": 0.005,  # critic学习率
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
        return F.softmax(self.fc3(x), dim=-1)

# Critic价值网络
class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# A2C算法实现
class A2C_Agent:
    def __init__(self, state_dim, action_dim):
        self.gamma = config["gamma"]
        self.critic = Critic(state_dim)
        self.actor = Actor(state_dim, action_dim)
        self.actor_lr = config["actor_learning_rate"]
        self.critic_lr = config["critic_learning_rate"]
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.critic_lr)

    # 策略决策
    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        dist = torch.distributions.Categorical(self.actor(state))
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    # 模型更新
    def update(self, state, log_prob, reward, next_state, done):
        # 清空梯度
        self.critic_optim.zero_grad()
        self.actor_optim.zero_grad()
        # 格式转换
        state = torch.FloatTensor(state).unsqueeze(0)
        next_state = torch.FloatTensor(next_state).unsqueeze(0)
        reward = torch.FloatTensor([reward])
        # Critic更新
        current_v = self.critic(state)
        with torch.no_grad():
            next_v = self.critic(next_state)
            target_v = reward + self.gamma * next_v * (1 - done)
        critic_loss = F.mse_loss(current_v, target_v)
        critic_loss.backward()
        self.critic_optim.step()
        # Actor更新
        with torch.no_grad():
            current_v = current_v.clone().detach()
            advantage = target_v - current_v
        actor_loss = (-log_prob * advantage.detach()).sum()
        actor_loss.backward()
        self.actor_optim.step()

# 训练过程
def train_agent(env, agent, episodes=config["episodes"]):
    for episode in range(episodes):
        state = env.reset()[0]
        episode_reward = 0
        while True:
            action, log_prob = agent.get_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            agent.update(state, log_prob, reward, next_state, done)
            state = next_state
            episode_reward += reward
            if done or truncated: break
        print(f"Episode {episode:4d} | Reward: {episode_reward:4.1f}")

if __name__ == "__main__":
    # 初始化倒立摆
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    # 训练智能体
    agent = A2C_Agent(state_dim, action_dim)
    train_agent(env, agent)
    env.close()
    # 可视化演示
    env = gym.make('CartPole-v1', render_mode="human")
    while True:
        state = env.reset()[0]
        while True:
            action, _ = agent.get_action(state)
            state, _, done, truncated, _ = env.step(action)
            if done or truncated: break