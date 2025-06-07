import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# 训练参数配置
config = {
    "episodes": 1000,       # 最大训练轮次
    "gamma": 0.90,          # 折扣因子
    "actor_lr": 0.001,      # Actor学习率
    "critic_lr": 0.002,     # Critic学习率
    "hidden_size": 64,      # 隐藏层大小
}

# Actor策略网络
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, config["hidden_size"])
        self.fc2 = nn.Linear(config["hidden_size"], config["hidden_size"])
        self.fc3 = nn.Linear(config["hidden_size"], action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=-1)

# Critic价值网络
class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, config["hidden_size"])
        self.fc2 = nn.Linear(config["hidden_size"], config["hidden_size"])
        self.fc3 = nn.Linear(config["hidden_size"], 1)  # 输出单个状态值

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# A2C智能体
class A2C_Agent:
    def __init__(self, state_dim, action_dim):
        self.action_dim = action_dim
        self.gamma = config["gamma"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.critic = Critic(state_dim).to(self.device)
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config["critic_lr"])
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config["actor_lr"])

    # 概率策略决策
    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_probs = self.actor(state)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        return action.item()

    # 更新网络参数
    def update(self, state, action, reward, next_state, done):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        next_state = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        action = torch.tensor([action], dtype=torch.long).to(self.device)
        reward = torch.tensor([reward], dtype=torch.float32).to(self.device)
        done = torch.tensor([done], dtype=torch.float32).to(self.device)
        # Critic值函数更新
        current_v = self.critic(state)
        with torch.no_grad():
            next_v = torch.zeros(1) if done else self.critic(next_state)
            target_v = reward + self.gamma * next_v
        critic_loss = F.mse_loss(current_v, target_v)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        # Actor策略梯度更新 A(s,a) = Q(s, a) - V(s) ≈ r + γV(s') - V(s)
        with torch.no_grad(): advantage = (target_v - current_v).detach()
        dist = torch.distributions.Categorical(self.actor(state))
        log_prob = dist.log_prob(action)
        actor_loss = -log_prob * advantage
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

# 训练智能体
def train_agent(env, agent):
    rewards_list = list()
    for episode in range(config["episodes"]):
        state = env.reset()[0]
        episode_reward = 0
        while True:
            action = agent.get_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            agent.update(state, action, reward, next_state, done or truncated)
            state = next_state
            episode_reward += reward
            if done or truncated: break
        rewards_list.append(episode_reward)
        print(f"Episode {episode + 1}/{config['episodes']} | "f"Reward: {episode_reward:.1f}")
        if episode_reward == 500: break

if __name__ == "__main__":
    # 环境初始化
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    # 智能体训练
    agent = A2C_Agent(state_dim, action_dim)
    train_agent(env, agent)
    env.close()
    # 可视化演示
    test_env = gym.make('CartPole-v1', render_mode="human")
    while True:
        state = test_env.reset()[0]
        while True:
            action = agent.get_action(state)
            state, _, done, truncated, _ = test_env.step(action)
            if done or truncated: break