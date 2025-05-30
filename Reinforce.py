import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# 训练参数
config = {
    "learning_rate": 0.01,      # 学习率
    "target_reward": 500,       # 目标得分
    "gamma": 0.99,              # 折扣因子
    "episodes": 1000,           # 最大训练轮次
}

# 策略网络
class Reinforce(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Reinforce, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, action_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    # 模型推理
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))
        return x

# 模型训练
def train(policy, env):
    # 创建Adam优化器
    optimizer = optim.Adam(policy.parameters(), lr=config["learning_rate"])
    # 探索迭代
    for episode in range(config["episodes"]):
        rewards = list()    # 即时奖励列表
        log_probs = list()  # 对数概率列表
        episode_reward = 0  # 本回合总奖励
        # 重置初始状态，开始新一轮探索
        state = env.reset()[0]
        while True:
            # 预测当前状态动作概率
            state = torch.FloatTensor(state).unsqueeze(0)
            action_probs = policy(state)
            # 创建概率分布并采样
            dist = Categorical(action_probs)
            action = dist.sample()
            # 执行策略并收集环境反馈
            state, reward, done, truncated, _ = env.step(action.item())
            rewards.append(reward)
            episode_reward += reward
            log_probs.append(dist.log_prob(action))
            if done or truncated: break
        print(f"Episode {episode}, Reward: {episode_reward}")
        # 模型训练达标，中断训练
        if episode_reward >= config["target_reward"]: print("The training is completed!"); break
        # 反向计算折扣回报
        cumulative_reward = 0         # 当前累计回报
        discounted_rewards = list()   # 累计回报列表
        for r in reversed(rewards):
            cumulative_reward = r + config["gamma"] * cumulative_reward
            discounted_rewards.insert(0, cumulative_reward)
        # 数据归一化
        discounted_rewards = torch.FloatTensor(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)
        # 计算损失值
        policy_loss = list()
        for log_prob, reward in zip(log_probs, discounted_rewards):
            policy_loss.append(-log_prob * reward)
        policy_loss = torch.cat(policy_loss).sum()
        # 更新策略网络参数
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

# 模型测试
def test(policy):
    env = gym.make('CartPole-v1', render_mode="human")
    with torch.no_grad():
        while True:
            state = env.reset()[0]
            while True:
                state = torch.FloatTensor(state).unsqueeze(0)
                action_probs = policy(state)
                action = torch.argmax(action_probs)
                state, reward, done, _, _ = env.step(action.item())

if __name__ == "__main__":
    # 创建倒立摆场景
    env = gym.make('CartPole-v1')
    # 创建Reinforce模型
    policy = Reinforce(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n)
    # 模型训练
    train(policy, env)
    env.close()
    # 结果展示
    test(policy)
