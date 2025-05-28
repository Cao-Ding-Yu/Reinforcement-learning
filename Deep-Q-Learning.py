import gym
import torch
import random
import torch.nn as nn
from gym import Wrapper
from collections import deque
import torch.nn.functional as F

# 训练参数
config = {
    "learning_rate": 0.001,     # 学习率
    "gamma": 0.90,              # 折扣因子
    "epsilon": 1.0,             # 探索因子
    "epsilon_min": 0.005,       # 最小探索率
    "epsilon_decay": 0.99,      # 探索衰减系数
    "batch_size": 64,           # 批次数据量
    "memory_size": 2000,        # 经验回放缓存数
    "update_freq": 20,          # 网络更新频率
    "episodes": 1000,           # 最大训练轮次
    "test_episodes": 100,       # 测试轮次
}

# FrozenLake场景重写（可选）
class CustomRewardFrozenLake(Wrapper):
    def __init__(self, env):
        super().__init__(env)

    # 重写step奖励机制
    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)  # 原始环境逻辑
        if self.env.unwrapped.desc.flatten()[observation] == b'G': reward = 20
        elif self.env.unwrapped.desc.flatten()[observation] == b'H': reward = -1
        else: reward = -0.1    # 步数惩罚，加快收敛速度
        return observation, reward, terminated, truncated, info

# DQN网络
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.state_size = int(state_size)
        self.action_size = int(action_size)
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)

    # 模型推理
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    # greedy策略选择  state → action
    def select_action(self, state, epsilon=0.0):
        if random.random() < epsilon: # 随机探索策略
            return env.action_space.sample()
        else: # 最优价值策略
            state = F.one_hot(torch.tensor(state), num_classes=self.state_size).float().unsqueeze(0)
            with torch.no_grad(): return self(state).argmax().item()

# 模型训练
def train(env):
    state_size = env.observation_space.n
    action_size = env.action_space.n
    # 创建主副网络并统一参数
    model = DQN(state_size, action_size)
    target_model = DQN(state_size, action_size)
    target_model.load_state_dict(model.state_dict())
    # 创建Adam优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    # 创建经验回放缓存
    memory = deque(maxlen=config["memory_size"])
    # 网络训练
    result = list()
    update_count = 0
    for episode in range(config["episodes"]):
        state = env.reset()[0]
        while True:
            # 常规移动
            action = model.select_action(state, config["epsilon"])
            next_state, reward, done, _, _ = env.step(action)
            memory.append((state, action, reward, next_state, done))
            state = next_state
            # 经验回放
            if len(memory) >= config["batch_size"]:
                update_count += 1
                # 经验缓存随机采样
                batch = random.sample(memory, config["batch_size"])
                states, actions, rewards, next_states, dones = zip(*batch)
                # 训练数据格式转换
                states_one_hot = F.one_hot(torch.tensor(states, dtype=torch.long), num_classes=state_size).float()
                actions = torch.tensor(actions, dtype=torch.long)
                rewards = torch.tensor(rewards, dtype=torch.float32)
                next_states_one_hot = F.one_hot(torch.tensor(next_states, dtype=torch.long), num_classes=state_size).float()
                dones = torch.tensor(dones, dtype=torch.float32)
                # 计算批次预测值
                current_q = model(states_one_hot).gather(1, actions.unsqueeze(1)).squeeze(1)
                # 计算目标Q值
                with torch.no_grad():
                    next_q = target_model(next_states_one_hot)
                    max_next_q = next_q.max(1)[0]
                    target_q = rewards + config["gamma"] * max_next_q * (1 - dones)
                # 计算损失并优化
                loss = F.mse_loss(current_q, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # 更新目标网络
                if update_count % config["update_freq"] == 0: target_model.load_state_dict(model.state_dict())
            if done: # 任务结果统计
                if env.unwrapped.desc.flatten()[state] == b'G': result.append(True)
                elif env.unwrapped.desc.flatten()[state] == b'H': result.append(False)
                break
        # Epsilon衰减
        config["epsilon"] = max(config["epsilon_min"], config["epsilon"] * config["epsilon_decay"])
        # 统计训练进度
        if (episode+1) % 100 == 0: print(f"Episode {episode+1}, Success Rate: {sum(result[-100:])/100}, Epsilon: " + str(config["epsilon"]))
        if sum(result[-100:]) == 100: break
    return target_model

# 模型验证
def test(model, env):
    count = 0
    for i in range(config["test_episodes"]):
        state = env.reset()[0]
        while True:
            action = model.select_action(state)
            state, reward, done, truncated, _ = env.step(action)
            if done or truncated: break
        if env.unwrapped.desc.flatten()[state] == b'G': count += 1
    print(f"Success rate over " + str(config["test_episodes"]) + " test episodes: " + str(count * 100 /config["test_episodes"]) + "%")

if __name__ == "__main__":
    # 场景搭建
    env = CustomRewardFrozenLake(gym.make('FrozenLake8x8-v1', is_slippery=False))
    # 模型训练
    model = train(env)
    # 模型验证
    test(model, env)
    env.close()
    # pygame可视化
    env = gym.make('FrozenLake8x8-v1', render_mode="human", is_slippery=False)
    while True:
        state = env.reset()[0]
        while True:
            action = model.select_action(state)
            state, reward, done, _, _ = env.step(action)
            if done: break
