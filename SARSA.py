import gym
import numpy as np

# 训练参数
config = {
    'epsilon': 0.9,                         # 探索因子
    'epsilon_decay_value': 0.3,    # 探索降速
    'episodes': 1000,                   # 最大迭代
    'learning_rate': 0.2,               # 学习因子
    'discount_factor': 0.8,           # 折扣因子
}

# 算法训练
def sarsa():
    env = gym.make('FrozenLake-v1', is_slippery=False)                      # 非滑动场景
    q_table = np.zeros((env.observation_space.n, env.action_space.n))      # 初始化Q表
    # 训练过程
    success_rates = list()
    for episode in range(config['episodes']):
        # 选定初始state和action, 构造Q(s,a)
        state = env.reset()[0]
        if np.random.random() < config['epsilon']: action = env.action_space.sample()
        else:  action = np.argmax(q_table[state])
        while True:
            # 生成new_state和new_action构造Q(s',a')
            new_state, _, done, _, _ = env.step(action)
            if np.random.random() < config['epsilon']: next_action = env.action_space.sample()
            else: next_action = np.argmax(q_table[new_state])
            # 自定义奖励函数reward
            if env.desc[new_state // 4][new_state % 4] == b'H': reward = -10     # 掉进冰窟-10分
            elif env.desc[new_state // 4][new_state % 4] == b'G': reward = 10    # 找到宝藏+10分
            else: reward = -1.0                                                                             # 常规移动-1.0分
            # Q(s,a) ← Q(s,a) + α * ( reward + γQ(s',a') - Q(s,a) )
            q_table[state, action] = q_table[state, action] + config['learning_rate'] * (reward + config['discount_factor'] * q_table[new_state, next_action] - q_table[state, action])
            state = new_state
            action = next_action
            # 更新统计信息
            if done: success_rates.append(1 if env.desc[state // 4][state % 4] == b'G' else 0); break
        # 更新探索因子 & 进度控制
        if episode % 50 == 0:
            config['epsilon'] = max((config['epsilon'] - config['epsilon_decay_value']), 0)
            print(f"Episode: {episode:>5}, Success: {np.mean(success_rates[-100:]):.1%}, Epsilon: {config['epsilon']:.3f}")
            if np.mean(success_rates[-100:]) == 1.0: break
    env.close()
    return q_table

if __name__ == '__main__':
    # 算法训练
    q_table = sarsa()
    # 初始化environment和agent
    test_env = gym.make('FrozenLake-v1', render_mode='human', is_slippery=False)
    current_state = test_env.reset()[0]
    # 循环展示智能体行为
    while True:
        action = np.argmax(q_table[current_state])
        current_state, _, done, _, _ = test_env.step(action)
        if done: current_state = test_env.reset()[0]