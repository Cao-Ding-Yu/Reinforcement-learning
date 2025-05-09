import gym
import numpy as np

# 算法参数
config = {
    'epsilon': 1.0,                 # 初始探索率ε
    'decay_rate': 0.999,        # 探索衰减率
    'min_epsilon': 0.01,        # 最小探索率ε
    'episodes': 5000,           # 步数episodes
    'gamma': 0.9,                # 折扣因子γ
    'learning_rate': 0.01      # 学习率rate
}

# 重写step函数
def step(state, action):
    reward = -1.0
    new_pos = None
    current_pos = (state // 4, state % 4)
    if action == 0: new_pos = (current_pos[0], current_pos[1] - 1)
    elif action == 1: new_pos = (current_pos[0] + 1, current_pos[1])
    elif action == 2: new_pos = (current_pos[0], current_pos[1] + 1)
    elif action == 3: new_pos = (current_pos[0] - 1, current_pos[1])
    # 行动撞墙 -5.0分
    if new_pos[0] == -1 or new_pos[0] == 4 or new_pos[1] == -1 or new_pos[1] == 4:
        return state, -5.0
    # 掉进冰窟-5.0分
    if env.desc[new_pos[0]][new_pos[1]] == b'H':
        return state, -5.0
    # 找到宝藏+100分
    elif env.desc[new_pos[0]][new_pos[1]] == b'G':
        reward = 100.0
    return 4*new_pos[0]+new_pos[1], reward

# ε-greedy函数
# Π(a|s) = 1 - ε / |A(s)| * (|A(s)| -1)  or  Π(a|s) = ε / |A(s)|
def epsilon_greedy_action(state, Q, epsilon):
    best_action = np.random.choice(np.where(Q[state] == np.max(Q[state]))[0])
    if np.random.random() < 1 - epsilon / Q.shape[1] * (Q.shape[1]-1): return best_action
    else: return np.random.choice([x for x in range(Q.shape[1]) if x != best_action])

# MC ε-greedy 算法
def mc_greedy(env):
    G = 0
    path = list() # [(state, action, reward)]
    state = env.reset()[0] # 探索起始点
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    # 路径探索
    for episode in range(config['episodes']):
        action = epsilon_greedy_action(state, Q, config['epsilon'])
        new_state, reward = step(state, action)
        path.append((state, action, reward))
        state = new_state
    # 衰减率ε更新
    config['epsilon'] = max(config['epsilon'] * config['decay_rate'], config['min_epsilon'])
    # 回溯更新Q表
    for i in reversed(range(len(path))):
        state, action, reward = path[i]
        G = reward + config['gamma'] * G
        Q[state][action] += config['learning_rate'] * (G - Q[state][action])
    return Q

if __name__ == "__main__":
    # 创建训练环境
    env = gym.make('FrozenLake-v1', is_slippery=False)
    # 算法训练
    Q = mc_greedy(env)
    print(Q)
    # 算法测试
    test_env = gym.make('FrozenLake-v1', is_slippery=False, render_mode='human')
    current_state = test_env.reset()[0]
    while True:
        action = np.argmax(Q[current_state])
        current_state, _, done, _, _ = test_env.step(action)
        if done: current_state = test_env.reset()[0]