import numpy as np
import matplotlib.pyplot as plt

# MR算法    Wk_next = Wk - a(k) * g(Wk,ηk)
def robbins_monro(A, b, max_iter=20, noise_scale=0.1):
    # 初始化随机解
    x = np.zeros(A.shape[1])
    # 收敛过程初始化
    history = [x.copy()]
    # 迭代收敛
    for k in range(1, max_iter+1):
        a = 1.0 / k
        noise = noise_scale * np.random.randn(len(b))
        g = A @ x - b + noise
        x = x - a * g
        history.append(x.copy())
    return x, np.array(history)

if __name__ == "__main__":
    # 构建三元方程 Ax=b
    A = np.random.randn(3, 3) + 3 * np.eye(3) # 对角优势矩阵
    b = np.random.randn(3)
    # 标准解
    true_x = np.linalg.solve(A, b)
    print("true_x =", true_x)
    # Robbins-Monro解
    x, history = robbins_monro(A, b)
    print("Robbins-Monro_x =", x)
    # 收敛可视化
    plt.figure(figsize=(10, 6))
    for i in range(3): plt.plot(history[:, i], label=f"x[{i}]", alpha=0.7)
    plt.axhline(y=true_x[0], color='blue', linestyle='--', label="True x[0]")
    plt.axhline(y=true_x[1], color='orange', linestyle='--', label="True x[1]")
    plt.axhline(y=true_x[2], color='green', linestyle='--', label="True x[2]")
    plt.title("Robbins-Monro Convergence")
    plt.xlabel("Iteration")
    plt.ylabel("Parameter Value")
    plt.legend()
    plt.show()