import numpy as np
import matplotlib.pyplot as plt

# MBGD算法    Wk_next = Wk - a(k) * ▽f(Wk,Xk)
def mbgd(A, b, batch_size=3, learning_rate=0.01, max_iter=500):
    # 初始化随机解
    x = np.zeros(A.shape[1])
    # 收敛过程初始化
    history = [x.copy()]
    # 迭代收敛
    for k in range(1, max_iter+1):
        indices = np.random.choice(A.shape[0], batch_size, replace=False)
        A_batch, b_batch = A[indices], b[indices]
        f = A_batch.T @ (A_batch @ x - b_batch) / batch_size
        x -= learning_rate * f
        history.append(x.copy())
    return x, np.array(history)

if __name__ == "__main__":
    # 构建三元方程 Ax=b
    A = np.random.randn(3, 3) + 3 * np.eye(3) # 对角优势矩阵
    b = np.random.randn(3)
    # 标准解
    true_x = np.linalg.solve(A, b)
    print("true_x =", true_x)
    # MBGD解
    x, history = mbgd(A, b)
    print("MBGD_x =", x)
    # 收敛可视化
    plt.figure(figsize=(10, 6))
    for i in range(3): plt.plot(history[:, i], label=f"x[{i}]", alpha=0.7)
    plt.axhline(y=true_x[0], color='blue', linestyle='--', label="True x[0]")
    plt.axhline(y=true_x[1], color='orange', linestyle='--', label="True x[1]")
    plt.axhline(y=true_x[2], color='green', linestyle='--', label="True x[2]")
    plt.title("MBGD Convergence")
    plt.xlabel("Iteration")
    plt.ylabel("Parameter Value")
    plt.legend()
    plt.show()