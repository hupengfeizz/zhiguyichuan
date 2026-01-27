import numpy as np
import matplotlib.pyplot as plt

# ---------------------- 公共配置 ----------------------
# 初始权重
w_initial = 0.0
# 学习率
learning_rate = 0.01
# 迭代次数
epochs = 1000
# 目标值（我们要优化w到20）
target_w = 20

# ---------------------- 1. 批量梯度下降（BGD）实现 ----------------------
def bgd_optimize(w_init, lr, epochs):
    w = w_init
    # 记录每一步的w值和损失值，用于后续绘图
    w_history_bgd = []
    loss_history_bgd = []
    
    for _ in range(epochs):
        # BGD：基于全部样本计算梯度（这里任务简单，全局误差直接对应梯度）
        # 损失函数 loss = (w - 20)²
        gradient = 2 * (w - target_w)  # 全局梯度（核心：无随机性，基于整体）
        # 更新权重
        w = w - lr * gradient
        # 记录历史数据
        loss = np.square(w - target_w)
        w_history_bgd.append(w)
        loss_history_bgd.append(loss)
    
    return np.array(w_history_bgd), np.array(loss_history_bgd)

# ---------------------- 2. 随机梯度下降（SGD）实现 ----------------------
def sgd_optimize(w_init, lr, epochs):
    w = w_init
    # 记录每一步的w值和损失值，用于后续绘图
    w_history_sgd = []
    loss_history_sgd = []
    
    for _ in range(epochs):
        # SGD：随机引入单个样本的噪声（模拟单个样本计算的梯度，核心：有随机性）
        # 这里添加小范围随机噪声，模拟单个样本与全局样本的误差差异
        random_noise = np.random.normal(0, 3)  # 正态分布随机噪声，均值0，标准差3
        sample_gradient = 2 * (w - target_w) + random_noise  # 单个样本的梯度（带随机波动）
        # 更新权重
        w = w - lr * sample_gradient
        # 记录历史数据
        loss = np.square(w - target_w)
        w_history_sgd.append(w)
        loss_history_sgd.append(loss)
    
    return np.array(w_history_sgd), np.array(loss_history_sgd)

# ---------------------- 3. 运行两种算法并获取结果 ----------------------
w_bgd, loss_bgd = bgd_optimize(w_initial, learning_rate, epochs)
w_sgd, loss_sgd = sgd_optimize(w_initial, learning_rate, epochs)

# ---------------------- 4. 绘图对比（直观观察差异） ----------------------
plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示问题
plt.rcParams['axes.unicode_minus'] = False

# 创建2个子图，分别对比w的变化和损失的变化
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# 子图1：w值的迭代变化
ax1.plot(range(epochs), w_bgd, label='BGD（批量梯度下降）', color='blue', linewidth=2)
ax1.plot(range(epochs), w_sgd, label='SGD（随机梯度下降）', color='red', alpha=0.7, linewidth=1)
ax1.axhline(y=target_w, color='green', linestyle='--', label='目标w=20')
ax1.set_title('BGD vs SGD - w值迭代变化')
ax1.set_xlabel('迭代次数')
ax1.set_ylabel('w的值')
ax1.legend()
ax1.grid(alpha=0.3)

# 子图2：损失值的迭代变化
ax2.plot(range(epochs), loss_bgd, label='BGD（批量梯度下降）', color='blue', linewidth=2)
ax2.plot(range(epochs), loss_sgd, label='SGD（随机梯度下降）', color='red', alpha=0.7, linewidth=1)
ax2.axhline(y=0, color='green', linestyle='--', label='目标损失=0')
ax2.set_title('BGD vs SGD - 损失值迭代变化')
ax2.set_xlabel('迭代次数')
ax2.set_ylabel('损失值（loss=(w-20)²）')
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang TC', 'Heiti TC']
plt.rcParams['axes.unicode_minus'] = False
plt.show()

# ---------------------- 5. 输出最终结果对比 ----------------------
print(f"===== 最终结果对比 =====")
print(f"BGD 最终w值：{w_bgd[-1]:.6f}，最终损失值：{loss_bgd[-1]:.8f}")
print(f"SGD 最终w值：{w_sgd[-1]:.6f}，最终损失值：{loss_sgd[-1]:.8f}")