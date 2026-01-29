import matplotlib.pyplot as plt

# --- 初始化参数 ---
w0 = 0      # 截距初始值
w1 = 0      # 斜率初始值
mt0, mt1 = 0, 0  # 初始化一阶矩（动量项）
vt0, vt1 = 0, 0  # 初始化二阶矩（变稀疏项）

# --- 超参数设置 ---
p = 0.95         # 矩估计的指数衰减速率 (Adam中的 beta1 和 beta2)
lr = 0.02        # 初始学习率 (Learning Rate)
e = 10**-6       # 稳定项，防止除以0 (Epsilon)
alpha = 10**-6   # 权重停止阈值
beta = 10**-6    # 损失变化停止阈值

w0_list, w1_list, loss_list, iter_list = [], [], [], []

# --- 开始迭代优化 ---
for i in range(1, 3000):
    # 1. 计算当前的均方误差 (MSE Loss)
    # Loss = (1/n) * Σ(y_actual - (w1*x + w0))^2
    loss0 = 1 / 10 *((63.6 - 60*w1 - w0) ** 2
                   + (65.2 - 62*w1 - w0) ** 2
                   + (66 - 64*w1 - w0) ** 2
                   + (65.5-65*w1 - w0) ** 2
                   + (66.9-66*w1 - w0) ** 2
                   + (67.1-67*w1 - w0) ** 2
                   + (67.4-68*w1 - w0) ** 2
                   + (68.3-70*w1 - w0) ** 2
                   + (70.1-72*w1 - w0) ** 2
                   + (70-74*w1 - w0) ** 2)

    # 2. 计算偏导数 (Gradients)
    # 对w0求导：dL/dw0 = (1/n) * Σ[-2 * (y - (w1*x + w0))]
    Gradient0 = 1/10 * (-2*(63.6-60*w1 - w0)-2*(65.2 - 62*w1 - w0)-2*(66 - 64*w1 - w0)-2*(65.5-65*w1 - w0)
                    -2*(66.9-66*w1 - w0)-2*(67.1-67*w1 - w0)-2*(67.4-68*w1 - w0)-2*(68.3-70*w1 - w0)
                    -2*(70.1-72*w1 - w0)-2*(70-74*w1 - w0))
    # 对w1求导：dL/dw1 = (1/n) * Σ[-2 * x * (y - (w1*x + w0))]
    Gradient1 = 1/10 * (-60*(63.6-60*w1 - w0)-62*(65.2 - 62*w1 - w0)-64*(66 - 64*w1 - w0)-65*(65.5-65*w1 - w0)
                    -66*(66.9-66*w1 - w0)-67*(67.1-67*w1 - w0)-68*(67.4-68*w1 - w0)-70*(68.3-70*w1 - w0)
                    -72*(70.1-72*w1 - w0)-74*(70-74*w1 - w0))

    # 3. 更新 Adam 的动量项 (Moments Update)
    # 更新一阶矩（梯度的均值）
    mt0 = p * mt0 + (1 - p) * Gradient0
    mt1 = p * mt1 + (1 - p) * Gradient1
    
    res = p * vt0
    
    # 更新二阶矩（梯度平方的均值）
    vt0 = p * vt0 + (1 - p) * Gradient0**2
    vt1 = p * vt1 + (1 - p) * Gradient1**2

    # 4. 偏差修正 (Bias Correction)
    # 由于 mt 和 vt 初始化为0，初期会产生偏置，需除以 (1-p^i)
    hat_mt0 = mt0 / (1 - p**i)
    hat_mt1 = mt1 / (1 - p**i)
    hat_vt0 = vt0 / (1 - p**i)
    hat_vt1 = vt1 / (1 - p**i)

    # 5. 参数更新 (Parameters Update)
    # w = w - lr * m_hat / (sqrt(v_hat) + epsilon)
    w0 = w0 - lr * hat_mt0 / (hat_vt0 + e)**0.5
    w1 = w1 - lr * hat_mt1 / (hat_vt1 + e)**0.5

    # 6. 计算更新后的 Loss
    loss1 = 1 / 10 *((63.6 - 60*w1 - w0) ** 2
                   + (65.2 - 62*w1 - w0) ** 2
                   + (66 - 64*w1 - w0) ** 2
                   + (65.5-65*w1 - w0) ** 2
                   + (66.9-66*w1 - w0) ** 2
                   + (67.1-67*w1 - w0) ** 2
                   + (67.4-68*w1 - w0) ** 2
                   + (68.3-70*w1 - w0) ** 2
                   + (70.1-72*w1 - w0) ** 2
                   + (70-74*w1 - w0) ** 2)

    # 记录过程数据
    w0_list.append(w0)
    w1_list.append(w1)
    loss_list.append(loss1)
    iter_list.append(i)

    # 7. 收敛判断 (Stopping Criteria)
    if abs(loss0 - loss1) < beta:
        print(f'第{i}次迭代, 损失已收敛。w0={w0:.4f}, w1={w1:.4f}, loss={loss1:.4f}')
        break

# --- 可视化 ---
plt.plot(iter_list, loss_list, color='red', linestyle='--', linewidth=2, marker='o', markersize=2)
plt.title('Adam Optimization Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.grid(True)
plt.show()