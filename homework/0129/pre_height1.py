import matplotlib.pyplot as plt
import numpy as np  # 引入numpy简化数值计算，这是机器学习的常用工具

# --- 1. 准备数据集（不再硬编码，封装成数组，更简洁可复用）---
# 输入特征x：假设是年龄（10组数据）
x = np.array([60, 62, 64, 65, 66, 67, 68, 70, 72, 74])
# 输出标签y：对应身高（10组数据）
y = np.array([63.6, 65.2, 66, 65.5, 66.9, 67.1, 67.4, 68.3, 70.1, 70])
n = len(x)  # 样本数量，自动计算，无需硬编码10

# --- 2. 初始化模型参数 ---
w0 = 0.0  # 截距（偏置项），对应线性模型 y = w1*x + w0
w1 = 0.0  # 斜率（权重项）

# Adam算法专属：初始化一阶矩和二阶矩（均为0）
m_w0, m_w1 = 0.0, 0.0  # 一阶矩：梯度的"累积均值"（动量项）
v_w0, v_w1 = 0.0, 0.0  # 二阶矩：梯度平方的"累积均值"（自适应学习率项）

# --- 3. 设置Adam算法超参数（采用经典默认值，更易收敛）---
beta1 = 0.9    # 一阶矩的指数衰减速率（动量衰减），经典默认值0.9
beta2 = 0.999  # 二阶矩的指数衰减速率（自适应项衰减），经典默认值0.999
lr = 0.1       # 基础学习率，经典默认值常为0.001，此处针对本数据调大更易观察
epsilon = 1e-8 # 稳定项，防止分母为0，经典默认值1e-8
max_iter = 3000# 最大迭代次数
converge_beta = 1e-6  # 损失收敛阈值

# --- 4. 初始化记录列表，用于后续可视化 ---
history = {
    'w0': [], 'w1': [], 'loss': [], 'iteration': []
}

# --- 5. 开始Adam算法迭代优化 ---
for i in range(1, max_iter + 1):
    # 步骤1：计算当前预测值 & 损失（MSE 均方误差）
    y_pred = w1 * x + w0  # 线性模型预测值，向量运算，无需循环
    loss = np.mean((y - y_pred) ** 2)  # 均方误差，numpy内置函数，简洁高效
    
    # 步骤2：计算梯度（对w0、w1的偏导数）
    # 推导：dL/dw0 = (2/n) * Σ(y_pred - y) = 2*mean(y_pred - y)
    grad_w0 = 2 * np.mean(y_pred - y)
    # 推导：dL/dw1 = (2/n) * Σ(x * (y_pred - y)) = 2*mean(x * (y_pred - y))
    grad_w1 = 2 * np.mean(x * (y_pred - y))
    
    # 步骤3：更新Adam的一阶矩（动量项，累积梯度的均值）
    # 公式：m_t = beta1 * m_{t-1} + (1 - beta1) * grad_t
    m_w0 = beta1 * m_w0 + (1 - beta1) * grad_w0
    m_w1 = beta1 * m_w1 + (1 - beta1) * grad_w1
    
    # 步骤4：更新Adam的二阶矩（自适应项，累积梯度平方的均值）
    # 公式：v_t = beta2 * v_{t-1} + (1 - beta2) * grad_t^2
    v_w0 = beta2 * v_w0 + (1 - beta2) * (grad_w0 ** 2)
    v_w1 = beta2 * v_w1 + (1 - beta2) * (grad_w1 ** 2)
    
    # 步骤5：偏差修正（解决初始值为0导致的前期偏置问题）
    # 公式：m_hat_t = m_t / (1 - beta1^t)，v_hat_t = v_t / (1 - beta2^t)
    m_hat_w0 = m_w0 / (1 - beta1 ** i)
    m_hat_w1 = m_w1 / (1 - beta1 ** i)
    v_hat_w0 = v_w0 / (1 - beta2 ** i)
    v_hat_w1 = v_w1 / (1 - beta2 ** i)
    
    # 步骤6：更新模型参数（Adam核心更新公式）
    w0 -= lr * m_hat_w0 / (np.sqrt(v_hat_w0) + epsilon)
    w1 -= lr * m_hat_w1 / (np.sqrt(v_hat_w1) + epsilon)
    
    # 步骤7：记录迭代历史
    history['w0'].append(w0)
    history['w1'].append(w1)
    history['loss'].append(loss)
    history['iteration'].append(i)
    
    # 步骤8：收敛判断（损失变化小于阈值则停止迭代）
    if i > 1 and abs(history['loss'][-2] - loss) < converge_beta:
        print(f'第{i}次迭代，损失已收敛！')
        print(f'最终参数：w0（截距）={w0:.4f}，w1（斜率）={w1:.4f}')
        print(f'最终损失（MSE）={loss:.4f}')
        break

# --- 6. 可视化结果（双图展示：损失下降趋势 + 数据拟合效果）---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# 子图1：损失随迭代次数的变化
ax1.plot(history['iteration'], history['loss'], 'r--', linewidth=2, marker='o', markersize=2)
ax1.set_title('Adam Optimization: Loss vs Iteration')
ax1.set_xlabel('Iteration')
ax1.set_ylabel('MSE Loss')
ax1.grid(True, alpha=0.3)

# 子图2：原始数据与拟合直线
ax2.scatter(x, y, color='blue', label='Original Data', s=50)
ax2.plot(x, w1 * x + w0, color='red', linewidth=2, label='Fitted Line')
ax2.set_title('Data and Adam Fitted Line')
ax2.set_xlabel('x (Age)')
ax2.set_ylabel('y (Height)')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()