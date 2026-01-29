import numpy as np
import matplotlib.pyplot as plt

class SimpleAdam:
    def __init__(self, lr=0.1, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr          # 学习率
        self.beta1 = beta1    # 一阶矩衰减系数（动量）
        self.beta2 = beta2    # 二阶矩衰减系数（RMSprop）
        self.epsilon = epsilon
        self.m = 0            # 一阶矩（动量）
        self.v = 0            # 二阶矩（梯度平方平均）
        self.t = 0            # 迭代步数

    def update(self, x, grad):
        self.t += 1
        # 更新一阶矩（动量）
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        # 更新二阶矩（梯度平方平均）
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad **2)
        # 偏差校正（解决初始m、v接近0的问题）
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        # Adam更新规则
        x_new = x - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        return x_new

# 定义损失函数和梯度
def loss_function(x):
    return (x - 3)**2  # 目标：找到x=3使loss最小

def gradient(x):
    return 2 * (x - 3)  # 损失函数的梯度

# 训练过程
x = 0.0  # 初始值
adam = SimpleAdam(lr=0.1)
loss_history = []
x_history = []

for epoch in range(500):
    loss = loss_function(x)
    loss_history.append(loss)
    x_history.append(x)
    grad = gradient(x)
    x = adam.update(x, grad)  # Adam更新x

# 绘制loss下降曲线
plt.figure(figsize=(10, 4))
plt.subplot(1,2,1)
plt.plot(range(len(loss_history)), loss_history)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Adam Loss Decrease')

# 绘制x的收敛过程
plt.subplot(1,2,2)
plt.plot(range(len(x_history)), x_history, label='x value')
plt.axhline(y=3, color='r', linestyle='--', label='Optimal x=3')
plt.xlabel('Epoch')
plt.ylabel('x')
plt.title('x Convergence')
plt.legend()
plt.show()