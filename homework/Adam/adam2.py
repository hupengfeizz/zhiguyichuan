import numpy as np
import matplotlib.pyplot as plt

class SimpleAdam:
    def __init__(self, lr=0.1, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr          # 学习率
        self.beta1 = beta1    # 一阶矩衰减系数（动量）
        self.beta2 = beta2    # 二阶矩衰减系数（RMSprop）
        self.epsilon = epsilon
        self.m_w0 = 0            # 一阶矩（动量）
        self.m_w1 = 0            # 一阶矩（动量）
        self.v_w0 = 0            # 二阶矩（梯度平方平均）
        self.v_w1 = 0            # 二阶矩（梯度平方平均）
        self.t = 0            # 迭代步数

    def update(self, w0,w1, grad_w0,grad_w1):
        self.t += 1
        # 更新一阶矩（动量）
        self.m_w0 = self.beta1 * self.m_w0 + (1 - self.beta1) * grad_w0
        self.m_w1 = self.beta1 * self.m_w1 + (1 - self.beta1) * grad_w1
        # 更新二阶矩（梯度平方平均）
        self.v_w0 = self.beta2 * self.v_w0 + (1 - self.beta2) * (grad_w0 **2)
        self.v_w1 = self.beta2 * self.v_w1 + (1 - self.beta2) * (grad_w1 **2)
        # 偏差校正（解决初始m、v接近0的问题）
        m_hat_w0 = self.m_w0 / (1 - self.beta1**self.t)
        m_hat_w1 = self.m_w1 / (1 - self.beta1**self.t)
        v_hat_w0 = self.v_w0 / (1 - self.beta2**self.t)
        v_hat_w1 = self.v_w1 / (1 - self.beta2**self.t)
        # Adam更新规则
        x_new_w0 = w0 - self.lr * m_hat_w0 / (np.sqrt(v_hat_w0) + self.epsilon)
        x_new_w1 = w1 - self.lr * m_hat_w1 / (np.sqrt(v_hat_w1) + self.epsilon)
        return x_new_w0,x_new_w1

# 定义损失函数和梯度
def loss_function(w0,w1):
    # return (x - 3)**2  # 目标：找到x=3使loss最小
    l = 0.25*((63-(60*w1+w0))**2+(65.2-(62*w1+w0))**2)
    return l

def gradient(w0,w1):
    # return 2 * (x - 3)  # 损失函数的梯度
    partial_w0 = -64.1 + 61*w1+w0
    partial_w1 = -3911.2+61*w0+3722*w1
    
    return partial_w0,partial_w1

# 训练过程
# x = 0.0  # 初始值
w0 = 0
w1 = 0
adam = SimpleAdam(lr=0.1)
loss_history = []
w0_history = []
w1_history = []

for epoch in range(500):
    loss = loss_function(w0,w1)
    loss_history.append(loss)
    w0_history.append(w0)
    w1_history.append(w1)
    grad_w0,grad_w1 = gradient(w0,w1)
    w0,w1 = adam.update(w0,w1, grad_w0,grad_w1)  # Adam更新x

# 绘制loss下降曲线
plt.figure(figsize=(10, 4))
plt.subplot(1,2,1)
plt.plot(range(len(loss_history)), loss_history)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Adam Loss Decrease')

# 绘制x的收敛过程
plt.subplot(1,2,2)
plt.plot(range(len(w0_history)), w0_history, label='w0 value')
plt.plot(range(len(w1_history)), w1_history, label='w1 value')
# plt.axhline(y=3, color='r', linestyle='--', label='Optimal x=3')
plt.xlabel('Epoch')
plt.ylabel('x')
plt.title('x Convergence')
plt.legend()
plt.show()