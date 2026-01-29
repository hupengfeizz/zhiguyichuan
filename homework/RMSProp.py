import numpy as np
import matplotlib.pyplot as plt

def grad(w0, w1):
    # 计算原始梯度
    partial_w0 = -64.1 + 61*w1 + w0
    partial_w1 = -3911.2 + 61*w0 + 3722*w1
    return partial_w0, partial_w1

def loss(w0, w1):
    return 0.25 * ((63 - (60*w1 + w0))**2 + (65.2 - (62*w1 + w0))**2)

if __name__ == '__main__':
    # 超参数设置
    p = 0.1         # RMSprop 通常允许比纯梯度下降更大的学习率
    gamma = 0.9     # 衰减率 (Decay rate)
    epsilon = 1e-8  # 防止除以零
    
    w0, w1 = 0.0, 0.0
    s0, s1 = 0.0, 0.0  # 累计梯度平方和
    
    alpha = 1e-8
    beta = 1e-8
    loss_record = []
    iter_number = []
    loss_current = loss(w0, w1)

    for i in range(1, 1000): # 增加迭代次数以观察效果
        # 1. 获取当前梯度
        g0, g1 = grad(w0, w1)

        # 2. 更新梯度的平方移动平均 (RMSprop 核心)
        s0 = gamma * s0 + (1 - gamma) * (g0 ** 2)
        s1 = gamma * s1 + (1 - gamma) * (g1 ** 2)

        # 3. 计算自适应步长并更新参数
        delta_w0 = (p / np.sqrt(s0 + epsilon)) * g0
        delta_w1 = (p / np.sqrt(s1 + epsilon)) * g1
        
        w0 -= delta_w0
        w1 -= delta_w1

        # 记录与打印
        loss_new = loss(w0, w1)
        loss_record.append(loss_new)
        iter_number.append(i)

        print(f'迭代{i}: w0={w0:.4f}, w1={w1:.4f}, loss={loss_new:.6f}')

        # 停止条件
        if abs(g0) < beta and abs(g1) < alpha:
            print("梯度足够小，停止迭代")
            break
        if abs(loss_new - loss_current) < beta:
            print("损失变化极小，停止迭代")
            break
            
        loss_current = loss_new

    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(iter_number, loss_record)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('RMSprop Optimization')
    plt.grid(True)
    plt.show()