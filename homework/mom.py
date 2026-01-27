import numpy as np
import matplotlib.pyplot as plt


def grad(w0, w1):
    """返回损失函数对 w0 和 w1 的偏导数（梯度）"""
    partial_w0 = -64.1 + 61 * w1 + w0
    partial_w1 = -3911.2 + 61 * w0 + 3722 * w1
    return partial_w0, partial_w1


def loss(w0, w1):
    l = 0.25 * ((63 - (60 * w1 + w0))**2 + (65.2 - (62 * w1 + w0))**2)
    return l


if __name__ == '__main__':
    p = 0.0001          # 学习率 (learning rate)
    gamma = 0.5         # 动量系数 (momentum coefficient)
    w0, w1 = 0.0, 0.0
    v_w0, v_w1 = 0.0, 0.0   # 初始化速度

    iter = 0
    alpha = 1e-8        # 梯度停止阈值
    beta = 1e-8         # 损失变化停止阈值

    loss_record = []
    iter_number = []

    loss_current = loss(w0, w1)

    for i in range(1, 200):
        # 计算当前梯度
        partial_w0, partial_w1 = grad(w0, w1)

        # Momentum 更新：先更新速度，再更新参数
        v_w0 = gamma * v_w0 + p * partial_w0
        v_w1 = gamma * v_w1 + p * partial_w1

        w0_new = w0 - v_w0
        w1_new = w1 - v_w1

        w0, w1 = w0_new, w1_new
        loss_new = loss(w0, w1)

        iter += 1
        loss_record.append(loss_new)
        iter_number.append(iter)

        print(f'迭代{iter}: w0={w0:.4f}, w1={w1:.4f}, '
              f'w0梯度={partial_w0:.6f}, w1梯度={partial_w1:.6f}, '
              f'w0速度={v_w0:.6f}, w1速度={v_w1:.6f}, 损失={loss_new:.6f}')

        # 停止条件
        if abs(partial_w0) < alpha and abs(partial_w1) < alpha:
            print('梯度足够小，停止迭代')
            break

        if abs(loss_new - loss_current) < beta:
            print('损失变化小于阈值，停止迭代')
            break

        loss_current = loss_new

    plt.figure(figsize=(10, 6))
    plt.plot(iter_number, loss_record, marker='o', markersize=3)
    plt.xlabel('Iter')
    plt.ylabel('Loss')
    plt.title('Momentum Gradient Descent')
    plt.grid(True)
    plt.show()