import numpy as np
import matplotlib.pyplot as plt


def grad(w0, w1):
    """返回损失函数对 w0 和 w1 的偏导数（即梯度）"""
    partial_w0 = -64.1 + 61 * w1 + w0
    partial_w1 = -3911.2 + 61 * w0 + 3722 * w1
    return partial_w0, partial_w1


def loss(w0, w1):
    l = 0.25 * ((63 - (60 * w1 + w0)) ** 2 + (65.2 - (62 * w1 + w0)) ** 2)
    return l


if __name__ == '__main__':
    lr = 0.0001          # 学习率 p
    momentum = 0.5       # 动量系数 weight_r
    w0, w1 = 0.0, 0.0
    v0, v1 = 0.0, 0.0    # 初始速度

    iter = 0
    alpha = 1e-6         # 梯度停止阈值
    beta = 1e-6          # 损失变化停止阈值

    loss_record = []
    iter_number = []

    loss_current = loss(w0, w1)

    for i in range(1, 200):
        # 计算当前梯度
        partial_w0, partial_w1 = grad(w0, w1)

        # Momentum 更新：v = momentum * v + lr * grad
        v0 = momentum * v0 + lr * partial_w0
        v1 = momentum * v1 + lr * partial_w1

        # 参数更新：w = w - v
        w0_new = w0 - v0
        w1_new = w1 - v1

        # 计算新损失
        loss_new = loss(w0_new, w1_new)

        iter += 1
        loss_record.append(loss_new)
        iter_number.append(iter)

        # 打印信息（保持原风格）
        print(f'迭代{iter}: w0={w0_new:.4f}, w1={w1_new:.4f}, '
              f'w0梯度={partial_w0:.6f}, w1梯度={partial_w1:.6f}, '
              f'w0变化量={v0:.6f}, w1变化量={v1:.6f}, 损失={loss_new:.6f}')

        # 更新参数
        w0, w1 = w0_new, w1_new

        # 停止条件1：梯度足够小
        if abs(partial_w0) < alpha and abs(partial_w1) < alpha:
            print('梯度足够小，停止迭代')
            break

        # 停止条件2：损失变化小于阈值
        if abs(loss_new - loss_current) < beta:
            print('损失变化小于阈值，停止迭代')
            break

        loss_current = loss_new

    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(iter_number, loss_record, color='red', linestyle='--',
             linewidth=2, marker='o', markersize=5, markerfacecolor='blue')
    plt.xlabel('Iter')
    plt.ylabel('Loss')
    plt.title('Momentum')
    plt.grid(True)
    plt.show()