import numpy as np
import matplotlib.pyplot as plt


def grad_at(w0, w1):
    """计算在 (w0, w1) 处的梯度（偏导数）"""
    partial_w0 = -64.1 + 61 * w1 + w0
    partial_w1 = -3911.2 + 61 * w0 + 3722 * w1
    return partial_w0, partial_w1


def loss(w0, w1):
    return 0.25 * ((63 - (60 * w1 + w0)) ** 2 + (65.2 - (62 * w1 + w0)) ** 2)


if __name__ == '__main__':
    lr = 0.0001          # 学习率 p
    momentum = 0.5       # 动量系数 weight_r
    w0, w1 = 0.0, 0.0
    v0, v1 = 0.0, 0.0    # 初始速度（动量）

    iter = 0
    alpha = 1e-6         # 梯度停止阈值
    beta = 1e-6          # 损失变化停止阈值

    loss_record = []
    iter_number = []

    loss_current = loss(w0, w1)

    for i in range(1, 200):
        # 1. 计算当前损失（用于判断变化）
        loss0 = loss_current

        # 2. 预估下一步的位置（look-ahead）
        w0_est = w0 - momentum * v0
        w1_est = w1 - momentum * v1

        # 3. 在预估位置计算梯度
        grad0, grad1 = grad_at(w0_est, w1_est)

        # 4. 更新动量（速度）
        v0 = momentum * v0 + lr * grad0
        v1 = momentum * v1 + lr * grad1

        # 5. 更新参数
        w0_new = w0 - v0
        w1_new = w1 - v1

        # 6. 计算新损失
        loss_new = loss(w0_new, w1_new)

        iter += 1
        loss_record.append(loss_new)
        iter_number.append(iter)

        # 打印信息（保持原风格）
        print(f'迭代{iter}: w0={w0_new:.4f}, w1={w1_new:.4f}, '
              f'w0梯度(预估点)={grad0:.6f}, w1梯度(预估点)={grad1:.6f}, '
              f'w0变化量={v0:.6f}, w1变化量={v1:.6f}, 损失={loss_new:.6f}')

        # 更新参数
        w0, w1 = w0_new, w1_new

        # 停止条件1：梯度（在预估点）足够小
        if abs(grad0) < alpha and abs(grad1) < alpha:
            print('梯度足够小，停止迭代')
            break

        # 停止条件2：损失变化小于阈值
        if abs(loss_new - loss0) < beta:
            print('损失变化小于阈值，停止迭代')
            break

        loss_current = loss_new

    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(iter_number, loss_record, color='red', linestyle='--',
             linewidth=2, marker='o', markersize=5, markerfacecolor='blue')
    plt.xlabel('Iter')
    plt.ylabel('Loss')
    plt.title('NAG')
    plt.grid(True)
    plt.show()