import numpy as np
import matplotlib.pyplot as plt


def grad(p, w0, w1):
    partial_w0 = -64.1 + 61*w1+w0
    partial_w1 = -3911.2+61*w0+3722*w1
    delta_w0 = p * partial_w0
    delta_w1 = p * partial_w1


    w0_new = w0 - delta_w0
    w1_new = w1 - delta_w1

    return [w0_new, w1_new, delta_w0, delta_w1, partial_w0, partial_w1]


def loss(w0, w1):
    l = 0.25*((63-(60*w1+w0))**2+(65.2-(62*w1+w0))**2)
    return l


if __name__ == '__main__':
    p = 0.0001
    w0 = 0
    w1 = 0
    iter = 0
    alpha = 1e-8
    beta = 1e-8

    loss_record = []
    iter_number = []

    loss_current = loss(w0, w1)

    # while True:
    for i in range(1,200):
        w0_new, w1_new, delta_w0, delta_w1, partial_w0, partial_w1 = grad(
            p, w0, w1)

        w0, w1 = w0_new, w1_new
        loss_new = loss(w0, w1)

        iter += 1
        loss_record.append(loss_new)
        iter_number.append(iter)

        print(f'迭代{iter}: w0={w0:.4f}, w1={w1:.4f}, w0梯度={partial_w0:.6f}, w1梯度={partial_w1:.6f}, w0变化量={delta_w0:.6f},w1变化量={delta_w1:.6f} 损失={loss_new:.6f}')

        if abs(partial_w0) < beta and abs(partial_w1) < alpha:
            print(f'梯度足够小，停止迭代')
            break

        if abs(loss_new - loss_current) < beta:
            print(f'损失变化小于阈值，停止迭代')
            break

        loss_current = loss_new

    plt.figure(figsize=(10, 6))
    plt.plot(iter_number, loss_record)
    plt.xlabel('Iter')
    plt.ylabel('Loss')
    plt.show()
