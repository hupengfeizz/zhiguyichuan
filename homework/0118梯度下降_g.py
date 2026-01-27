import numpy as np
import matplotlib.pyplot as plt

def get_gradient(w):
    # L(w) = (w-20)^2 + 5
    # L'(w) = 2 * (w-20)
    return 2 * (w - 20)

def loss(w):
    return (w - 20)**2 + 5

if __name__ == '__main__':
    p = 0.1  # 学习率
    w = 2.0  # 初始权重
    
    loss_threshold = 0.001
    grad_threshold = 0.01
    
    iters = [0]
    losses = [loss(w)]
    
    current_iter = 0
    while True:
        grad = get_gradient(w)
        
        # 检查梯度阈值
        if abs(grad) < grad_threshold:
            break
            
        # 更新权重
        w_new = w - p * grad
        loss_new = loss(w_new)
        
        # 检查损失变化阈值
        if abs(loss_new - losses[-1]) < loss_threshold:
            w = w_new
            current_iter += 1
            iters.append(current_iter)
            losses.append(loss_new)
            break
            
        # 记录数据
        w = w_new
        current_iter += 1
        iters.append(current_iter)
        losses.append(loss_new)
        
        print(f'Iter {current_iter}: w = {w:.4f}, Loss = {loss_new:.4f}')

    # 绘图优化
    plt.plot(iters, losses, marker='o', markersize=4)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Gradient Descent Convergence')
    plt.grid(True)
    # plt.savefig('gd_plot.png')
    plt.show()