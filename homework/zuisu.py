import numpy as np
import matplotlib.pyplot as plt

def loss(w):
    """损失函数 L(w) = (w-20)² + 5"""
    return (w - 20)**2 + 5

def compute_gradient(w):
    """计算梯度 ∇L(w)"""
    return 2 * (w - 20)

def compute_negative_gradient_unit_vector(gradient):
    """计算负梯度及其单位向量"""
    negative_gradient = -gradient
    # 为了避免除以0，加一个极小值保护
    norm = np.linalg.norm(negative_gradient) + 1e-10
    unit_vector = negative_gradient / norm
    return negative_gradient, unit_vector

def compute_exponential_decay_step(initial_step, decay_rate, iteration):
    """指数衰减步长计算：ρ_k = initial_step * (decay_rate)^iteration"""
    return initial_step * (decay_rate ** iteration)

if __name__ == '__main__':
    # 1. 初始化参数
    w = 2.0  # 初始点
    initial_step = 0.5  # 初始步长
    decay_rate = 0.99   # 衰减率
    alpha = 1e-6        # 梯度停止阈值
    beta = 1e-8         # 损失变化停止阈值
    iter = 0
    loss_history = []
    iter_history = []
    loss_current = loss(w)

    while True:
        # 2. 计算梯度、负梯度及其单位向量
        gradient = compute_gradient(w)
        negative_gradient, unit_vector = compute_negative_gradient_unit_vector(gradient)
        
        # 3. 检查梯度是否足够小
        if abs(negative_gradient) <= alpha:
            break
        
        # 4. 计算指数衰减的最佳步长
        rho_k = compute_exponential_decay_step(initial_step, decay_rate, iter)
        
        # 5. 更新权重
        w_new = w + rho_k * unit_vector
        
        # 6. 检查损失变化是否足够小
        loss_new = loss(w_new)
        if abs(loss_new - loss_current) <= beta:
            w = w_new
            break
        
        # 记录历史
        loss_history.append(loss_new)
        iter_history.append(iter)
        
        # 更新状态
        w = w_new
        loss_current = loss_new
        iter += 1
        
        # 打印过程
        print(f"Iter {iter}: w={w:.6f}, loss={loss_new:.6f}, step={rho_k:.6f}, gradient={gradient:.6f}")

    print(f"\n最终结果：w={w:.6f}, loss={loss(w):.6f}, 迭代次数={iter}")

    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(iter_history, loss_history, 'b-o')
    plt.xlabel('迭代次数')
    plt.ylabel('损失值')
    plt.title('梯度下降（指数衰减步长 + 单位方向向量）')
    plt.grid(True)
    plt.show()