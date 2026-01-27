import numpy as np
import matplotlib.pyplot as plt

# 计算梯度向量
def compute_grad(w0, w1):
    partial_w0 = -64.1 + 61 * w1 + w0
    partial_w1 = -3911.2 + 61 * w0 + 3722 * w1
    return np.array([partial_w0, partial_w1])

# 计算损失
def loss(w0, w1):
    l = 0.25 * ((63 - (60 * w1 + w0)) ** 2 + (65.2 - (62 * w1 + w0)) ** 2)
    return l
 # 指数衰减步长
def exponential_decay_step(p0, k, decay_rate=0.9, decay_interval=100):
    return p0 * (decay_rate ** (k // decay_interval))

# # 精确线搜索（可选）
# def exact_line_search(w, s):
#     w0, w1 = w
#     s0, s1 = s
#     a = 0.25 * ((60*s1 + s0)**2 + (62*s1 + s0)**2)
#     b = 0.25 * (2*(63 - (60*w1 + w0))*(60*s1 + s0) + 2*(65.2 - (62*w1 + w0))*(62*s1 + s0))
#     if a == 0:
#         return 0.0
#     return -b / (2 * a)

if __name__ == '__main__':
    # 初始化参数
    w = np.array([0.0, 0.0])  # w0, w1
    p0 = 0.1  # 初始步长
    k = 0  # 迭代次数
    alpha = 1e-2  # 梯度阈值
    beta = 1e-2  # 损失变化阈值
    loss_record = []
    step_record = []
    iter_number = []

    loss_current = loss(*w)

    # while True:
    for i in range(1,200):
        # 2. 计算负梯度及其单位向量
        grad_vec = compute_grad(*w)
        s_k = -grad_vec  # 负梯度
        grad_norm = np.linalg.norm(s_k)
        if grad_norm < 1e-12:
            s_unit = np.zeros_like(s_k)
        else:
            s_unit = s_k / grad_norm  # 单位向量

        # 3. 检查梯度是否足够小
        if grad_norm <= alpha:
            print(f"梯度足够小，停止迭代")
            break

        # 4. 计算最佳步长（二选一）
        # --- 选项1：指数衰减步长
        p_k = exponential_decay_step(p0, k)
        # --- 选项2：精确线搜索
        # p_k = exact_line_search(w, s_unit)

        # 5. 更新权重：步长 × 单位向量
        w_new = w + p_k * s_unit

        # 6. 检查损失变化
        loss_new = loss(*w_new)
        if abs(loss_new - loss_current) <= beta:
            print(f"损失变化小于阈值，停止迭代")
            break

        # 7. 更新迭代状态
        w = w_new
        loss_current = loss_new
        k += 1

        # 记录
        loss_record.append(loss_new)
        step_record.append(p_k)
        iter_number.append(k)
        print(f'迭代{k}: w0={w[0]:.4f}, w1={w[1]:.4f}, 梯度模长={grad_norm:.6f}, 步长={p_k:.8f}, 损失={loss_new:.6f}')
        
        # if k==100:
        #     break

    # 绘图：双图并排
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 图a：损失变化
    ax1.plot(iter_number, loss_record, color='tab:blue', linewidth=0.8, marker='.', markersize=2)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.set_title('(a) Loss vs Iteration')
    ax1.grid(True, alpha=0.3)

    # 图b：步长变化
    ax2.plot(iter_number, step_record, color='tab:orange')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Step Size')
    ax2.set_title('(b) Step Size vs Iteration')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()