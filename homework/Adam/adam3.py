import numpy as np
import matplotlib.pyplot as plt

class SimpleAdam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr          # 学习率（针对双权重，这里调小更稳定）
        self.beta1 = beta1    # 一阶矩（动量）衰减系数
        self.beta2 = beta2    # 二阶矩（梯度平方平均）衰减系数
        self.epsilon = epsilon# 防止分母为0的小常数
        self.m = np.zeros(2)  # 一阶矩，对应w0、w1两个权重，初始化为0
        self.v = np.zeros(2)  # 二阶矩，对应w0、w1两个权重，初始化为0
        self.t = 0            # 迭代步数，用于偏差校正

    def update(self, weights, grads):
        """
        多权重Adam更新
        :param weights: 当前权重数组 [w0, w1]
        :param grads: 当前梯度数组 [grad_w0, grad_w1]
        :return: 更新后的权重数组 [w0_new, w1_new]
        """
        self.t += 1
        # 1. 更新一阶矩（动量，对应w0、w1分别更新）
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        # 2. 更新二阶矩（梯度平方平均，对应w0、w1分别更新）
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grads ** 2)
        # 3. 偏差校正（解决初始m、v接近0的问题，针对w0、w1分别校正）
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        # 4. Adam核心更新规则，计算新权重
        weights_new = weights - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        return weights_new

# ---------------------- 你的损失函数和梯度计算（保留核心，微调格式）----------------------
def get_partial_grads(w0, w1):
    """提取w0、w1的偏导数（来自你的grad函数，仅返回梯度，不做朴素更新）"""
    partial_w0 = -64.1 + 61 * w1 + w0
    partial_w1 = -3911.2 + 61 * w0 + 3722 * w1
    return np.array([partial_w0, partial_w1])  # 转为数组，适配Adam

def loss(w0, w1):
    """你的原始损失函数"""
    l = 0.25 * ((63 - (60 * w1 + w0)) ** 2 + (65.2 - (62 * w1 + w0)) ** 2)
    return l

# ---------------------- 基于Adam的训练过程 ----------------------
if __name__ == '__main__':
    # 1. 初始化参数
    adam = SimpleAdam(lr=0.001)  # Adam学习率（双权重场景下0.001更稳定）
    weights = np.array([0.0, 0.0])  # 初始w0=0, w1=0
    alpha = 1e-8
    beta = 1e-8
    loss_record = []
    iter_number = []
    iter_count = 0
    loss_current = loss(weights[0], weights[1])  # 初始损失

    # 2. 迭代训练（最多8000步，保留原停止条件）
    for i in range(1, 8000):
        iter_count += 1
        w0, w1 = weights[0], weights[1]
        # 获取当前w0、w1的偏导数（梯度）
        grads = get_partial_grads(w0, w1)
        # Adam算法更新权重
        weights_new = adam.update(weights, grads)
        # 计算新损失
        w0_new, w1_new = weights_new[0], weights_new[1]
        loss_new = loss(w0_new, w1_new)

        # 3. 记录数据
        loss_record.append(loss_new)
        iter_number.append(iter_count)

        # 4. 打印迭代信息（保持和你原代码格式一致）
        delta_w0 = w0 - w0_new  # 权重变化量（原代码逻辑：旧值-新值）
        delta_w1 = w1 - w1_new
        print(f'迭代{iter_count}: w0={w0_new:.4f}, w1={w1_new:.4f}, '
              f'w0梯度={grads[0]:.6f}, w1梯度={grads[1]:.6f}, '
              f'w0变化量={delta_w0:.6f},w1变化量={delta_w1:.6f} 损失={loss_new:.6f}')

        # 5. 停止条件判断（和你原代码一致）
        if abs(grads[0]) < beta and abs(grads[1]) < alpha:
            print(f'梯度足够小，停止迭代')
            break
        if abs(loss_new - loss_current) < beta:
            print(f'损失变化小于阈值，停止迭代')
            break

        # 6. 更新当前权重和当前损失
        weights = weights_new
        loss_current = loss_new

    # 7. 绘制损失下降曲线（和你原代码一致）
    plt.figure(figsize=(10, 6))
    plt.plot(iter_number, loss_record)
    plt.xlabel('Iter')
    plt.ylabel('Loss')
    plt.title('Adam Algorithm - Loss Decrease')
    plt.show()