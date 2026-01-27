# bgd_sgd_demo.py

import numpy as np
import matplotlib.pyplot as plt


class LinearRegressionBGD:
    """使用批量梯度下降（BGD）实现线性回归"""
    
    def __init__(self, learning_rate=0.01, max_iters=1000, tol=1e-6):
        self.learning_rate = learning_rate
        self.max_iters = max_iters
        self.tol = tol  # 收敛阈值
        self.costs = []  # 记录每次迭代的损失
    
    def fit(self, X, y):
        """
        训练模型
        :param X: 特征矩阵 (n_samples, n_features)
        :param y: 目标向量 (n_samples,)
        """
        m, n = X.shape
        # 添加偏置项（截距）
        X_b = np.c_[np.ones((m, 1)), X]  # (m, n+1)
        self.theta = np.random.randn(n + 1)  # 初始化参数
        
        for i in range(self.max_iters):
            # 所有样本的预测值
            y_pred = X_b.dot(self.theta)
            # 计算梯度（对所有样本求平均）
            gradient = (2 / m) * X_b.T.dot(y_pred - y)
            # 更新参数
            self.theta -= self.learning_rate * gradient
            
            # 计算损失（MSE）
            cost = (1 / m) * np.sum((y_pred - y) ** 2)
            self.costs.append(cost)
            
            # 检查是否收敛
            if len(self.costs) > 1 and abs(self.costs[-2] - self.costs[-1]) < self.tol:
                print(f"BGD 在第 {i+1} 次迭代后收敛。")
                break
    
    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b.dot(self.theta)


class LinearRegressionSGD:
    """使用随机梯度下降（SGD）实现线性回归"""
    
    def __init__(self, learning_rate=0.01, max_epochs=1000, tol=1e-6, random_state=42):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.tol = tol
        self.random_state = random_state
        self.costs = []  # 每个 epoch 的平均损失
    
    def fit(self, X, y):
        """
        训练模型
        :param X: 特征矩阵 (n_samples, n_features)
        :param y: 目标向量 (n_samples,)
        """
        m, n = X.shape
        X_b = np.c_[np.ones((m, 1)), X]
        self.theta = np.random.randn(n + 1)
        np.random.seed(self.random_state)
        
        for epoch in range(self.max_epochs):
            # 随机打乱数据
            indices = np.random.permutation(m)
            X_b_shuffled = X_b[indices]
            y_shuffled = y[indices]
            
            epoch_cost = 0.0
            for i in range(m):
                xi = X_b_shuffled[i:i+1]  # shape (1, n+1)
                yi = y_shuffled[i]
                # 单个样本预测
                y_pred_i = xi.dot(self.theta)
                # 单个样本梯度
                gradient = 2 * xi.T.dot(y_pred_i - yi).flatten()
                # 更新参数
                self.theta -= self.learning_rate * gradient
                # 累加损失
                epoch_cost += (y_pred_i - yi) ** 2
            
            avg_cost = epoch_cost / m
            self.costs.append(avg_cost)
            
            # 检查收敛（可选）
            if len(self.costs) > 1 and abs(self.costs[-2] - self.costs[-1]) < self.tol:
                print(f"SGD 在第 {epoch+1} 轮后收敛。")
                break
    
    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b.dot(self.theta)


# 示例：生成模拟数据并比较 BGD 与 SGD
if __name__ == "__main__":
    # 设置随机种子
    np.random.seed(42)
    
    # 生成数据: y = 3 + 2*x1 + noise
    X = 2 * np.random.rand(100, 1)
    y = 3 + 2 * X.flatten() + np.random.randn(100) * 0.5

    # 使用 BGD
    bgd_model = LinearRegressionBGD(learning_rate=0.1, max_iters=1000)
    bgd_model.fit(X, y)
    print("BGD 参数:", bgd_model.theta)

    # 使用 SGD
    sgd_model = LinearRegressionSGD(learning_rate=0.01, max_epochs=50)
    sgd_model.fit(X, y)
    print("SGD 参数:", sgd_model.theta)

    # 可视化损失曲线
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(bgd_model.costs, label="BGD")
    plt.title("BGD Loss Curve")
    plt.xlabel("Iteration")
    plt.ylabel("MSE")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(sgd_model.costs, label="SGD", color='orange')
    plt.title("SGD Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()

    plt.tight_layout()
    plt.show()