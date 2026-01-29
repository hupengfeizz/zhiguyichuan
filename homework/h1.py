import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import GridSearchCV

class AdamRegressor(BaseEstimator, RegressorMixin):
    """
    手动实现封装的 Adam 线性回归类，兼容 Scikit-Learn 接口
    """
    def __init__(self, lr=0.02, p=0.95, max_iter=3000, beta_threshold=10**-6):
        self.lr = lr
        self.p = p
        self.max_iter = max_iter
        self.beta_threshold = beta_threshold
        self.w0 = 0
        self.w1 = 0
        self.iter_count_ = 0 # 记录实际收敛的迭代次数

    def fit(self, X, y):
        # 展平数据
        X = X.flatten()
        y = y.flatten()
        n = len(y)
        
        # 初始化 Adam 矩变量
        mt0, mt1 = 0, 0
        vt0, vt1 = 0, 0
        e = 10**-6
        
        self.w0, self.w1 = 0, 0
        
        for i in range(1, self.max_iter + 1):
            # 计算预测值
            y_pred = self.w1 * X + self.w0
            loss0 = np.mean((y - y_pred)**2)
            
            # 计算梯度
            # dL/dw0 = -2/n * sum(y - y_pred)
            grad0 = -2/n * np.sum(y - y_pred)
            # dL/dw1 = -2/n * sum(X * (y - y_pred))
            grad1 = -2/n * np.sum(X * (y - y_pred))
            
            # Adam 矩更新
            mt0 = self.p * mt0 + (1 - self.p) * grad0
            mt1 = self.p * mt1 + (1 - self.p) * grad1
            vt0 = self.p * vt0 + (1 - self.p) * grad0**2
            vt1 = self.p * vt1 + (1 - self.p) * grad1**2
            
            # 偏差修正
            hat_mt0 = mt0 / (1 - self.p**i)
            hat_mt1 = mt1 / (1 - self.p**i)
            hat_vt0 = vt0 / (1 - self.p**i)
            hat_vt1 = vt1 / (1 - self.p**i)
            
            # 参数更新
            self.w0 -= self.lr * hat_mt0 / (np.sqrt(hat_vt0) + e)
            self.w1 -= self.lr * hat_mt1 / (np.sqrt(hat_vt1) + e)
            
            # 计算新 Loss
            y_pred_new = self.w1 * X + self.w0
            loss1 = np.mean((y - y_pred_new)**2)
            
            self.iter_count_ = i
            # 判断收敛
            if abs(loss0 - loss1) < self.beta_threshold:
                break
        return self

    def predict(self, X):
        return self.w1 * X.flatten() + self.w0

    def score(self, X, y):
        # 使用负均方误差作为评分（GridSearchCV 默认最大化得分，所以 MSE 越小得分越高）
        y_pred = self.predict(X)
        return -np.mean((y.flatten() - y_pred)**2)

# --- 准备数据 ---
# 你的原始身高数据
X_data = np.array([60, 62, 64, 65, 66, 67, 68, 70, 72, 74]).reshape(-1, 1)
y_data = np.array([63.6, 65.2, 66, 65.5, 66.9, 67.1, 67.4, 68.3, 70.1, 70]).reshape(-1, 1)

# --- 设置网格搜索 ---
param_grid = {
    'lr': [0.01, 0.05, 0.1, 0.5],          # 尝试不同的学习率
    'p': [0.9, 0.95, 0.99],               # 尝试不同的动量衰减系数
    'beta_threshold': [10**-6]            # 收敛阈值固定
}

# 使用 GridSearchCV 进行搜索
# cv=2 因为样本极少，实际大数据集建议 5 或 10
grid_search = GridSearchCV(AdamRegressor(), param_grid, cv=2, scoring='neg_mean_squared_error')
grid_search.fit(X_data, y_data)

# --- 输出最优结果 ---
print("="*30)
print(f"最优参数组合: {grid_search.best_params_}")
print(f"最小负均方误差: {grid_search.best_score_:.4f}")

# 使用最优参数重新实例化并查看迭代次数
best_model = grid_search.best_estimator_
print(f"在该参数下，收敛所需的迭代次数: {best_model.iter_count_}")
print(f"最终权重: w0={best_model.w0:.4f}, w1={best_model.w1:.4f}")
print("="*30)

# --- 可视化对比 ---
plt.scatter(X_data, y_data, color='blue', label='Actual Data')
plt.plot(X_data, best_model.predict(X_data), color='red', label='Adam Optimized Line')
plt.title(f'Best Fit: lr={best_model.lr}, p={best_model.p}')
plt.legend()
plt.grid(True)
plt.show()