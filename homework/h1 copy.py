import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import GridSearchCV

class AdamRegressor(BaseEstimator, RegressorMixin):
    """
    支持记录 Loss 历史轨迹的 Adam 优化器类
    """
    def __init__(self, lr=0.02, p=0.95, max_iter=3000, beta_threshold=10**-6):
        self.lr = lr
        self.p = p
        self.max_iter = max_iter
        self.beta_threshold = beta_threshold
        self.w0 = 0
        self.w1 = 0
        self.iter_count_ = 0
        self.loss_history_ = [] # 用于存储每次迭代的损失值

    def fit(self, X, y):
        X = X.flatten()
        y = y.flatten()
        n = len(y)
        mt0, mt1, vt0, vt1 = 0, 0, 0, 0
        e = 10**-6
        
        self.w0, self.w1 = 0, 0
        self.loss_history_ = [] # 重置记录
        
        for i in range(1, self.max_iter + 1):
            y_pred = self.w1 * X + self.w0
            current_loss = np.mean((y - y_pred)**2)
            self.loss_history_.append(current_loss)
            
            # 计算梯度
            grad0 = -2/n * np.sum(y - y_pred)
            grad1 = -2/n * np.sum(X * (y - y_pred))
            
            # Adam 更新矩
            mt0 = self.p * mt0 + (1 - self.p) * grad0
            mt1 = self.p * mt1 + (1 - self.p) * grad1
            vt0 = self.p * vt0 + (1 - self.p) * grad0**2
            vt1 = self.p * vt1 + (1 - self.p) * grad1**2
            
            # 偏差修正
            hat_mt0 = mt0 / (1 - self.p**i)
            hat_mt1 = mt1 / (1 - self.p**i)
            hat_vt0 = vt0 / (1 - self.p**i)
            hat_vt1 = vt1 / (1 - self.p**i)
            
            # 更新参数
            prev_w0, prev_w1 = self.w0, self.w1
            self.w0 -= self.lr * hat_mt0 / (np.sqrt(hat_vt0) + e)
            self.w1 -= self.lr * hat_mt1 / (np.sqrt(hat_vt1) + e)
            
            # 再次计算新 Loss 用于判断收敛
            new_y_pred = self.w1 * X + self.w0
            new_loss = np.mean((y - new_y_pred)**2)
            
            self.iter_count_ = i
            # 收敛判断 (beta 阈值)
            if abs(current_loss - new_loss) < self.beta_threshold:
                break
                
        return self

    def predict(self, X):
        return self.w1 * X.flatten() + self.w0

    def score(self, X, y):
        # 负均方误差，GridSearchCV 默认最大化此分值
        y_pred = self.predict(X)
        return -np.mean((y.flatten() - y_pred)**2)

# --- 准备原始数据 ---
X_data = np.array([60, 62, 64, 65, 66, 67, 68, 70, 72, 74]).reshape(-1, 1)
y_data = np.array([63.6, 65.2, 66, 65.5, 66.9, 67.1, 67.4, 68.3, 70.1, 70]).reshape(-1, 1)

# --- 网格搜索：寻找最佳学习率 lr 和 衰减率 p ---
print("正在进行网格搜索以寻找最优超参数...")
param_grid = {
    'lr': [0.01, 0.02, 0.05, 0.1, 0.2, 0.5],
    'p': [0.9, 0.95, 0.99]
}

grid_search = GridSearchCV(AdamRegressor(), param_grid, cv=2, scoring='neg_mean_squared_error')
grid_search.fit(X_data, y_data)

# --- 输出搜索结果 ---
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

print("\n" + "="*40)
print(f"网格搜索完成！")
print(f"最优参数: {best_params}")
print(f"收敛迭代次数: {best_model.iter_count_}")
print(f"最终权重: w0 = {best_model.w0:.4f}, w1 = {best_model.w1:.4f}")
print(f"最终 Loss: {best_model.loss_history_[-1]:.6f}")
print("="*40 + "\n")

# --- 绘图部分 ---
plt.figure(figsize=(14, 5))

# 图1：最优参数下的 Loss 下降曲线
plt.subplot(1, 2, 1)
plt.plot(range(1, best_model.iter_count_ + 1), best_model.loss_history_, 
         color='darkorange', linewidth=2, label=f'Best params: {best_params}')
plt.yscale('log') # 使用对数坐标轴，更清晰观察收敛
plt.title('Loss Convergence (Log Scale)')
plt.xlabel('Iteration')
plt.ylabel('Mean Squared Error (Loss)')
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.legend()

# 图2：拟合结果对比图
plt.subplot(1, 2, 2)
plt.scatter(X_data, y_data, color='blue', label='Ground Truth', zorder=3)
plt.plot(X_data, best_model.predict(X_data), color='red', 
         linewidth=2, label='Adam Regression Line', zorder=2)
plt.title('Linear Regression Fit Result')
plt.xlabel('X (Feature)')
plt.ylabel('Y (Target)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()