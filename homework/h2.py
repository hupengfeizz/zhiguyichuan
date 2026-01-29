import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import RandomizedSearchCV

class AdamRegressor(BaseEstimator, RegressorMixin):
    """
    手动封装的 Adam 线性回归类
    """
    def __init__(self, lr=0.02, p=0.95, max_iter=3000, beta_threshold=10**-6):
        self.lr = lr
        self.p = p
        self.max_iter = max_iter
        self.beta_threshold = beta_threshold
        self.w0 = 0
        self.w1 = 0
        self.iter_count_ = 0
        self.loss_history_ = []

    def fit(self, X, y):
        X = X.flatten()
        y = y.flatten()
        n = len(y)
        mt0, mt1, vt0, vt1 = 0, 0, 0, 0
        e = 1e-6
        
        self.w0, self.w1 = 0, 0
        self.loss_history_ = []
        
        for i in range(1, self.max_iter + 1):
            y_pred = self.w1 * X + self.w0
            loss0 = np.mean((y - y_pred)**2)
            self.loss_history_.append(loss0)
            
            # 计算梯度
            grad0 = -2/n * np.sum(y - y_pred)
            grad1 = -2/n * np.sum(X * (y - y_pred))
            
            # Adam 矩更新
            mt0 = self.p * mt0 + (1 - self.p) * grad0
            mt1 = self.p * mt1 + (1 - self.p) * grad1
            vt0 = self.p * vt0 + (1 - self.p) * grad0**2
            vt1 = self.p * vt1 + (1 - self.p) * grad1**2
            
            # 偏差修正 (这里的 p 统一代表 beta1 和 beta2)
            hat_mt0 = mt0 / (1 - self.p**i)
            hat_mt1 = mt1 / (1 - self.p**i)
            hat_vt0 = vt0 / (1 - self.p**i)
            hat_vt1 = vt1 / (1 - self.p**i)
            
            # 更新权重
            self.w0 -= self.lr * hat_mt0 / (np.sqrt(hat_vt0) + e)
            self.w1 -= self.lr * hat_mt1 / (np.sqrt(hat_vt1) + e)
            
            # 再次计算新 Loss 用于判断收敛
            new_y_pred = self.w1 * X + self.w0
            loss1 = np.mean((y - new_y_pred)**2)
            
            self.iter_count_ = i
            if abs(loss0 - loss1) < self.beta_threshold:
                break
        return self

    def predict(self, X):
        return self.w1 * X.flatten() + self.w0

    def score(self, X, y):
        # 返回负 MSE 以供 Scikit-Learn 优化
        y_pred = self.predict(X)
        return -np.mean((y.flatten() - y_pred)**2)

# --- 准备数据 ---
X_data = np.array([60, 62, 64, 65, 66, 67, 68, 70, 72, 74]).reshape(-1, 1)
y_data = np.array([63.6, 65.2, 66, 65.5, 66.9, 67.1, 67.4, 68.3, 70.1, 70]).reshape(-1, 1)

# --- 配置 RandomizedSearchCV ---
# 我们可以使用连续分布（如 np.linspace 或 np.logspace）来让搜索更随机化
param_distributions = {
    'lr': np.logspace(-2, 0, 100),  # 在 0.001 到 1.0 之间随机抽取 100 个对数间距值
    'p': np.linspace(0.1, 0.9999, 200), # 在 0.8 到 0.99 之间随机抽取 20 个值
    'beta_threshold': [1e-6]
}

# n_iter=20 表示随机抽取 20 组不同的参数组合进行测试
random_search = RandomizedSearchCV(
    AdamRegressor(), 
    param_distributions=param_distributions, 
    n_iter=1000, 
    cv=2, 
    scoring='neg_mean_squared_error',
    random_state=42 # 保证结果可复现
)

print("正在进行随机搜索优化...")
random_search.fit(X_data, y_data)

# --- 获取最优模型 ---
best_model = random_search.best_estimator_
best_params = random_search.best_params_

# --- 打印输出 ---
print("\n" + "="*45)
print(f"【随机搜索完成】")
print(f"最佳参数组合: {best_params}")
print(f"最终收敛次数: {best_model.iter_count_} 次")
print(f"最终权重结果: w0 = {best_model.w0:.4f}, w1 = {best_model.w1:.4f}")
print(f"最小训练 Loss: {best_model.loss_history_[-1]:.6f}")
print("="*45 + "\n")

# --- Loss 变化图和拟合图 ---
plt.figure(figsize=(15, 6))

# 子图1: Loss 曲线
plt.subplot(1, 2, 1)
plt.plot(range(1, len(best_model.loss_history_) + 1), best_model.loss_history_, 
         color='blue', linewidth=1.5, label='Loss Path')
plt.yscale('log') # 逻辑坐标系更能体现从大 loss 到小 loss 的收敛细节
plt.title(f"Adam Loss Convergence\n(lr={best_params['lr']:.4f}, p={best_params['p']:.4f})")
plt.xlabel("Iteration")
plt.ylabel("MSE Loss (Log Scale)")
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.legend()

# 子图2: 最终拟合效果
plt.subplot(1, 2, 2)
plt.scatter(X_data, y_data, color='red', label='Original Data', s=50)
x_range = np.linspace(X_data.min()-2, X_data.max()+2, 100).reshape(-1, 1)
plt.plot(x_range, best_model.predict(x_range), color='black', 
         linestyle='-', linewidth=2, label='Adam Best Fit')
plt.title("Regression Result Post-Optimization")
plt.xlabel("Height/Feature")
plt.ylabel("Target Value")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()