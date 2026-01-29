# -*- coding: gbk -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter

from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score

# 导入数据
data = pd.read_excel(r"/Users/macbook/Documents/data/FL199.xlsx")
X = data.values[:, 1:]
y = data.values[:, 0]

# Define wavelength range
wl = np.linspace(942, 1701, X.shape[1])
# wl = np.linspace(400, 1680, num=484)

# Calculate derivatives
# X1 = savgol_filter(X, 11, polyorder=2, deriv=1)
# X2 = savgol_filter(X, 13, polyorder=2, deriv=2)

# Define the PLS regression object
pls = PLSRegression(n_components=8)
# Fit data
pls.fit(X, y)

# Plot spectra
plt.figure(figsize=(8, 9))
with plt.style.context(()):
    ax1 = plt.subplot(211)
    plt.plot(wl, X.T)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylabel('reflectance', fontsize=18)

    ax2 = plt.subplot(212, sharex=ax1)
    plt.plot(wl, np.abs(pls.coef_[:, 0]))
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('Wavelength (nm)', fontsize=18)
    plt.ylabel('Absolute value of PLS coefficients', fontsize=18)
    # 调整图像布局
    plt.tight_layout()

    plt.show()

