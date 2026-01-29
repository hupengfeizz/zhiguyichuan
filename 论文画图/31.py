# -*- coding: gbk -*-
import matplotlib.pyplot as plt
import numpy as np
import random

N = 100
RMSECV = [random.randint(1, 1000) for _ in range(100)]
min_index = np.argmin(RMSECV)

with plt.style.context(('seaborn-whitegrid')):
    fig = plt.figure()
    fonts = 16
    plt.figure(figsize=(8, 6))
    plt.grid(False)

    # 设置横坐标轴刻度
    x_ticks = np.arange(0, N+1, 10)  # 从0到N，每隔10一个刻度
    plt.xticks(x_ticks, fontsize=16)

    plt.yticks(fontsize=18)
    plt.xlabel('Monte Carlo iterations', fontsize=fonts)
    plt.ylabel('RMSECV', fontsize=fonts)
    plt.plot(np.arange(N), RMSECV)
    plt.axvline(x=min_index, color='r', linestyle='-')

    # 在左下角添加文字
    plt.text(5, min(RMSECV) + 50, "(a)", fontsize=16, color='black', ha='left', va='bottom')

    # 调整图像布局
    plt.tight_layout()

    plt.show()
