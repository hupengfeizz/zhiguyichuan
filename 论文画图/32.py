
import matplotlib.pyplot as plt
import numpy as np
import random

N=100
RMSECV=[random.randint(1, 1000) for _ in range(100)]
min_index = np.argmin(RMSECV)

with plt.style.context(('seaborn-whitegrid')):
    fig = plt.figure()
    fonts = 16
    plt.figure(figsize=(8, 6))
    plt.grid(False)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel('Monte Carlo iterations', fontsize=fonts)
    plt.ylabel('RMSECV', fontsize=fonts)
    plt.plot(np.arange(N) ,RMSECV)
    plt.axvline(x=min_index, color='r', linestyle='-')
    plt.show()