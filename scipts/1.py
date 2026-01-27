import numpy as np

arr = np.random.randint(5,10, 5)
print(arr)
# [9 5 6 8 9]
print(np.bincount(arr))