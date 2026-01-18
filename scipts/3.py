import numpy as np

arr1=np.arange(60).reshape(4,3,5)
print(f'arr1:\n{arr1}')
print('*'*30)
print(arr1[0:2])
print('-'*30)
print(arr1[::3])