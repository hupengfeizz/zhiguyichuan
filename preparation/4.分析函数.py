''' 分析函数： axis '''
import numpy as np

''' 1. max-最大、min-最小、sum-和、mean-平均 '''
np.random.seed(1)
arr3d = np.random.randint(10, 30,(2, 3, 4))
# print(arr3d)
# print(np.max(arr3d))

print('axis=0')
print(np.max(arr3d, axis=0))        # shape(axis=0, 3, 4)

print('axis=1')
print(np.max(arr3d, axis=1))        # shape(2, axis=1, 4)

print('axis=2')
print(np.max(arr3d, axis=2))        # shape(2, 3, axis=2)


''' 2. 比较大小：maximum、minimum '''
arr3 = np.random.randint(15, 25,(2, 3, 4))
print(arr3)
print(np.maximum(arr3d, arr3))
print(np.maximum(arr3d, 20))      # 广播


''' 3. average(数组,axis=None, weights=权重, returned=False) 
    weights = （值*权重+值*权重） / 权重和     returned=True
'''
print('average：加权平均，没有设置 weights 与 mean 一样')
# print(np.average(arr3, axis=0))
weight = np.arange(1, 25).reshape(2,3,4)
print(weight)
print(np.average(arr3, weights=weight, returned=True))

''' 4. median 求中位数，排序后，求中间 2 个的平均值'''
print(np.median(weight))    # (13+12)/2
print(np.median(weight, axis=1))
print(np.median(weight, axis=2))


''' 5. argmax, argmin 返回最小或最大值的索引 '''
print(arr3)
print(np.argmin(arr3,axis=1))

''' 6. bincount(1d数组, weights和) 统计一维数组中非负整数出现的次数 
    weights 与数组维度一样 1 维
'''
print('bincount(最大值)：返回一维数组 0~最大值')
arr = np.random.randint(5,10, 5)
print(arr)
# [9 5 6 8 9]
print(np.bincount(arr))
# [0 0 0 0 0 1 1 0 1 2]

weight = [0.1, 0.2, 0.3, 0.4, 0.5]
print(np.bincount(arr, weight))
# [0 0 0 0 0         1   1   0   1    2]
# [0.  0.  0.  0.  0.  0.2 0.3 0.  0.4  0.6]


''' 标准差 std、方差 var '''
arr2 = np.array([[10, 20, 30],
                 [40, 50,60]])
mean = np.mean(arr2)
# print(mean)     # 35.0    25  15  5  5 15  25
print(np.std(arr2, axis=0))   # (((10-(10+40)/2)**2 + (40-(10+40)/2)**2)/2)**0.5

print(np.std(arr2, axis=1))   # (10-(10+20+30)/3)**2 + (20-(10+20+30)/3)**2 + (30-(10+20+30)/3)**2


# cov 协方差
