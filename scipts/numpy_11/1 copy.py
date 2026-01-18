import numpy as np

''' numpy 索引和切片与列表 list 是一样的【从 0 开始索引】 '''
lst2 = [[10, 20, 30],  # 0 索引  [10, 20, 30]   0 索引 10
        [40, 50, 60]]  # 1 索引  [40, 50, 60]   0 索引 40
# lst2[索引值]

''' 1. 索引：ndarray[索引] '''
arr2 = np.array(lst2)  # shape(2, 3)   axis(0, 1)
print(arr2[0])
print(arr2[0].shape)  # shape(3,)
# print(arr2[2])
print(arr2[0][2])

print(arr2[1, 1])  # 50  第二行的第二个值

''' 2. 切片：ndarray[起始索引 : 停止索引 : 步长, 索引] '''
# 注意：三个点 ... 切片只能使用一次 :
# ndarray[切片, 切片, 切片]    逗号维度减一

print(arr2[:1])
print(arr2[:1].shape)  # shape(1, 3)

arr3 = np.arange(60).reshape(4, 3, 5)
# print(arr3)
print(arr3[0:2])
print(arr3[::2])

print('#' * 50)
print(arr3[:, 0:2, ::2])
#  axis轴： 0全部,  1取0,1,  2取0,2,4

print('#' * 50)
print(arr3[..., ::2])  # axis = 2
print(arr3[:, :, ::2])

print(arr3[..., ::2, :])  # axis = -1 最后一个