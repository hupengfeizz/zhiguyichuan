import numpy as np

# sparse=False：返回单一数组
dense_result = np.indices((3, 4), sparse=False)
print("sparse=False 的输出:")
print(f"类型: {type(dense_result)}")
print(f"形状: {dense_result.shape}")  # (2, 3, 4)
print(f"包含: 2个完整的3×4网格数组堆叠在一起\n")

# sparse=True：返回元组
sparse_result = np.indices((3, 4), sparse=True)  
print("sparse=True 的输出:")
print(f"类型: {type(sparse_result)}")
print(f"长度: {len(sparse_result)}")  # 2
print(f"包含: 2个压缩的坐标数组组成的元组")