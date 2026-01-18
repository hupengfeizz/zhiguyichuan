import numpy as np

# dense（默认）- 返回完整的网格数组
dense_indices = np.indices((3, 4), sparse=False)
print("密集格式形状:", dense_indices.shape)
print(f'dense_indices:\n{dense_indices}')

# sparse - 返回稀疏格式，节省内存
sparse_indices = np.indices((3, 4), sparse=True)
print(f'sparse_indices\n{sparse_indices}')
print("稀疏格式:")
for i, arr in enumerate(sparse_indices):
    print(f"维度 {i}: shape={arr.shape}, {arr}")