import numpy as np

weight=np.arange(1,25).reshape(2,3,4)

print(np.median(weight,axis=0))