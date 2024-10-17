import numpy as np

temp=np.random.rand(1,3,900,900)*6-3
print(temp)
np.save('patch_rand.npy',temp)