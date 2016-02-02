import numpy as np

a = np.array([10,5,4,3])

b = np.argsort(a)
print a
print b
print(a[b])