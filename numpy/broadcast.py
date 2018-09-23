import numpy as np

a = np.array([[1,2,3],
              [2,3,4],
              [12,31,22],
              [2,2,2]])
b = np.array([1,2,3])

for i in range(4):
    a[i,:] += b

print(a)
# [[ 2  4  6]
#  [ 3  5  7]
#  [13 33 25]
#  [ 3  4  5]]
a + np.tile(b,(4,1))
print(a)

#广播
print(a + b)