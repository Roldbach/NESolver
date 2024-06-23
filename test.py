import numpy as np


test = np.arange(9).reshape((3,3))
test = np.array([1,2,3]).reshape((1,3)) * test
print(test)
