import numpy as np
from matplotlib import pyplot as plt
import math
import time
a = np.zeros((int(1000),int(1000)))
b = np.zeros((1000,1000))
for i in range(1000):
    for j in range(1000):
        a[i,j] = i*2+j*4.83214
        b[i,j] = i*275+j*7.1
        
for k in range(1000):
    np.dot(a,b)