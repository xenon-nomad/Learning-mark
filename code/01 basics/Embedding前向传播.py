import numpy as np
np.random.seed(0)
def linear(w,b,x):
    return w@x+b
def ReLu(x):
    return np.maximum(0,x)
x  = np.array([1.0, -2.0])

W1 = np.array([[ 1.0, -1.0],
               [ 0.0,  2.0],
               [ 1.0,  1.0]])
b1 = np.array([0.0, 0.0, 1.0])

W2 = np.array([[ 1.0,  0.0, -1.0]])
b2 = np.array([0.5])

pre=linear(W1,b1,x)
relu=ReLu(pre)
out=linear(W2,b2,relu)

print("pre=",pre)
print("mid=",relu)
print("out=",out)
