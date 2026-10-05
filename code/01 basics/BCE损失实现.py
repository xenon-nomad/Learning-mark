import numpy as np
import torch
def bce_loss(p,y):
    eps=1e-7
    return -(y*np.log(p+eps)+(1-y)*np.log(1-p+eps)).mean()
ps = np.array([0.9, 0.5, 0.1])
ys = np.array([1.0, 1.0, 1.0])
print('手写 BCE:', bce_loss(ps, ys))