import numpy as np
import torch

def forward(x,w1,b1,w2,b2):
    h_pre=w1@x+b1
    h=np.maximum(0,h_pre)
    out=w2@h+b2
    cache=(x,w2,h_pre,h)
    return out,cache
def backward(out,y,cache):
    x,w2,h_pre,h=cache

    dL_out=2.0*(out-y)
    dL_b2=dL_out.copy()
    dL_w2=dL_out.reshape(1,1)*h.reshape(1,3)

    dL_dh = (w2.T @ dL_out).reshape(3)

    dL_dh_pre = dL_dh * (h_pre > 0)

    dL_db1 = dL_dh_pre.copy()
    dL_dW1 = dL_dh_pre.reshape(3, 1) * x.reshape(1, 2)

    return dL_w2,dL_b2,dL_dW1,dL_db1
x  = np.array([1.0, -2.0])                 # (2,)
y  = 1.0                                   # 目标
W1 = np.array([[ 1.0, -1.0],
               [ 0.0,  2.0],
               [ 1.0,  1.0]])              # (3,2)
b1 = np.array([0.0, 0.0, 1.0])             # (3,)
W2 = np.array([[ 1.0,  0.0, -1.0]])        # (1,3)
b2 = np.array([0.5])                       # (1,)

out, cache = forward(x, W1, b1, W2, b2)
print('out =', out)                        # 期待 [3.5]
grads = backward(out, y, cache)
print('dL_dW1 =\n', grads[0])
print('dL_db1 =', grads[1])
print('dL_dW2 =', grads[2])
print('dL_db2 =', grads[3])