import numpy as np
import torch
def softmax(z):
    z = np.array(z, dtype=float)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()

def softmax_loss(z,correct):
    p=softmax(z)
    return -np.log(p[correct])
def infonce(S):
    n = S.shape[0]
    total = 0.0
    for i in range(n):
        total += softmax_loss(S[i], correct=i)   # 第 i 行的正确类 = 第 i 个物品
    return total / n
##最有意思的设计就是batch的矩阵是把i用户的正确类别等于i