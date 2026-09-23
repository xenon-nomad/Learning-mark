import torch
import torch.nn as nn
import numpy as np
class MF(nn.Module):
    def __init__(self,user_num,item_num,k=32,mu=3.0):
        super().__init__()
        self.p=nn.Embedding(user_num,k)
        self.q=nn.Embedding(item_num,k)
        self.bu=nn.Embedding(user_num,1)
        self.bi=nn.Embedding(item_num,1)
        self.mu=mu
    def forward(self,u,i):
        return ((self.p(u) * self.q(i)).sum(1)
                + self.bu(u).squeeze(1)
                + self.bi(i).squeeze(1)
                + self.mu)

