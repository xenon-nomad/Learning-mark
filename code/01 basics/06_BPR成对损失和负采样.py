import math
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from collections import defaultdict
def HR(rec_list,trust_set,k):
    Topk=set(rec_list[:k])
    return 1 if Topk&trust_set else 0
def Recall(rec_list,trust_set,k):
    Topk=set(rec_list[:k])
    hits=len(Topk&trust_set)
    return hits/len(trust_set)
def NDCG(rec_list,trust_set,k):
    dcg=0.0
    for rank,item in enumerate(rec_list[:k],start=1):
        if item in trust_set:
            dcg+=1/math.log2(rank+1)
    ideal_len=min(len(trust_set),k)
    idcg=0.0
    for r in range(1,ideal_len+1):
        idcg+=1/math.log2(r+1)

    return dcg/idcg

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# ===== R0/R1 数据（与 D04 完全一致，不多解释）=====
cols = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('ml-100k/u.data', sep='\t', names=cols, engine='python')
df['user_id'] -= 1
df['item_id'] -= 1
num_users = df['user_id'].max() + 1
num_items = df['item_id'].max() + 1

idx = np.random.permutation(len(df))
split = int(len(df) * 0.9)
train_df, val_df = df.iloc[idx[:split]], df.iloc[idx[split:]]

def to_tensor(d):
    return (torch.tensor(d['user_id'].values),
            torch.tensor(d['item_id'].values),
            torch.tensor(d['rating'].values, dtype=torch.float32))

train_u, train_i, train_r = to_tensor(train_df)
val_u, val_i, val_r = to_tensor(val_df)

user_items = defaultdict(set)
for u, i in zip(train_df['user_id'], train_df['item_id']):
    user_items[u].add(i)

def sample_negative(user_id):
    """随机抽物品，直到抽中用户没看过的——没看过的就是'负样本'"""
    while True:
        neg = random.randrange(num_items)
        if neg not in user_items[user_id]:
            return neg

def bpr_loss(pos_scores,neg_scores):
    return -torch.log(torch.sigmoid(pos_scores-neg_scores)).mean()

class MF(nn.Module):
    def __init__(self,num_users,item_users,dim=16):
        super().__init__()
        self.user=nn.Embedding(num_users,dim)
        self.item=nn.Embedding(item_users,dim)
        self.user_bias=nn.Embedding(num_users,1)
        self.item_bias=nn.Embedding(item_users,1)
        self.mu=3.52
        nn.init.normal_(self.user.weight,std=0.1)
        nn.init.normal_(self.item.weight,std=0.1)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self,u,i):
        pu=self.user(u)
        qi=self.item(i)
        dot=(pu*qi).sum(dim=1)
        return self.mu+self.user_bias(u).squeeze(1)+self.item_bias(i).squeeze(1)+dot
model=MF(num_users,num_items,dim=16)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
epochs=30

losses=[]
for epoch in range(epochs):
    model.train()
    train_neg= torch.tensor([sample_negative(u) for u in train_u.tolist()])
    pos_pred=model(train_u,train_i)
    neg_pred=model(train_u,train_neg)

    loss=bpr_loss(pos_pred,neg_pred)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())
    print(f'epoch {epoch + 1:2d} | BPR loss {losses[-1]:.4f}')

plt.plot(losses, label='BPR loss')
plt.xlabel('epoch'); plt.legend()
plt.savefig('bpr_loss.png', dpi=150)