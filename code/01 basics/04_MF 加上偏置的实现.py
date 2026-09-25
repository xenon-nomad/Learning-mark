import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os, urllib.request, zipfile
import ssl
from torch.utils.data import DataLoader, TensorDataset
ssl._create_default_https_context = ssl._create_unverified_context
if not os.path.exists("./ml-100k/u.data"):
    urllib.request.urlretrieve(
        "https://files.grouplens.org/datasets/movielens/ml-100k.zip", "ml-100k.zip")
    zipfile.ZipFile("ml-100k.zip").extractall(".")
torch.manual_seed(42)
np.random.seed(42)
###-------------前面是数据的导入和划分，固定的API，先skip----------------###
# ===== R0 读数据 =====
# u.data：tab 分隔、无表头，4 列：user_id, item_id, rating, timestamp
cols = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv("./ml-100k/u.data", sep="\t",
                 names=["user_id", "item_id", "rating", "ts"], engine="python")


df['user_id'] -= 1
df['item_id'] -= 1
num_users = df['user_id'].max() + 1   # 943
num_items = df['item_id'].max() + 1   # 1682
mu = df['rating'].mean()              # 全局均值 μ（约 3.52）

# ===== R1 划分 90/10（固定种子已设，可复现）=====
idx = np.random.permutation(len(df))
split = int(len(df) * 0.9)
train_df, val_df = df.iloc[idx[:split]], df.iloc[idx[split:]]

def to_tensor(d):
    return (torch.tensor(d['user_id'].values),
            torch.tensor(d['item_id'].values),
            torch.tensor(d['rating'].values, dtype=torch.float32))

train_u, train_i, train_r = to_tensor(train_df)
val_u, val_i, val_r = to_tensor(val_df)

# ===== R2 模型 =====
class MF(nn.Module):
    def __init__(self, num_users, num_items, dim=64, use_bias=True):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, dim)
        self.item_emb = nn.Embedding(num_items, dim)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        self.use_bias = use_bias
        self.mu = mu
        nn.init.normal_(self.user_emb.weight, std=0.1)
        nn.init.normal_(self.item_emb.weight, std=0.1)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, u, i):
        pu = self.user_emb(u)
        qi = self.item_emb(i)
        dot = (pu * qi).sum(dim=1)
        pred = self.mu + dot
        if self.use_bias:
            bu = self.user_bias(u).squeeze(1)   # (N, 1) → (N,) 和点积对齐
            bi = self.item_bias(i).squeeze(1)
            pred = pred + bu + bi
        return pred

model = MF(num_users, num_items, dim=16, use_bias=True)

# ===== R4 RMSE =====
@torch.no_grad()
def rmse(model, u, i, r):
    model.eval()
    pred = model(u, i)
    return torch.sqrt(((pred - r) ** 2).mean())

# ===== R3 训练（全量批：这个数据规模 CPU 秒级，先求跑通）=====
loss_fn = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.005,weight_decay=0.01)
epochs = 150
train_curve, val_curve = [], []
for epoch in range(epochs):
    model.train()

    pred = model(train_u, train_i)
    loss = loss_fn(pred, train_r)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    train_curve.append(rmse(model, train_u, train_i, train_r).item())
    val_curve.append(rmse(model, val_u, val_i, val_r).item())
    print(f'epoch {epoch+1:2d} | train RMSE {train_curve[-1]:.4f} | val RMSE {val_curve[-1]:.4f}')

# ===== R5 曲线 =====
plt.plot(train_curve, label='train')
plt.plot(val_curve, label='val')
plt.xlabel('epoch'); plt.ylabel('RMSE'); plt.legend()
plt.savefig('mf_curve.png', dpi=150)
print('best val RMSE:', min(val_curve))