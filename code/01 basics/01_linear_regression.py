import torch
import random


def synthetic_data(w,b,num_examples):
    x=torch.normal(0,1,(num_examples,len(w)))# x生成的正态分布的随机数
    y=torch.matmul(x,w)+b #表示X与w矩阵相乘的结果，其结果是一维的
    y+=torch.normal(0,0.01,y.shape)
    return x,y.reshape(-1,1)

true_w=torch.tensor([2,-3.4])
true_b=4.2
feature,label=synthetic_data(true_w,true_b,1000)
## 从零实现线性回归，所以不用打包好的函数，上面是数据生成部分

def data_iter(batch_size,feature,label):
    num_examples=len(feature)
    indices=list(range(num_examples))
    random.shuffle(indices)
    for i in range(0,num_examples,batch_size):
        batch_indices=torch.tensor(indices[i:min(i+batch_size,num_examples)])
        yield feature[batch_indices],label[batch_indices]

## 以上是读取数据，是采用了按批次读取数据的方法
def linreg(X,w,b):
    return torch.matmul(X,w)+b##定义模型

def squared_loss(y_hat,y):
    return (y_hat-y.reshape(y_hat.shape))**2/2
def sgd(params,lr,batch_size):
    with torch.no_grad():
        for param in params:
            param-=lr*param.grad/batch_size
            param.grad.zero_()

w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss
batch_size=10
for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, feature, label):
        l = loss(net(X, w, b), y)  # X和y的小批量损失
        # 因为l形状是(batch_size,1)，而不是一个标量。l中的所有元素被加到一起，
        # 并以此计算关于[w,b]的梯度
        l.sum().backward()
        sgd([w, b], lr, batch_size)  # 使用参数的梯度更新参数
    with torch.no_grad():
        train_l = loss(net(feature, w, b), label)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')