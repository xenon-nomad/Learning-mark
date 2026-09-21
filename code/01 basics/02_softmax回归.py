import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.2860,), (0.3530,))])
train_set = datasets.FashionMNIST("./data", train=True, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=256, shuffle=True)


def softmax(X):
    X_exp=torch.exp(X)
    partition=X_exp.sum(1,keepdim=True)
    return X_exp/partition

def Cross_entropy(y_hat,y):
    return -torch.log(y_hat[range(len(y_hat)),y])
def sgd(params,lr,batch_size):
    with torch.no_grad():
        for param in params:
            param-=lr*param.grad/batch_size
            param.grad.zero_()
num_inputs, num_outputs = 784, 10
W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)
def net(X):
    return softmax(torch.matmul(X.reshape(-1,W.shape[0]),W)+b)
lr,epochs=0.1,10
for epoch in range(epochs):
    total_loss=0
    for X,y in train_loader:
        l=Cross_entropy(net(X),y)
        l.sum().backward()
        sgd([W,b],lr,len(y))
        total_loss+=l.sum().item()
    print(f'epoch {epoch + 1} done,loss={total_loss/len(train_set)}')