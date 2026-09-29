
import numpy as np

np.random.seed(42)

class MyEmbedding:
    def __init__(self, num_embeddings, embedding_dim):
        self.W = np.random.randn(num_embeddings, embedding_dim) * 0.1
        self.cache = None

    def forward(self, idx):
        self.cache = idx
        return self.W[idx]

    def backward(self, grad_out):
        dW = np.zeros_like(self.W)
        for row, g in zip(self.cache, grad_out):
            dW[row] += g          # += 而非 =：重复索引必须叠加，= 会静默覆盖
        return dW


W = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
idx = [2, 0, 2]
grad_out = np.array([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]])

emb = MyEmbedding(3, 2)
emb.W = W.copy()
print('forward:\n', emb.forward(idx))
print('backward dW:\n', emb.backward(grad_out))


