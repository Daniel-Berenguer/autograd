import numpy as np
from autograd.Tensor import Tensor
from autograd.Layer import *




def categorical_cross_entropy(pred, target):
    return -(((target * (pred + 1e-9).log()).sum(axis=1)).mean(axis=0))


X = Tensor(np.random.rand(10, 15).astype(np.float32))
nn = MLP(15, 3, nLayers=3, nhidden=[32, 16])

out = nn.forward(X)
Y_true = Tensor(np.array([[1,0,0], [0,1,0], [0,0,1], [1,0,0], [0,1,0], [0,0,1], [1,0,0], [0,1,0], [0,0,1], [1,0,0]], dtype=np.float32))
loss = categorical_cross_entropy(out, Y_true)
loss.backward()
print(loss)
print("Gradients:")
for param in nn.params:
    print(f"g: {param.grad}")