import numpy as np
from autograd.Tensor import Tensor
from autograd.Layer import ConvLayer



convL = ConvLayer(channelsIn=2)
X = Tensor(np.random.rand(32, 2, 5, 5).astype(np.float32))  # Example input
out = convL.forward(X)
print("Output shape:", out.data.shape)
out.backward()
print("Gradient shape:", convL.filters.grad.shape)