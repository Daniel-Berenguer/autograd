from .Tensor import Tensor
import numpy as np

class LinearLayer:
    def __init__(self, nin, nout, act="relu"):
        self.nin = nin
        self.nout = nout
        self.W = Tensor(np.random.rand(nin, nout).astype(np.float32) * np.sqrt(2.0 / nin))  # He initialization
        self.bias = Tensor(np.zeros((1, nout), dtype=np.float32))  # Bias initialized to zero
        self.params = [self.W, self.bias]
        self.act = act

    def forward(self, X):
        out = X @ self.W + self.bias
        
        if self.act == "relu":
            out = out.relu()
        elif self.act == "softmax":
            out = out.softmax()

        return out
    
    def zero_grad(self):
        for param in self.params:
            param.grad = np.zeros_like(param.data, dtype=np.float32)
    

class MLP:
    def __init__(self, nin, nout, nLayers=3, nhidden=[64, 64]):
        self.layers = [
            LinearLayer(nin, nhidden[0])
        ]

        for i in range(1, nLayers-1):
            self.layers.append(LinearLayer(nhidden[i-1], nhidden[i]))

        self.layers.append(LinearLayer(nhidden[-1], nout, "softmax"))  # Last layer with softmax activation

        # Collect all parameters from the layers
        self.params = []
        for layer in self.layers:
            self.params.extend(layer.params)

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()