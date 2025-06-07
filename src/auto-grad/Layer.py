from Tensor import Tensor
import numpy as np

class LinearLayer:
    def __init__(self, nin, nout):
        self.nin = nin
        self.nout = nout
        self.W = Tensor(np.rand(nout, nin) * np.sqrt(2.0 / nin))  # He initialization
        self.bias = Tensor(np.zeros((nout, 1)))
        self.params = [self.W, self.bias]

    def forward(self, x):
        return (self.W @ x + self.bias).relu()
    

class MLP:
    def __init__(self, nin, nout, nLayers=3, nhidden=[64, 64]):
        self.layers = [
            LinearLayer(nin, nhidden[0])
        ]

        for i in range(1, nLayers-1):
            self.layers.append(LinearLayer(nhidden[i-1], nhidden[i]))

        self.layers.append(LinearLayer(nhidden[-1], nout))

        self.params = []
        for layer in self.layers:
            self.params.extend(layer.params)

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    

def categorical_cross_entropy(C, pred, target):
    for i in range(C):
        loss = -np.sum(target * np.log(pred + 1e-9))
    
    return Tensor(loss)