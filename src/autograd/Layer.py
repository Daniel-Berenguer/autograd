from .Tensor import Tensor
import numpy as np

class LinearLayer:
    def __init__(self, nin, nout, act="relu", batchNorm=True):
        self.nin = nin
        self.nout = nout
        self.W = Tensor((np.random.rand(nin, nout).astype(np.float32) - 0.5) * np.sqrt(2.0 / nin))  # Kaiming He init
        self.batchNorm = batchNorm
        self.bias = Tensor(np.zeros((1, nout), dtype=np.float32))  # Bias initialized to zero

        if batchNorm:      
            self.gain = Tensor(np.ones((1, nout), dtype=np.float32))

            # Initialized running mean and std as numpy arrays as no grads are required
            self.runningMean = np.zeros((1, nout), dtype=np.float32)
            self.runningStd = np.ones((1, nout), dtype=np.float32)

            self.params = [self.W, self.bias, self.gain]

        else:
            self.params = [self.W, self.bias]

        

        self.act = act

    def forward(self, X, training=True):
        out = X @ self.W

        if (not self.batchNorm):
            out = out + self.bias
        
        else:
            if training:
                batchMean = out.mean(axis=0)
                batchStd = out.std(axis=0, mean=batchMean)
                out = (out - batchMean) / (batchStd + 1e-5) * self.gain + self.bias

                # Update running mean and std
                self.runningMean = 0.99 * self.runningMean + 0.01 * batchMean.data
                self.runningStd = 0.99 * self.runningStd + 0.01 * batchStd.data

            else:
                # Use running mean and std for inference
                out = (out - self.runningMean) / (self.runningStd + 1e-5) * self.gain + self.bias

        
        if self.act == "relu":
            out = out.relu()
        elif self.act == "softmax":
            out = out.softmax()
        return out
    
    def zero_grad(self):
        for param in self.params:
            param.grad = np.zeros_like(param.data, dtype=np.float32)
    

class MLP:
    def __init__(self, nin, nout, nLayers=3, nhidden=[64, 64], batchNorm=True):
        self.layers = [
            LinearLayer(nin, nhidden[0], "relu", batchNorm)  # First layer with ReLU activation
        ]

        for i in range(1, nLayers-1):
            self.layers.append(LinearLayer(nhidden[i-1], nhidden[i], "relu", batchNorm))

        self.layers.append(LinearLayer(nhidden[-1], nout, "softmax", batchNorm=False))  # Last layer with softmax activation

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