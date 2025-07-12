from .Tensor import Tensor
import numpy as np


class Layer:
    def __init__(self):
        self.params = []  # List to hold parameters of the layer

    def zero_grad(self):
        for param in self.params:
            param.grad = np.zeros_like(param.data, dtype=np.float32)

class LinearLayer(Layer):
    """
    A linear layer with optional batch normalization and activation function.
    """
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
    
class ConvLayer(Layer):
    def __init__(self, channelsIn=1, kernel_shape=(3,3), padding=1, nFilters=8, act="relu"):
        kernel_height, kernel_width = kernel_shape
        self.padding = padding
        self.nFilters = nFilters
        self.filtersShape = (nFilters, channelsIn, kernel_height, kernel_width)
        self.filters = Tensor((np.random.rand(*self.filtersShape).astype(np.float32) 
                               - 0.5) * np.sqrt(2.0 / (kernel_height * kernel_width * channelsIn))) # Kaiming He init
        self.act = act
        self.params = [self.filters]  # List to hold parameters of the layer


    def forward(self, X, training=None):
        # Assuming X is a 4D tensor with shape (batch_size, channelsIn, height_in, width_in)
        out = X.convolution(self.filters, padding=self.padding)

        if self.act == "relu":
            out = out.relu()
        elif self.act == "softmax":
            out = out.softmax()

        return out
    
class MaxPoolLayer(Layer):
    def __init__(self, kernel_size=(2,2), stride=2):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, X, training=None):
        # Assuming X is a 4D tensor with shape (batch_size, channels, height, width)
        out = X.maxPool(kernel_shape=self.kernel_size, stride=self.stride)
        return out

class Flatten(Layer):
    def forward(self, X, training=None):
        out = X.flatten()  # Flatten the input tensor
        return out


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

    def forward(self, x, training=True):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()



class Sequential:
    def __init__(self, *layers):
        self.layers = layers
        self.params = []
        for layer in layers:
            self.params.extend(layer.params)

    def forward(self, x, training=True):
        for layer in self.layers:
            x = layer.forward(x, training=training)
        return x
    
    def zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()


class CNN:
    def __init__(self, channelsIn, nOut, nConvLayers=2, nFilters=[8, 16], kernel_shapes=[(3,3), (3,3)], paddings=[1,1], act="relu",
                  poolShapes=[(2,2), (2,2)], poolStrides=[2,2], LNin=784 , nLinLayers=2, nHidden=[128]):
        self.layers = []

        for i in range(nConvLayers):
            if (i > 0):
                channelsIn = nFilters[i-1]  # Update channelsIn for subsequent layers
            self.layers.append(ConvLayer(channelsIn=channelsIn, nFilters=nFilters[i], kernel_shape=kernel_shapes[i],
                                          padding=paddings[i], act=act))
            self.layers.append(MaxPoolLayer(kernel_size=poolShapes[i], stride=poolStrides[i]))

        # Flatten layer
        self.layers.append(Flatten())

        # Add linear layers
        self.layers.append(LinearLayer(LNin, nHidden[0], act="relu", batchNorm=True))
        for i in range(1, nLinLayers-1):
            self.layers.append(LinearLayer(nHidden[i-1], nHidden[i], act="relu", batchNorm=True))

        # Output layer
        self.layers.append(LinearLayer(nHidden[-1], nOut, act="softmax", batchNorm=False))

        # Collect all parameters from the layers
        self.params = []
        for layer in self.layers:
            self.params.extend(layer.params)

    def forward(self, x, training=True):
        for layer in self.layers:
            x = layer.forward(x, training=training)
        return x
    
    def zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()

    