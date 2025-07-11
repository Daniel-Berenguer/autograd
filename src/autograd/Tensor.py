import numpy as np

class Tensor:
    """ Stores a tensor and its gradient """

    def __init__(self, data, _children=[]):
        self.data = data if isinstance(data, np.ndarray) else np.array(data, dtype=np.float32)

        self.grad = np.zeros_like(self.data, dtype=np.float32)

        if self.data.ndim == 0:
            self.data = self.data.reshape(1,1)
        if self.grad.ndim == 0:
            self.grad = self.grad.reshape(1,1)

        self._backward = lambda: None
        self._children = set(_children)
    
    def __repr__(self):
        return f"Tensor(data={self.data})"
    

    def backward(self):
        # We first perform a topological sort of the graph

        topo_order = []
        visited = set()
        def topological_sort(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    topological_sort(child)
                topo_order.append(v)
        topological_sort(self)

        # Then we compute the gradients in reverse order

        self.grad = np.ones_like(self.data, dtype=np.float32)
        for v in reversed(topo_order):
            v._backward()

    
    def __add__(self, other):
        if isinstance(other, Tensor):
            other_data = other.data
            if (self.data.shape != other.data.shape):
                if (self.data.shape[0] == other.data.shape[0] and other.data.shape[1] == 1):
                    other.data = np.tile(other.data, (1, self.data.shape[1]))
                elif (self.data.shape[1] == other.data.shape[1] and other.data.shape[0] == 1):
                    other_data = np.tile(other.data, (self.data.shape[0], 1))
                else:
                    raise ValueError("Shapes of tensors do not match for addition.")
        else:
            other = Tensor(np.full_like(self.data, other, dtype=np.float32))
            other_data = other.data

        out = Tensor(self.data + other_data, _children={self, other})


        def _backward():
            self.grad += out.grad
            if (other.grad.shape != self.grad.shape):
                if (other.grad.shape[0] == self.grad.shape[0] and self.grad.shape[1] == 1):
                    other.grad += out.grad.sum(axis=1)
                elif (other.grad.shape[1] == self.grad.shape[1] and other.grad.shape[0] == 1):
                    other.grad += out.grad.sum(axis=0)
            else:
                other.grad += out.grad
        out._backward = _backward

        return out
    
    def __sub__(self, other):
        if isinstance(other, Tensor):
            other_data = other.data
            if (self.data.shape != other.data.shape):
                if (self.data.shape[0] == other.data.shape[0] and other.data.shape[1] == 1):
                    other.data = np.tile(other.data, (1, self.data.shape[1]))
                elif (self.data.shape[1] == other.data.shape[1] and other.data.shape[0] == 1):
                    other_data = np.tile(other.data, (self.data.shape[0], 1))
                else:
                    raise ValueError("Shapes of tensors do not match for substraction.")
        else:
            other = Tensor(np.full_like(self.data, other, dtype=np.float32))
            other_data = other.data

        out = Tensor(self.data - other_data, _children={self, other})


        def _backward():
            self.grad += out.grad
            if (other.grad.shape != self.grad.shape):
                if (other.grad.shape[0] == self.grad.shape[0] and self.grad.shape[1] == 1):
                    other.grad -= out.grad.sum(axis=1)
                elif (other.grad.shape[1] == self.grad.shape[1] and other.grad.shape[0] == 1):
                    other.grad -= out.grad.sum(axis=0)
            else:
                other.grad -= out.grad
        out._backward = _backward

        return out
    
    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data, _children={self, other})

        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad
        out._backward = _backward

        return out
    
    def log(self):
        out = Tensor(np.log(self.data), _children={self})

        def _backward():
            self.grad += (1 / self.data) * out.grad
        out._backward = _backward

        return out
    
    def __pow__(self, exp):
        out = Tensor(self.data ** exp, _children={self})

        def _backward():
            self.grad += (exp * (self.data ** (exp - 1))) * out.grad
        
        out._backward = _backward

        return out
    
    def exp(self):
        out = Tensor(np.exp(self.data), _children={self})

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward

        return out
    
    def __mul__(self, other):
        if isinstance(other, Tensor):
            other_data = other.data
            if (self.data.shape != other.data.shape):
                if (self.data.shape[0] == other.data.shape[0] and other.data.shape[1] == 1):
                    other.data = np.tile(other.data, (1, self.data.shape[1]))
                elif (self.data.shape[1] == other.data.shape[1] and other.data.shape[0] == 1):
                    other_data = np.tile(other.data, (self.data.shape[0], 1))
                else:
                    raise ValueError("Shapes of tensors do not match for multiplication.")
        else:
            other = Tensor(np.full_like(self.data, other, dtype=np.float32))
            other_data = other.data

        out = Tensor(self.data * other_data, _children={self, other})

        def _backward():
            self.grad += out.grad * other.data
            if (other.grad.shape != self.grad.shape):
                if (other.grad.shape[0] == self.grad.shape[0] and self.grad.shape[1] == 1):
                    other.grad += (out.grad * self.data).sum(axis=1)
                elif (other.grad.shape[1] == self.grad.shape[1] and other.grad.shape[0] == 1):
                    other.grad += (out.grad * self.data).sum(axis=0)
            else:
                other.grad += out.grad * self.data
        out._backward = _backward

        return out
    

    def convolution(self, filtersT, padding=0):


        # Filters shape is (nFilters, channelsIn, kernel_height, kernel_width)
        # Data shape is (batch_size, channelsIn, height_in, width_in)
        # Out data is (batch_size, nFilters, height_out, width_out)

        filters = filtersT.data

        padded_data = np.pad(self.data, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant', constant_values=0)


        height_out = padded_data.shape[2] - filters.shape[2] + 1
        width_out = padded_data.shape[3] - filters.shape[3] + 1
        
        out_data = np.zeros((padded_data.shape[0], filters.shape[0], height_out, width_out), dtype=np.float32)

        for i in range(height_out):
            for j in range(width_out):

                # slice shape is (batch_size, channelsIn, kernel_height, kernel_width)
                # filters shape is (nFilters, channelsIn, kernel_height, kernel_width)
                # out_data shape is (batch_size, nFilters, height_out, width_out)

                slice = padded_data[:, :, i: i + filters.shape[2], j:j + filters.shape[3]]
                out_data[:, :, i, j] = (slice[:, np.newaxis, :, :, :] * filters[np.newaxis, :, :, :, :]).sum(axis=(2, 3, 4)) # sum over channelsIn, kernel_height, kernel_width

        out = Tensor(out_data, _children={self, filtersT})

        def _backward():
            # Gradient for the input and filters
            for i in range(height_out):
                for j in range(width_out):
                    slice = padded_data[:, :, i: i + filters.shape[2], j:j + filters.shape[3]]

                    padded_grad = np.pad(self.grad, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant', constant_values=0)
                    
                    new_self_grad = (out.grad[:, :, i, j, np.newaxis, np.newaxis, np.newaxis] * filters[np.newaxis, :, :, :, :]).sum(axis=1)  # sum over nFilters
                    
                    padded_grad[:, :, i : i + filters.shape[2], j : j + filters.shape[3]] += new_self_grad
                    self.grad = padded_grad[:, :, padding: -padding, padding: -padding]  # Remove padding
                    
                    filtersT.grad += (out.grad[:, :, i, j, np.newaxis, np.newaxis, np.newaxis] 
                                      * slice[:, np.newaxis, :, :, :]).sum(axis=0)  # sum over batch_size

        out._backward = _backward
        return out


        
    
    def __truediv__(self, other):
        if isinstance(other, Tensor):
            other_data = other.data
            if (self.data.shape != other.data.shape):
                if (self.data.shape[0] == other.data.shape[0] and other.data.shape[1] == 1):
                    other_data = np.tile(other.data, (1, self.data.shape[1]))
                elif (self.data.shape[1] == other.data.shape[1] and other.data.shape[0] == 1):
                    other_data = np.tile(other.data, (self.data.shape[0], 1))
                else:
                    raise ValueError("Shapes of tensors do not match for multiplication.")
        else:
            other = Tensor(np.full_like(self.data, other, dtype=np.float32))
            other_data = other.data

        out = Tensor(self.data / other_data, _children={self, other})

        def _backward():
            self.grad += out.grad * (1/other_data)
            if (other.grad.shape != self.grad.shape):
                if (other.grad.shape[0] == self.grad.shape[0] and (other.grad.ndim == 1 or other.grad.shape[1] == 1)):
                    other.grad = other.grad + (out.grad * (-self.data/ (other_data ** 2))).sum(axis=1, keepdims=True) 
                else:
                    other.grad = other.grad + (out.grad * (-self.data/ (other_data ** 2))).sum(axis=0, keepdims=True)
            else:
                other.grad += out.grad * (-self.data/ (other.data ** 2))    
        out._backward = _backward

        return out
    
    def sum(self, axis=0):
        if (axis == 1):
            out = Tensor(np.sum(self.data, axis=axis, keepdims=True), _children={self})
        else:
            out = Tensor(np.sum(self.data, axis=axis, keepdims=True), _children={self})


        if axis == 0:
            def _backward():
                self.grad = self.grad + np.repeat(out.grad, self.data.shape[0], axis=0)
        else:
            def _backward():
                self.grad = self.grad + np.repeat(out.grad, self.data.shape[1], axis=1)
            """"
            if (axis == 0):
                self.grad = np.tile(out.grad, (self.data.shape[0], 1))
            else:
                self.grad = np.tile(out.grad, (1, self.data.shape[1]))"""
            
            
        out._backward = _backward

        return out
    
    def __neg__(self):
        out = Tensor(-self.data, _children={self})

        def _backward():
            self.grad += -out.grad
        out._backward = _backward
        return out
    
    def mean(self, axis=0):
        out = self.sum(axis=axis) / self.data.shape[axis]
        return out
    
    def std(self, axis=0, mean=0):
        transf = (self - mean) ** 2
        out = transf.mean(axis=axis) ** 0.5
        return out
    
    def relu(self):
        out = Tensor(np.maximum(0, self.data), _children={self})

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out
    
    
    
    def softmax(self):
        sExp = self.exp()
        s = sExp.sum(axis=1)
        out = sExp / s
        return out