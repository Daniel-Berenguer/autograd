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
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, _children={self, other})

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
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