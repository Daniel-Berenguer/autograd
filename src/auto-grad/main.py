import numpy as np

class Tensor:
    """ Stores a tensor and its gradient """

    def __init__(self, data, _children=None):
        self.data = data
        self.grad = 0
        self._backward = lambda: None
        self._children = set(_children)
    
    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"
    

    def backward(self):
        # TODO - implement topological sort and backward pass
        return None

    
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, _children={self, other})
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data, _children={self, other})
        return out