from . import Tensor
import numpy as np

if __name__ == "__main__":
    print("Hello, AutoGrad!")

    W = Tensor(np.array([1, 2, 3]).reshape(1, 3))
    x = Tensor(np.array([1, 2, 3]).reshape(3, 1)) # Reshape to a column vector
    b = Tensor(3) # Reshape to a column vector
    y = W @ x + b
    y.backward()
    print(W.grad)
