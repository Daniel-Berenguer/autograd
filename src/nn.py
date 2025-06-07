import numpy as np
from autograd.Tensor import Tensor
from autograd.Layer import *
import pickle

# Load data
pickle_path = "data/iris/pickled/"
with open(pickle_path + "X_train.pkl", "rb") as f:
    X_train = pickle.load(f)

with open(pickle_path + "Y_train.pkl", "rb") as f:
    Y_train = pickle.load(f)

with open(pickle_path + "X_test.pkl", "rb") as f:
    X_test = pickle.load(f)

with open(pickle_path + "Y_test.pkl", "rb") as f:
    Y_test = pickle.load(f)


def categorical_cross_entropy(pred, target):
    return -(((target * (pred + 1e-9).log()).sum(axis=1)).mean(axis=0))


nn = MLP(4, 3, nLayers=2, nhidden=[15])

MINIBATCH_SIZE = 15
LR = 0.001

for i in range(100):
    # build minibatch
    ix = np.random.randint(0, X_train.shape[0], MINIBATCH_SIZE)
    X_batch = Tensor(X_train[ix])
    Y_batch = Tensor(Y_train[ix])

    # forward pass
    out = nn.forward(X_batch)
    print(out)
    loss = categorical_cross_entropy(out, Y_batch)
    print(f"Batch nº{i+1}. Loss:", loss.data)
    # backward pass
    nn.zero_grad()
    loss.backward()
    # update parameters
    for param in nn.params:
        param.data -= LR * param.grad
