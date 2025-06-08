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


nn = MLP(4, 3, nLayers=3, nhidden=[10, 10])

N = X_train.shape[0]
MINIBATCH_SIZE = 5
LR = 0.01
EPOCHS = 80
ITERATIONS = (N // MINIBATCH_SIZE) * EPOCHS

for i in range(ITERATIONS):
    # build minibatch
    ix = np.random.randint(0, X_train.shape[0], MINIBATCH_SIZE)
    X_batch = Tensor(X_train[ix])
    Y_batch = Tensor(Y_train[ix])
    if (i > ITERATIONS // 2):
        # Reduce learning rate after half of the iterations
        LR = 0.001

    # forward pass
    out = nn.forward(X_batch)
    loss = categorical_cross_entropy(out, Y_batch)
    if (i % 10 == 0):
        print(f"Batch nº{i+1}. Loss:", loss.data)
    # backward pass
    nn.zero_grad()
    loss.backward()
    # update parameters
    for param in nn.params:
        param.data -= LR * param.grad

# Evaluate on test set
X_test_tensor = Tensor(X_test)
out_test = nn.forward(X_test_tensor)
loss_test = categorical_cross_entropy(out_test, Tensor(Y_test))
print(f"Test Loss: {loss_test.data}")

# Evaluate accuracy
predictions = out_test.data.argmax(axis=1)
accuracy = (predictions == Y_test.argmax(axis=1)).mean()
print(f"Test Accuracy: {accuracy * 100:.2f}%")
