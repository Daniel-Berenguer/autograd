import numpy as np
from autograd.Tensor import Tensor
from autograd.Layer import CNN

DATA_PATH = "data/mnist/"

train_data = np.loadtxt(DATA_PATH + "mnist_train.csv", delimiter=",")
test_data = np.loadtxt(DATA_PATH + "mnist_test.csv", delimiter=",")

Y_train = train_data[:, 0].astype(np.int32)
X_train = train_data[:, 1:].astype(np.float32).reshape(-1, 1, 28, 28) / 255.0  # Normalize pixel values
Y_test = test_data[:, 0].astype(np.int32)
X_test = test_data[:, 1:].astype(np.float32).reshape(-1, 1, 28, 28) / 255.0  # Normalize pixel values

# Convert labels to one-hot encoding
Y_train = np.eye(int(Y_train.max() + 1))[Y_train].astype(np.float32)
Y_test = np.eye(int(Y_test.max() + 1))[Y_test].astype(np.float32)

def categorical_cross_entropy(pred, target):
    return -(((target * (pred + 1e-9).log()).sum(axis=1)).mean(axis=0))

cnn = CNN(1, 10, nConvLayers=2, nFilters=[8, 16], kernel_shapes=[(3,3), (3,3)], paddings=[1,1], act="relu",
                  poolShapes=[(2,2), (2,2)], poolStrides=[2,2], LNin=784 , nLinLayers=2, nHidden=[128])

C = 10  # Number of classes, assuming labels are 0-indexed

N = X_train.shape[0]
MINIBATCH_SIZE = 32
LR = 0.1
EPOCHS = 4
ITERATIONS = (N // MINIBATCH_SIZE) * EPOCHS

for i in range(ITERATIONS+1):
    # build minibatch
    ix = np.random.randint(0, N, MINIBATCH_SIZE)
    X_batch = Tensor(X_train[ix])
    Y_batch = Tensor(Y_train[ix])
    if (i == round(ITERATIONS*0.7)):
        # Reduce learning rate after some iterations
        LR *= 0.1

    # forward pass
    out = cnn.forward(X_batch)
    loss = categorical_cross_entropy(out, Y_batch)
    if (i % 100 == 0):
        print(f"Epoch nº{i // (N // MINIBATCH_SIZE)}. Loss:", loss.data)
    # backward pass
    cnn.zero_grad()
    loss.backward()
    # update parameters
    for param in cnn.params:
        param.data -= LR * param.grad

# Evaluate on test set
X_test_tensor = Tensor(X_test)
out_test = cnn.forward(X_test_tensor)
loss_test = categorical_cross_entropy(out_test, Tensor(Y_test))
print(f"Test Loss: {loss_test.data}")

# Evaluate accuracy
predictions = out_test.data.argmax(axis=1)
accuracy = (predictions == Y_test.argmax(axis=1)).mean()
print(f"Test Accuracy: {accuracy * 100:.2f}%")

