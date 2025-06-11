import numpy as np
from autograd.Tensor import Tensor
from autograd.Layer import LinearLayer, MLP

DATA_PATH = "data/mnist/"

#with open(DATA_PATH + "mnist_train.csv", "r") as f:
    #train_data = np.array([line.strip().split(",") for line in f.readlines()])

train_data = np.loadtxt(DATA_PATH + "mnist_train.csv", delimiter=",")
test_data = np.loadtxt(DATA_PATH + "mnist_test.csv", delimiter=",")

Y_train = train_data[:, 0].astype(np.int32)
X_train = train_data[:, 1:].astype(np.float32) / 255.0  # Normalize pixel values
Y_test = test_data[:, 0].astype(np.int32)
X_test = test_data[:, 1:].astype(np.float32) / 255.0  # Normalize pixel values

# Convert labels to one-hot encoding
Y_train = np.eye(int(Y_train.max() + 1))[Y_train].astype(np.float32)
Y_test = np.eye(int(Y_test.max() + 1))[Y_test].astype(np.float32)


print("Training data shape:", X_train.shape, Y_train.shape)

def categorical_cross_entropy(pred, target):
    return -(((target * (pred + 1e-9).log()).sum(axis=1)).mean(axis=0))

nin = X_train.shape[1]
C = 10  # Number of classes, assuming labels are 0-indexed

print(nin, C)

nn = MLP(nin, C, nLayers=5, nhidden=[64, 64, 32, 32], batchNorm=True)

N = X_train.shape[0]
MINIBATCH_SIZE = 32
LR = 0.15
EPOCHS = 10
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
    out = nn.forward(X_batch)
    loss = categorical_cross_entropy(out, Y_batch)
    if (i % 100 == 0):
        print(f"Epoch nº{i // (N // MINIBATCH_SIZE)}. Loss:", loss.data)
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
