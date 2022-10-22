from cProfile import label
from pickle import NONE
import torch
from torch import nn # nn contains all of PyTorch's building blocks for neural networks
import matplotlib.pyplot as plt

print(f"PyTorch Version {torch.__version__}")

#  1. Data (preparing and loading)
# Get Data into numerical representation
# Build a model to learn patterns in the numerical representation

# To showcase this, let's create some know data using the linear regression formular
# Create kwon parameters

weight = 0.7
bias = 0.3

start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias

print(f"X[:10] => {X[:10]}")
print(f"y[:10] => {y[:10]}")

# Splitting data into traing and test sets
train_split = int(0.8 * len(X))
print(f"80% Training Set => {train_split}")

X_train = X[:train_split]
y_train = X[:train_split] # first data (80%) for training

X_test = X[train_split:]
y_test = X[train_split:]  # rest data (20%) for testing

print(f"Length Training Set X_train => {len(X_train)}")
print(f"Length Training Set y_train => {len(y_train)}")

print(f"Length Training Set X_test => {len(X_test)}")
print(f"Length Training Set y_test => {len(y_test)}")

# Visualize data

def plot_predictions(train_data=X_train,
                     train_labels=y_train,
                     test_data= X_test,
                     test_labels=y_test,
                     predictions=None):

    plt.figure(figsize=(10,7))

    # Plot training data in blue
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")

    # Plot training data in green
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

    # Are there predictions ?
    if predictions is not None:
        # Plot predictions
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

    # Show legend
    plt.legend(prop={"size" : 14});

#plot_predictions()
#plt.show()

# 2. Build model

# start with random values (weight and bias)
# look at training data and adjust the random values to better represent the ideal values
# weight and bias values with we created the values

# Using two main algorithms
# 1. Gradient descent
# 2. back propagation

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1,requires_grad=True,
                                                 dtype=torch.float))

        self.bias = nn.Parameter(torch.randn(1, requires_grad=True,
                                                dtype=torch.float))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x + self.bias  # linear regression formula

# Create a random seed
torch.manual_seed(42)

# Create an instance of the model
model_0 = LinearRegressionModel()

# Print out the parameters
print(f"LinearRegressionModel parameters => {list(model_0.parameters())} ")
print(f"LinearRegressionModel parameters => {model_0.state_dict()} ")

# Making predictions using torch.inference_mode()

with torch.inference_mode(): # NO gradient descent with inference mode for making predictions
    y_preds = model_0(X_test)

print(f"y_preds => {y_preds} ")

plot_predictions(predictions=y_preds)
plt.show()

# 3. Train model


