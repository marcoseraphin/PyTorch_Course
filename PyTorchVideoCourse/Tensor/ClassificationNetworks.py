from turtle import forward
import torch
import sklearn
from torch import Tensor, nn # nn contains all of PyTorch's building blocks for neural networks
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import pandas as pd

# Neural Network classification

# Make 1000 circles
n_samples = 1000

# Create circles

# https://docs.w3cub.com/scikit_learn/modules/generated/sklearn.datasets.make_circles
X, y = make_circles(n_samples=n_samples,
                     noise=0.03,
                     random_state=42)

print(f"First 5 samples of X:\n {X[:5]}")
print(f"First 5 samples of y:\n {y[:5]}")

# Make a DataFrame of circle data
circles = pd.DataFrame({"X1" : X[:, 0],
                        "X2" : X[:, 1],
                        "label": y})

print(f"circles.head(10) => {circles.head(10)}")

# Visualize circles
plt.scatter(x=X[:, 0],
            y=X[:, 1],
            c=y,
            cmap=plt.cm.RdYlBu);
#plt.show()

# Turn data into tensors
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

print(f"Feastures Tensor X => {X[:5]}")
print(f"Label y => {y[:5]}")

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.2,     # 20% test data, 80% train data
                                                    random_state=42)   # random seed

print(f"Length of X_train => {len(X_train)}")
print(f"Length of X_test  => {len(X_test)}")

# Build a model to classify the blue and red dots

# Set the model to use the target device MPS (GPU)
# Set the device      
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

class CircleModelv1(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=5) # takes in 2 input features and upscales to 5 output features
        self.layer_2 = nn.Linear(in_features=5, out_features=1) # input 5 input features and 1 output features

    def forward(self, x):
        return self.layer_2(self.layer_1(x)) # pass data thru layer 1 and than layer 2

model_0 = CircleModelv1().to(device=device)

print(f"CircleModelv1 model_0 => {model_0}")
print(f"Device of CircleModelv1 model_0 => {next(model_0.parameters()).device}")

# using nn.Sequential()
model_0 = nn.Sequential(nn.Linear(in_features=2, out_features=5),
                        nn.Linear(in_features=5, out_features=1).to(device=device))

print(f"Sequential CircleModelv1 model_0 => {model_0}")

# Make predictions
print(f"State Dict of CircleModelv1 model_0 => {model_0.state_dict()}")

model_0.to(device=device)

with torch.inference_mode():
    untrained_preds = model_0(X_test.to(device))
    print(f"untrained_preds of CircleModelv1 model_0 => {untrained_preds[:5]}")

# Setup loss function and optimizer (for classification use cross entropy)
loss_fn = nn.BCEWithLogitsLoss() # sigmoid activation function built-in

optimizer = torch.optim.SGD(model_0.parameters(), lr=0.1)

# Calculate accuracy 
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc


