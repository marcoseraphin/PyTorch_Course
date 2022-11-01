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
