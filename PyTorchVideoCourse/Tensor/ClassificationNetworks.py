from turtle import forward
import requests
from pathlib import Path
import torch
import sklearn
from torch import Tensor, nn # nn contains all of PyTorch's building blocks for neural networks
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import pandas as pd
from helper_functions import plot_predictions, plot_decision_boundary

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


# Logits
y_logits = model_0(X_test.to(device))[:5]
print(f"y_logits of model_0 => {y_logits}")

# Use the sigmiod actication function on our model logits to turn them into prediction probabilities
y_pred_probs = torch.sigmoid(y_logits)
print(f"Rounded y_pred_probs of model_0 => {torch.round(y_pred_probs)}")

# Find predicted lablels
y_preds = torch.round(y_pred_probs)

# Full (logits => pred probs => pred labels)
y_pred_labels = torch.round(torch.sigmoid(model_0(X_test.to(device))[:5]))

# Check equality
print(f"y_preds equal y_pred_labels of model_0 => {torch.eq(y_preds.squeeze(), y_pred_labels.squeeze())}")
print(f"y_preds of model_0 => {torch.squeeze(y_preds)}")

# Train the model
torch.manual_seed(42)

epochs = 100

# Put data to the MPS
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

for epoch in range(epochs):
    model_0.train()

    # Forward pass
    y_logits = model_0(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))

    # Calculate loss and accuarcy (loss_fn is BCEWithLogitsLoss, so it requires logits as input)
    loss = loss_fn(y_logits, 
                   y_train)

    acc = accuracy_fn(y_true=y_train,
                      y_pred=y_pred)

    # Optimizer zero grad
    optimizer.zero_grad()

    # Loss backward
    loss.backward()

    # Optimizer step (gradient descent)
    optimizer.step()

    # Testing
    model_0.eval()
    with torch.inference_mode():
        test_logits = model_0(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))

        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(y_true=y_test, y_pred=test_pred)

    if epoch % 10 == 0:
        print(f"EPoch {epoch} | Loss: {loss:.5f}, Acc: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")

# Make predictions
if Path("helper_functions.py").is_file():
    print("helper_functions.py is already downloaded")
else:
    print("Donwloading helper_functions.py")
    request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
    with open("helper_functions.py", "wb") as f:
        f.write(request.content)

# Plot decision boundary of the model
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.title("Train")
plot_decision_boundary(model_0, X_train, y_train)
plt.subplot(1,2,2)
plt.title("Test")
plot_decision_boundary(model_0, X_test, y_test)
plt.show()

# Non-linear Model
class CircleModelNonLinear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=10)
        self.layer_2 = nn.Linear(in_features=10, out_features=10)
        self.layer_3 = nn.Linear(in_features=10, out_features=1)
        self.relu = nn.ReLU()  # ReLU is a non-linear activation function

    def forward(self, x):
        return self.layer_3(self.relu(self.layer_2(self.layer_1(x))))

model_nonlinear = CircleModelNonLinear().to(device=device)

# Setup loss function and optimizer (for classification use cross entropy)
loss_fn_nonlinear = nn.BCEWithLogitsLoss() # sigmoid activation function built-in
optimizer_nonlinear = torch.optim.SGD(model_nonlinear.parameters(), lr=0.1)

# Train the non-linear model
torch.manual_seed(42)

epochs_nonlinear = 2000

for epoch_nonlinear in range(epochs_nonlinear):
    model_nonlinear.train()

    # Forward pass
    y_logits_nonlinear = model_nonlinear(X_train).squeeze()
    y_pred_nonlinear = torch.round(torch.sigmoid(y_logits_nonlinear))

    # Calculate loss and accuarcy (loss_fn is BCEWithLogitsLoss, so it requires logits as input)
    loss_nonlinear = loss_fn(y_logits_nonlinear, 
                             y_train)

    acc_nonlinear = accuracy_fn(y_true=y_train,
                                y_pred=y_pred_nonlinear)

    # Optimizer zero grad
    optimizer_nonlinear.zero_grad()

    # Loss backward
    loss_nonlinear.backward()

    # Optimizer step (gradient descent)
    optimizer_nonlinear.step()

    # Testing
    model_nonlinear.eval()
    with torch.inference_mode():
        test_logits_nonlinear = model_nonlinear(X_test).squeeze()
        test_pred_nonlinear = torch.round(torch.sigmoid(test_logits_nonlinear))

        test_loss_nonlinear = loss_fn(test_logits_nonlinear, y_test)
        test_acc_nonlinear = accuracy_fn(y_true=y_test, y_pred=test_pred_nonlinear)

    if epoch_nonlinear % 10 == 0:
        print(f"EPoch Non-linear : {epoch_nonlinear} | Loss: {loss_nonlinear:.5f}, Acc: {acc_nonlinear:.2f}% | Test loss: {test_loss_nonlinear:.5f}, Test acc: {test_acc_nonlinear:.2f}%")


# Plot decision boundary of the non-linear model
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.title("Train")
plot_decision_boundary(model_nonlinear, X_train, y_train)
plt.subplot(1,2,2)
plt.title("Test")
plot_decision_boundary(model_nonlinear, X_test, y_test)
plt.show()
