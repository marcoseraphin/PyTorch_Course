from cProfile import label
from pickle import NONE
from pyexpat import model
from turtle import forward
import torch
from torch import Tensor, nn # nn contains all of PyTorch's building blocks for neural networks
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

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

plot_predictions()
plt.show()

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

# Setup a loss function
loss_fn = nn.L1Loss()

# Setup an optimizer (stochastic gradient descent)
optimizer = torch.optim.SGD(model_0.parameters(),
                            lr=0.01) # learning rate

# Building a training loop (and a testing loop)
epochs = 300  # 300 loop thru the data

# Tracking different learning values
epoch_count = []
loss_values = []
test_loss_values = []

# loop thru the data
for epoch in range(epochs):

    # Set the model to training mode
    model_0.train() # train mode in PytTorch sets all parameters that require gradients to gradients

    # Forward pass the data thru the model
    y_pred = model_0(X_train)

    # Calculate the loss (compare predictions with train target data)
    loss = loss_fn(y_pred, y_train)
    #print(f"Loss value = {loss}")

    #  Optimizer zero grad (reset optimizer value )
    optimizer.zero_grad() 

    # Perform backpropagation on the loss with respect to the paramters of the model
    loss.backward()

    # Step the optimizer (perform gradient descent)
    optimizer.step()

    # Testing 
    model_0.eval() 
    with torch.inference_mode():  # turns off gradient tracking
        # Forward pass test data
        test_pred = model_0(X_test)

        # Calculate the test loss
        test_loss = loss_fn(test_pred, y_test)
    
    if epoch % 10 == 0:
        epoch_count.append(epoch)
        loss_values.append(loss)
        test_loss_values.append(test_loss)

        print(f"Epoch: {epoch} | Loss: {loss} | Test loss: {test_loss}")


#print(f"LinearRegressionModel parameters => {model_0.state_dict()} ")

# Plot the loss curves
#plt.plot(epoch_count, np.array(torch.tensor(loss_values).numpy()), label="Train loss")
#plt.plot(epoch_count, test_loss_values, label="Test loss")
#plt.title("Training and Testing loss curves")
#plt.ylabel("Loss")
#plt.xlabel("Epochs")
#plt.legend()
#plt.show()

def plot_learning_curves(epoch_count=epoch_count,
                         loss_values=loss_values,
                         test_loss_values=test_loss_values):
    plt.plot(epoch_count, np.array(torch.tensor(loss_values).numpy()), label="Train loss")
    plt.plot(epoch_count, np.array(torch.tensor(test_loss_values).numpy()), label="Test loss")
    plt.title("Training and Testing loss curves")
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.legend();

plot_learning_curves()
plt.show()

with torch.inference_mode(): # NO gradient descent with inference mode for making predictions
    y_preds_after_learning = model_0(X_test)

print(f"y_preds => {y_preds_after_learning} ")

plot_predictions(predictions=y_preds_after_learning)
plt.show()

# Saving a model 

# Create model directory
MODEL_PATH = Path("savedmodels")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# Create a model save path
MODEL_NAME = "01_pytorch_workflow_model_0.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
print(f"Saving the model to MODEL_SAVE_PATH = {MODEL_SAVE_PATH}")

torch.save(obj=model_0.state_dict(), f=MODEL_SAVE_PATH)

# Load model
# New instance of the model class and load the state_dict of the saved model 
loaded_model_0 = LinearRegressionModel()
loaded_model_0.load_state_dict(torch.load(f=MODEL_SAVE_PATH))

print(f"Loading state_dict from saved model = {loaded_model_0.state_dict()}")

# Make some predictions on the loaded model
model_0.eval()
with torch.inference_mode():
    orig_model_preds = model_0(X_test)

loaded_model_0.eval()
with torch.inference_mode():
    load_model_preds = loaded_model_0(X_test)

print(f"Compare model_0 preds with loaded_model_0 preds => {orig_model_preds == load_model_preds}")

# Linear regression
# Create a model without settings weight and bias parameter manually
# Use nn.Linear() as Layer
class LinearRegressionModel_V2(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer = nn.Linear(in_features=1,
                                     out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)

# Create a random seed
torch.manual_seed(42)

# Create an instance of the linear regression layer model
# =======================================================
model_1 = LinearRegressionModel_V2()
print(f"Linear layer with LinearRegressionModel_V2 = {model_1}")
print(f"Linear layer with LinearRegressionModel_V2 state_dict() = {model_1.state_dict()}")

# Set the model to use the target device MPS (GPU)
# Set the device      
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

model_1.to(device) # move the model to the GPU
print(f"Model 1 uses GPU (MPS) now: {next(model_1.parameters()).device}")

loss_fn1 = nn.L1Loss()
optimizer1 = torch.optim.SGD(params=model_1.parameters(),
                             lr=0.01)

epochs1 = 300

# Move the data to the GPU
X_train = X_train.to(device)
y_train = y_train.to(device)
X_test = X_test.to(device)
y_test = y_test.to(device)

# loop thru the data
for epoch1 in range(epochs1):

    # Set the model to training mode
    model_1.train() # train mode in PytTorch sets all parameters that require gradients to gradients

    # Forward pass the data thru the model
    y_pred1 = model_1(X_train)

    # Calculate the loss (compare predictions with train target data)
    loss1 = loss_fn1(y_pred1, y_train)
    #print(f"Loss value = {loss}")

    #  Optimizer zero grad (reset optimizer value )
    optimizer1.zero_grad() 

    # Perform backpropagation on the loss with respect to the paramters of the model
    loss1.backward()

    # Step the optimizer (perform gradient descent)
    optimizer1.step()

    # Testing 
    model_1.eval() 
    with torch.inference_mode():  # turns off gradient tracking
        # Forward pass test data
        test_pred1 = model_1(X_test)

        # Calculate the test loss
        test_loss1 = loss_fn1(test_pred1, y_test)
    
    if epoch1 % 10 == 0:
        print(f"Epoch: {epoch1} | Loss: {loss1} | Test loss: {test_loss1}")


print(f"Print LinearRegressionModel_V2 after learning = {model_1.state_dict()}")


model_1.eval()

with torch.inference_mode():
    y_pred = model_1(X_test)

print(f"y_preds => {y_preds} ")

plot_predictions(predictions=y_pred.cpu())
plt.show()


print(f"Ready")




