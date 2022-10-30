import torch
from torch import Tensor, nn # nn contains all of PyTorch's building blocks for neural networks
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1)

# Splitting data into traing and test sets
train_split = int(0.8 * len(X))

X_train = X[:train_split]
y_train = X[:train_split] # first data (80%) for training

X_test = X[train_split:]
y_test = X[train_split:]  # rest data (20%) for testing

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
    y_pred1 = model_1(X_test)

#print(f"y_preds => {y_pred1}")

plot_predictions(predictions=y_pred1.cpu())
plt.show()
