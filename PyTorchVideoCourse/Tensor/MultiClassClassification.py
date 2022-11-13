import torch
from torch import Tensor, nn # nn contains all of PyTorch's building blocks for neural networks
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from helper_functions import plot_predictions, plot_decision_boundary
from torchmetrics import Accuracy

NUM_CLASSES = 4
NUM_FEATURES = 2
RANDOM_SEED = 42

# Calculate accuracy 
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

# Create multi-class data
X_blob, y_blob = make_blobs(n_samples=1000, 
                            n_features=NUM_FEATURES, 
                            centers=NUM_CLASSES,
                            cluster_std=1.5,
                            random_state=RANDOM_SEED)

# Turn data into tensors
X_blob = torch.from_numpy(X_blob).type(torch.float)
y_blob = torch.from_numpy(y_blob).type(torch.LongTensor)

# Split into train and test data
X_blob_train, X_blob_test, y_blob_train, y_blob_test = train_test_split(X_blob,
                                                                        y_blob,
                                                                        test_size=0.2,
                                                                        random_state=RANDOM_SEED)

# Plot data
plt.figure(figsize=(10,7))
plt.scatter(X_blob[:, 0], X_blob[:, 1], c=y_blob, cmap=plt.cm.RdYlBu);
#plt.show()

# Building a multi-class classification model

# Set the model to use the target device MPS (GPU)
# Set the device      
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

class BlobModel(nn.Module):
    def __init__(self, input_features, out_features, hidden_units=8) -> None:
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_units),
            #nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            #nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=out_features)
        )

    def forward(self, x):
        return self.linear_layer_stack(x)


model_blob = BlobModel(input_features=2, 
                       out_features=4,
                       hidden_units=8).to(device=device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_blob.parameters(),
                             lr=0.1)

model_blob.eval()
with torch.inference_mode():
    y_logits = model_blob(X_blob_test.to(device))

print(f"y_logits => {y_logits[:10]}") # => these are logits (raw output of the model)
print(f"y_blob_test => {y_blob_test[:10]}")

# We must convert our logits to predictions probabilities and than to prediction labels
# Use and activation function => SoftMax for multi-class classification
y_pred_probs = torch.softmax(y_logits, dim=1)

print(f"y_pred_probs => {y_pred_probs[:10]}") # => these are probabilities
print(f"y_blob_test => {y_blob_test[:10]}")

# Convert model prediction probabilities to prediction labels
y_preds = torch.argmax(y_pred_probs, dim=1)
print(f"y_preds => {y_preds[:10]}")

# Train the model
torch.manual_seed(42)

epochs = 100

# Put data to the MPS
X_blob_train, y_blob_train = X_blob_train.to(device), y_blob_train.to(device)
X_blob_test, y_blob_test = X_blob_test.to(device), y_blob_test.to(device)

for epoch in range(epochs):

    model_blob.train()

    # Forward pass
    y_logits = model_blob(X_blob_train).squeeze()
    y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)

    # Calculate loss and accuarcy (loss_fn is BCEWithLogitsLoss, so it requires logits as input)
    loss = loss_fn(y_logits, 
                   y_blob_train)

    acc = accuracy_fn(y_true=y_blob_train,
                      y_pred=y_pred)

    # Optimizer zero grad
    optimizer.zero_grad()

    # Loss backward
    loss.backward()

    # Optimizer step (gradient descent)
    optimizer.step()

    # Testing
    model_blob.eval()
    with torch.inference_mode():
        test_logits = model_blob(X_blob_test).squeeze()
        test_pred = torch.softmax(test_logits, dim=1).argmax(dim=1)

        test_loss = loss_fn(test_logits, y_blob_test)
        test_acc = accuracy_fn(y_true=y_blob_test, y_pred=test_pred)

    if epoch % 10 == 0:
        print(f"EPoch {epoch} | Loss: {loss:.5f}, Acc: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")

# Making and avaluating predictions
model_blob.eval()
with torch.inference_mode():
    y_logits = model_blob(X_blob_test.to(device))

# We must convert our logits to predictions probabilities and than to prediction labels
# Use and activation function => SoftMax for multi-class classification
y_pred_probs = torch.softmax(y_logits, dim=1)

# Convert model prediction probabilities to prediction labels
y_preds = torch.argmax(y_pred_probs, dim=1)
print(f"y_preds     => {y_preds[:10]}")
print(f"y_blob_test => {y_blob_test[:10]}")

# Plot decision boundary of the model
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.title("Train")
plot_decision_boundary(model_blob, X_blob_train, y_blob_train)
plt.subplot(1,2,2)
plt.title("Test")
plot_decision_boundary(model_blob, X_blob_test, y_blob_test)
plt.show()

# Use torchmetrics (extra package => conda install -c conda-forge torchmetrics)

# Setup metric
torchmetrics_accuary = Accuracy().to(device=device)

# Calculate accuracy
print(f"The model has {torchmetrics_accuary(y_preds, y_blob_test)}% (torchmetrics_accuary) Accuracy")
