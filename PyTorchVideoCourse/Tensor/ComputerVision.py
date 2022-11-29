import torch
from torch import nn
import pandas as pd
import random

# TorchVision
import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from helper_functions import accuracy_fn
from timeit import default_timer as timer
import time
from tqdm.auto import tqdm
from pathlib import Path
#import torchmetrics, mlxtend

# Import matpltlib
import matplotlib.pyplot as plt

print(f"PyTorch version: {torch.__version__}")
print(f"Torchvision version: {torchvision.__version__}")

# Use GPU 
# Set the model to use the target device MPS (GPU)
# Set the device      
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# Getting a FashionMNIST dataset

# Setup training data
train_data = datasets.FashionMNIST(root="data", # where to store data
                                   train=True, #  we want the train dataset
                                   download=True, # we want to download
                                   transform=torchvision.transforms.ToTensor(), # transform the data
                                   target_transform=None) # transform the target labels

test_data = datasets.FashionMNIST(root="data", # where to store data
                                  train=False, #  we want the train dataset
                                  download=True, # we want to download
                                  transform=torchvision.transforms.ToTensor(), # transform the data
                                  target_transform=None) # transform the target labels

print(f"train data: {train_data}")
print(f"test data: {test_data}")

print(f"Length of train data: {len(train_data)}")
print(f"Length of test data: {len(test_data)}")

class_names = train_data.classes
print(f"Classes of train data: {class_names}")

class_to_idx = train_data.class_to_idx
print(f"Classes indexes of train data: {class_to_idx}")

sample_image, sample_label = train_data[0]
print(f"Image Shape of train data: {sample_image.shape}")
print(f"Label index of train data: {sample_label}")
print(f"Label name of train data: {class_names[sample_label]}")

plt.imshow(sample_image.squeeze(), cmap="gray")
plt.title(class_names[sample_label])
plt.axis(False)

# plot random image
torch.manual_seed(42)
fig = plt.figure(figsize=(9,9))
rows, cols = 4,4
for i in range(1, rows*cols+1):
    random_idx = torch.randint(0, len(train_data), size=[1]).item()
    img, label = train_data[random_idx]
    fig.add_subplot(rows, cols, i)
    plt.imshow(img.squeeze(), cmap="gray")
    plt.title(class_names[label])
    plt.axis(False)

#plt.show()

# Turn the data into batches
BATCH_SIZE = 32

train_dataloader = DataLoader(dataset=train_data,
                              batch_size=BATCH_SIZE,
                              shuffle=True)

test_dataloader = DataLoader(dataset=test_data,
                             batch_size=BATCH_SIZE,
                             shuffle=False)

train_features_batch, train_labels_batch = next(iter(train_dataloader))

print(f"Length of Train DataLoader: {len(train_dataloader)} batches of {BATCH_SIZE}")
print(f"Length of Test DataLoader: {len(test_dataloader)} batches of {BATCH_SIZE}")

random_idx = torch.randint(0, len(train_features_batch), size=[1]).item()
img, label = train_features_batch[random_idx], train_labels_batch[random_idx]

plt.imshow(img.squeeze(), cmap="gray")
plt.title(class_names[label])
plt.axis(False)
#plt.show()

print(f"Image size: {img.shape}")
print(f"Label {label}, Label Size: {label.shape} ")

flatten_model = nn.Flatten()
x = train_features_batch[0]

# Flatten the sample
output = flatten_model(x)
print(f"Shape before flattening: {x.shape}")     # => color channel, height, width
print(f"Shape after flattening: {output.shape}") # => color channel, height * width

class FashionMNISTModelV0(nn.Module):
    def __init__(self,
                 input_shape: int,
                 hidden_units: int,
                 output_shape: int) -> None:
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape,
                       out_features=hidden_units),
            nn.Linear(in_features=hidden_units,
                       out_features=output_shape)
        )

    def forward(self, x):
        return self.layer_stack(x)

model_0 = FashionMNISTModelV0(input_shape=784, # 28x28 pixels of the image after flatten
                             hidden_units=10,
                             output_shape=len(class_names) ) 

dummy_x = torch.rand([1,1,28,28]) # Batch = 1, ColorChannel = 1, Width = 28, Height = 28
print(f"Output of dummy_x thru the model_0: {model_0(dummy_x)}") # one value per label class

# Loss, optimizer and metrics
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_0.parameters(),
                            lr=0.1)

# Create a function to time our experiments
def print_train_time(start: float,
                     end: float,
                     device: torch.device = None):
        total_time = end - start
        print(f"Train time on {device}: {total_time:.3f} seconds")
        return total_time

start_time = timer()
time.sleep(1)
end_time= timer()
print_train_time(start=start_time,
                 end=end_time,
                 device="cpu")

# Training loop based on bacthes => update parameter per batch and not per epoch
train_time_start_on_cpu = timer()
epochs = 3

for epoch in tqdm(range(epochs)): # tqdm => progress bar
    print(f"Epoch: {epoch}")
    train_loss = 0
    for batch, (X, y) in enumerate(train_dataloader): # X = image, y = label
        model_0.train()
        y_pred = model_0(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % 400 == 0:
            print(f"Looked at {batch * len(X)} / {len(train_dataloader.dataset)} samples.")

    # Divide total train loss by length of train dataloader
    train_loss /= len(train_dataloader)

    # Testing
    test_loss, test_acc = 0,0
    model_0.eval()
    with torch.inference_mode():
        for X_test,y_test in test_dataloader:
            test_pred = model_0(X_test)
            test_loss += loss_fn(test_pred, y_test)
            test_acc += accuracy_fn(y_true=y_test, y_pred=test_pred.argmax(dim=1))
    
        test_loss /= len(test_dataloader)
        test_acc /= len(test_dataloader)

    print(f"Train Loss: {train_loss:.5f} | Test loss: {test_loss:.5f}, Test acc: {test_acc:.4f}%")

train_time_end_on_cpu = timer()
total_train_time_model0 = print_train_time(start=train_time_start_on_cpu,
                                            end=train_time_end_on_cpu,
                                            device=str(next(model_0.parameters()).device))

# Make predictions function
def eval_model(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               accuracy_fn,
               device: torch.device = device):
    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in tqdm(data_loader):
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(y_true=y,
                               y_pred=y_pred.argmax(dim=1))
        
        loss /= len(data_loader)
        acc /= len(data_loader)

    return {"model_name": model.__class__.__name__,
           "model_loss": loss.item(),
           "model_acc": acc}


model_0_results = eval_model(model=model_0,
                             data_loader=test_dataloader,
                             loss_fn=loss_fn,
                             accuracy_fn=accuracy_fn,
                             device="cpu")


print(f"eval_model call for model_0: {model_0_results}") 

# Using a better model with non-linearity
class FashionMNISTModelV1 (nn.Module):
    def __init__(self,
                 input_shape: int,
                 hidden_units: int,
                 output_shape: int) -> None:
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(), # flatten image inputs into a single vector
            nn.Linear(in_features=input_shape,
                      out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units,
                      out_features=output_shape),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor):
        return self.layer_stack(x)

model_1 = FashionMNISTModelV1(input_shape=784,
                              hidden_units=10,
                              output_shape=len(class_names)).to(device=device)

# Loss, optimizer and metrics
loss_fn1 = nn.CrossEntropyLoss()
optimizer1 = torch.optim.SGD(params=model_1.parameters(),
                             lr=0.1)

print(f"model_1 now using device: {next(model_1.parameters()).device}")

# Functionizing traing and evaluation/testing loops
def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device: torch.device = device):

    train_loss, train_acc = 0, 0
    model.train()

    for batch, (X, y) in enumerate(dataloader): # X = image, y = label
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss
        train_acc += accuracy_fn(y_true=y,
                                 y_pred=y_pred.argmax(dim=1)) # from logits to prediction lables
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Divide total train loss and accuracy by length of train dataloader
    train_loss /= len(dataloader)
    train_acc /= len(dataloader)

    print(f"Train Loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")

def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              accuracy_fn,
              device: torch.device = device):

    test_loss, test_acc = 0,0
    model.eval()
    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            test_pred = model(X)
            test_loss += loss_fn(test_pred, y)
            test_acc += accuracy_fn(y_true=y, y_pred=test_pred.argmax(dim=1))
    
        test_loss /= len(dataloader)
        test_acc /= len(dataloader)

        print(f"Test Loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%")

train_time_start_on_gpu = timer()
epochs = 3

for epoch in tqdm(range(epochs)): # tqdm => progress bar
    print(f"EPoch {epoch} on GPU {device} ")
    train_step(model=model_1,
               dataloader=train_dataloader,
               loss_fn=loss_fn1,
               optimizer=optimizer1,
               accuracy_fn=accuracy_fn,
               device=device)

    test_step(model=model_1,
              dataloader=test_dataloader,
              loss_fn=loss_fn1,
              accuracy_fn=accuracy_fn,
              device=device)

train_time_end_on_gpu = timer()
total_time_train_model_1_gpu = print_train_time(start=train_time_start_on_gpu, 
                                                end=train_time_end_on_gpu, 
                                                device=device)

model_1_results = eval_model(model=model_1,
                             data_loader=test_dataloader,
                             loss_fn=loss_fn1,
                             accuracy_fn=accuracy_fn,
                             device=device)

print(f"eval_model call for model_1: {model_1_results}") 


# CNN Network (Convolutional neural network)
class FashionMNISTModuleV2CNN(nn.Module):
    """Model architecture replicates TinyVGG
    network from CNN explainer website
    """
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
        super().__init__()

        # Conv layer 1
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1), # values we can set ourselves => hyperparameters
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        # Conv layer 2
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
            out_channels=hidden_units,
            kernel_size=3,
            stride=1,
            padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        # Classifier output layer to turn features into labels
        self.classifier = nn.Sequential(
            nn.Flatten(), # flatten image inputs into a single vector
            nn.Linear(in_features=hidden_units*7*7,  # => output shape of conv_block_2 = [7,7]
                      out_features=output_shape)
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        #print(f"Output conv_block_1 shape: {x.shape}")
        x = self.conv_block_2(x)
        #print(f"Output conv_block_2 shape: {x.shape}")
        x = self.classifier(x)
        #print(f"Output classifier shape: {x.shape}")
        return x


model_2 = FashionMNISTModuleV2CNN(input_shape=1,   # number of color channel (here 1 = black/white)
                                  hidden_units=10,
                                  output_shape=len(class_names)).to(device)

#print(f"CNN model_2: {model_2}") 

images = torch.randn(size=[32,3,64,64])
test_image = images[0]

print(f"Image batch shape: {images.shape}") 
print(f"Single Image shape: {test_image.shape}") 
print(f"Test Image: {test_image}") 

# Create a single test con2d layer for understanding kernel_size (filter window), stride and padding
# => https://poloclub.github.io/cnn-explainer/
conv_layer = nn.Conv2d(in_channels=3,   # number of color channel (here 3)
                        out_channels=10,
                        kernel_size=(3,3),
                        stride=1,
                        padding=0)

# pass the Test data (test image) thru the conv layer
conv_output = conv_layer(test_image)
print(f"conv_output: {conv_output}") 

print(f"Single Image shape: {test_image.shape}") 
print(f"conv_output Shape: {conv_output.shape}") 

# MaxPool2d
max_pool_layer = nn.MaxPool2d(kernel_size=2)

test_image_conv = conv_layer(test_image)
print(f"Image after conv layer shape: {test_image_conv.shape}") 
test_image_trough_conv_and_max_pool = max_pool_layer(test_image_conv)
print(f"Image after conv layer and maxpool layer shape: {test_image_trough_conv_and_max_pool.shape}") 

# Simple demo of maxpool2d
random_tensor = torch.randn(size=[1,1,2,2])
print(f"random_tensor: {random_tensor}") 
print(f"random_tensor Shape: {random_tensor.shape}") 

max_pool_layer_demo = nn.MaxPool2d(kernel_size=2)
max_pool_tensor = max_pool_layer(random_tensor)
print(f"max_pool_tensor: {max_pool_tensor}") 
print(f"max_pool_tensor Shape: {max_pool_tensor.shape}") 

print(f"sample_image Shape: {img.shape}") 
#plt.imshow(img.squeeze(), cmap="gray")
#plt.show()
output_model_2 = model_2(img.unsqueeze(0).to(device))
print(f"output_model_2: {output_model_2}") 

# Train the CNN
loss_fn_cnn = nn.CrossEntropyLoss()
optimzer_cnn = torch.optim.SGD(params=model_2.parameters(), lr=0.01)

train_time_start_on_gpu = timer()

epochs = 3
for epoch in tqdm(range(epochs)): # tqdm => progress bar
    print(f"EPoch {epoch} on GPU {device} ")
    train_step(model=model_2,
               dataloader=train_dataloader,
               loss_fn=loss_fn_cnn,
               optimizer=optimzer_cnn,
               accuracy_fn=accuracy_fn,
               device=device)

    test_step(model=model_2,
              dataloader=test_dataloader,
              loss_fn=loss_fn_cnn,
              accuracy_fn=accuracy_fn,
              device=device)

train_time_end_on_gpu = timer()
total_time_train_model_1_gpu = print_train_time(start=train_time_start_on_gpu, 
                                                end=train_time_end_on_gpu, 
                                                device=device)

model_2_results = eval_model(model=model_2,
                             data_loader=test_dataloader,
                             loss_fn=loss_fn_cnn,
                             accuracy_fn=accuracy_fn,
                             device=device)

print(f"eval_model call for CNN model_2: {model_1_results}") 

# compare results
compare_results = pd.DataFrame([model_0_results, model_1_results, model_2_results])
print(f"compare_results: {compare_results}") 

def make_predictions(model: torch.nn.Module,
                     data: list,
                     device: torch.device = device):
    pred_probs = []
    model.to(device)
    model.eval()
    with torch.inference_mode():
        for sample in data:
            sample = torch.unsqueeze(sample, dim=0).to(device=device)
            pred_logits = model(sample)
            pred_prob = torch.softmax(pred_logits.squeeze(), dim=0)
            pred_probs.append(pred_prob.cpu())

    return torch.stack(pred_probs)

test_samples_list = []
test_labels_list = []
for sample, label in random.sample(list(test_data), k=9):
    test_samples_list.append(sample)
    test_labels_list.append(label)

# Make predictions
pred_probs = make_predictions(model=model_2,
                              data=test_samples_list)

# Convert prediction probalilities into labels
pred_classes = pred_probs.argmax(dim=1)
print(f"pred_classes: {pred_classes}") 
print(f"test_labels_list: {test_labels_list}") 

plt.figure(figsize=(9,9))
nrows = 3
ncols = 3
for i, sample in enumerate(test_samples_list):
    plt.subplot(nrows, ncols, i+1)
    plt.axis(False)
    plt.imshow(sample.squeeze(), cmap="gray")
    pred_label = class_names[pred_classes[i]]
    truth_label =  class_names[test_labels_list[i]]
    tite_text = f"Pred: {pred_label} | Truth: {truth_label}"
    if pred_label == truth_label:
        plt.title(tite_text, fontsize=10, c="g")
    else:
        plt.title(tite_text, fontsize=10, c="r")
plt.show()


# Making a confusion matrix
# https://torchmetrics.readthedocs.io/en/stable/classification/confusion_matrix.html

# Make predictions with trained model
y_preds = []
model_2.eval()
with torch.inference_mode():
    for X, y in tqdm(test_dataloader, desc="Making predictions...."):
        X, y = X.to(device), y.to(device)
        y_logits = model_2(X)
        y_pred = torch.softmax(y_logits.squeeze(), dim=0).argmax(dim=1)
        y_preds.append(y_pred.cpu())

#print(f"y_preds: {y_preds}") 
y_pred_tensor = torch.cat(y_preds)
print(f"First 10 y_pred_tensor values: {y_pred_tensor[:10]}") 

#print(f"mlextend version: {mlxtend.__version__}")

# try:
#     import torchmetrics, mlxtend
#     print(f"mlextend version: {mlxtend.__version__}")
#     assert int(mlxtend.__version__.split(".")[1] >= 19, "mlxtend should be version 0.19.0 or higher")
# except:
#     !pip install torchmetrics -U mlxtend
#     import torchmetrics, mlxtend
#     print(f"mlextend version: {mlxtend.__version__}")

# Save the model_2
MODEL_PATH = Path("savedmodels")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "CNN_Model_2.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

print(f"Saving a model to : {MODEL_SAVE_PATH}") 
torch.save(obj=model_2.state_dict(), f=MODEL_SAVE_PATH)

loaded_model_2 = FashionMNISTModuleV2CNN(input_shape=1,   # number of color channel (here 1 = black/white)
                                  hidden_units=10,
                                  output_shape=len(class_names))


print(f"Loading model 2 from : {MODEL_SAVE_PATH}") 
loaded_model_2.load_state_dict(torch.load(f=MODEL_SAVE_PATH))
loaded_model_2.to(device)

torch.manual_seed(42)
loaded_model_2_results = eval_model(model=loaded_model_2,
                                     data_loader=test_dataloader,
                                     loss_fn=loss_fn_cnn,
                                     accuracy_fn=accuracy_fn)

compare_results = pd.DataFrame([model_2_results, loaded_model_2_results])
print(f"compare_results of model_2 and loaded_model_2: {compare_results}") 
