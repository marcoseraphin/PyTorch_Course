import torch
from torch import nn

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
# fig = plt.figure(figsize=(9,9))
# rows, cols = 4,4
# for i in range(1, rows*cols+1):
#     random_idx = torch.randint(0, len(train_data), size=[1]).item()
#     img, label = train_data[random_idx]
#     fig.add_subplot(rows, cols, i)
#     plt.imshow(img.squeeze(), cmap="gray")
#     plt.title(class_names[label])
#     plt.axis(False)

# #plt.show()

# # Turn the data into batches
# BATCH_SIZE = 32

# train_dataloader = DataLoader(dataset=train_data,
#                               batch_size=BATCH_SIZE,
#                               shuffle=True)

# test_dataloader = DataLoader(dataset=test_data,
#                              batch_size=BATCH_SIZE,
#                              shuffle=False)

# train_features_batch, train_labels_batch = next(iter(train_dataloader))

# print(f"Length of Train DataLoader: {len(train_dataloader)} batches of {BATCH_SIZE}")
# print(f"Length of Test DataLoader: {len(test_dataloader)} batches of {BATCH_SIZE}")

# random_idx = torch.randint(0, len(train_features_batch), size=[1]).item()
# img, label = train_features_batch[random_idx], train_labels_batch[random_idx]

# plt.imshow(img.squeeze(), cmap="gray")
# plt.title(class_names[label])
# plt.axis(False)
# #plt.show()

# print(f"Image size: {img.shape}")
# print(f"Label {label}, Label Size: {label.shape} ")

# flatten_model = nn.Flatten()
# x = train_features_batch[0]

# # Flatten the sample
# output = flatten_model(x)
# print(f"Shape before flattening: {x.shape}")     # => color channel, height, width
# print(f"Shape after flattening: {output.shape}") # => color channel, height * width

# class FashionMNISTModelV0(nn.Module):
#     def __init__(self,
#                  input_shape: int,
#                  hidden_units: int,
#                  output_shape: int) -> None:
#         super().__init__()
#         self.layer_stack = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(in_features=input_shape,
#                        out_features=hidden_units),
#             nn.Linear(in_features=hidden_units,
#                        out_features=output_shape)
#         )

#     def forward(self, x):
#         return self.layer_stack(x)

# model0 = FashionMNISTModelV0(input_shape=784, # 28x28 pixels of the image after flatten
#                              hidden_units=10,
#                              output_shape=len(class_names) ) 

# dummy_x = torch.rand([1,1,28,28]) # Batch = 1, ColorChannel = 1, Width = 28, Height = 28
# print(f"Output of dummy_x thru the model_0: {model0(dummy_x)}") # one value per label class

# # Loss, optimizer and metrics
# loss_fn = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(params=model0.parameters(),
#                             lr=0.1)

# # Create a function to time our experiments
# def print_train_time(start: float,
#                      end: float,
#                      device: torch.device = None):
#         total_time = end - start
#         print(f"Train time on {device}: {total_time:.3f} seconds")
#         return total_time

# start_time = timer()
# time.sleep(1)
# end_time= timer()
# print_train_time(start=start_time,
#                  end=end_time,
#                  device="cpu")

# # Training loop based on bacthes => update parameter per batch and not per epoch
# train_time_start_on_cpu = timer()
# epochs = 3

# for epoch in tqdm(range(epochs)): # tqdm => progress bar
#     print(f"Epoch: {epoch}")
#     train_loss = 0
#     for batch, (X, y) in enumerate(train_dataloader): # X = image, y = label
#         model0.train()
#         y_pred = model0(X)
#         loss = loss_fn(y_pred, y)
#         train_loss += loss
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         if batch % 400 == 0:
#             print(f"Looked at {batch * len(X)} / {len(train_dataloader.dataset)} samples.")

#     # Divide total train loss by length of train dataloader
#     train_loss /= len(train_dataloader)

#     # Testing
#     test_loss, test_acc = 0,0
#     model0.eval()
#     with torch.inference_mode():
#         for X_test,y_test in test_dataloader:
#             test_pred = model0(X_test)
#             test_loss += loss_fn(test_pred, y_test)
#             test_acc += accuracy_fn(y_true=y_test, y_pred=test_pred.argmax(dim=1))
    
#         test_loss /= len(test_dataloader)
#         test_acc /= len(test_dataloader)

#     print(f"Train Loss: {train_loss:.5f} | Test loss: {test_loss:.5f}, Test acc: {test_acc:.4f}%")

# train_time_end_on_cpu = timer()
# total_train_time_model0 = print_train_time(start=train_time_start_on_cpu,
#                                             end=train_time_end_on_cpu,
#                                             device=str(next(model0.parameters()).device))

# # Make predictions function
# def eval_model(model: torch.nn.Module,
#                data_loader: torch.utils.data.DataLoader,
#                loss_fn: torch.nn.Module,
#                accuracy_fn,
#                device: torch.device = device):
#     loss, acc = 0, 0
#     model.eval()
#     with torch.inference_mode():
#         for X, y in tqdm(data_loader):
#             X, y = X.to(device), y.to(device)
#             y_pred = model(X)
#             loss += loss_fn(y_pred, y)
#             acc += accuracy_fn(y_true=y,
#                                y_pred=y_pred.argmax(dim=1))
        
#         loss /= len(data_loader)
#         acc /= len(data_loader)

#     return {"model_name": model.__class__.__name__,
#            "model_loss": loss.item(),
#            "model_acc": acc}


# model0_results = eval_model(model=model0,
#                              data_loader=test_dataloader,
#                              loss_fn=loss_fn,
#                              accuracy_fn=accuracy_fn,
#                              device="cpu")


# print(f"eval_model call for model_0: {model0_results}") 

# # Using a better model with non-linearity
# class FashionMNISTModelV1 (nn.Module):
#     def __init__(self,
#                  input_shape: int,
#                  hidden_units: int,
#                  output_shape: int) -> None:
#         super().__init__()
#         self.layer_stack = nn.Sequential(
#             nn.Flatten(), # flatten image inputs into a single vector
#             nn.Linear(in_features=input_shape,
#                       out_features=hidden_units),
#             nn.ReLU(),
#             nn.Linear(in_features=hidden_units,
#                       out_features=output_shape),
#             nn.ReLU()
#         )

#     def forward(self, x: torch.Tensor):
#         return self.layer_stack(x)

# model_1 = FashionMNISTModelV1(input_shape=784,
#                               hidden_units=10,
#                               output_shape=len(class_names)).to(device=device)

# # Loss, optimizer and metrics
# loss_fn1 = nn.CrossEntropyLoss()
# optimizer1 = torch.optim.SGD(params=model_1.parameters(),
#                              lr=0.1)

# print(f"model_1 now using device: {next(model_1.parameters()).device}")

# # Functionizing traing and evaluation/testing loops
# def train_step(model: torch.nn.Module,
#                dataloader: torch.utils.data.DataLoader,
#                loss_fn: torch.nn.Module,
#                optimizer: torch.optim.Optimizer,
#                accuracy_fn,
#                device: torch.device = device):

#     train_loss, train_acc = 0, 0
#     model.train()

#     for batch, (X, y) in enumerate(dataloader): # X = image, y = label
#         X, y = X.to(device), y.to(device)
#         y_pred = model(X)
#         loss = loss_fn(y_pred, y)
#         train_loss += loss
#         train_acc += accuracy_fn(y_true=y,
#                                  y_pred=y_pred.argmax(dim=1)) # from logits to prediction lables
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#     # Divide total train loss and accuracy by length of train dataloader
#     train_loss /= len(dataloader)
#     train_acc /= len(dataloader)

#     print(f"Train Loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")

# def test_step(model: torch.nn.Module,
#               dataloader: torch.utils.data.DataLoader,
#               loss_fn: torch.nn.Module,
#               accuracy_fn,
#               device: torch.device = device):

#     test_loss, test_acc = 0,0
#     model.eval()
#     with torch.inference_mode():
#         for X, y in dataloader:
#             X, y = X.to(device), y.to(device)
#             test_pred = model(X)
#             test_loss += loss_fn(test_pred, y)
#             test_acc += accuracy_fn(y_true=y, y_pred=test_pred.argmax(dim=1))
    
#         test_loss /= len(dataloader)
#         test_acc /= len(dataloader)

#         print(f"Test Loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%")

# train_time_start_on_gpu = timer()
# epochs = 3

# for epoch in tqdm(range(epochs)): # tqdm => progress bar
#     print(f"EPoch {epoch} on GPU {device} ")
#     train_step(model=model_1,
#                dataloader=train_dataloader,
#                loss_fn=loss_fn1,
#                optimizer=optimizer1,
#                accuracy_fn=accuracy_fn,
#                device=device)

#     test_step(model=model_1,
#               dataloader=test_dataloader,
#               loss_fn=loss_fn1,
#               accuracy_fn=accuracy_fn,
#               device=device)

# train_time_end_on_gpu = timer()
# total_time_train_model_1_gpu = print_train_time(start=train_time_start_on_gpu, 
#                                                 end=train_time_end_on_gpu, 
#                                                 device=device)

# model_1_results = eval_model(model=model_1,
#                              data_loader=test_dataloader,
#                              loss_fn=loss_fn1,
#                              accuracy_fn=accuracy_fn,
#                              device=device)

# print(f"eval_model call for model_1: {model_1_results}") 


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
            nn.Linear(in_features=hidden_units*1*1,
                      out_features=output_shape)
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        print(x.shape)
        x = self.conv_block_2(x)
        print(x.shape)
        x = self.classifier(x)
        return x


model_2 = FashionMNISTModuleV2CNN(input_shape=1,   # number of color channel (here 1 = black/white)
                                   hidden_units=10,
                                   output_shape=len(class_names)).to(device)

print(f"CNN model_2: {model_2}") 

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

# simple demo of maxpool2d
random_tensor = torch.randn(size=[1,1,2,2])
print(f"random_tensor: {random_tensor}") 
print(f"random_tensor Shape: {random_tensor.shape}") 

max_pool_layer_demo = nn.MaxPool2d(kernel_size=2)
max_pool_tensor = max_pool_layer(random_tensor)
print(f"max_pool_tensor: {max_pool_tensor}") 
print(f"max_pool_tensor Shape: {max_pool_tensor.shape}") 
