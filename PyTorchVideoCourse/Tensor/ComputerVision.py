import torch
from torch import nn

# TorchVision
import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

# Import matpltlib
import matplotlib.pyplot as plt

print(f"PyTorch version: {torch.__version__}")
print(f"Torchvision version: {torchvision.__version__}")

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

model0 = FashionMNISTModelV0(input_shape=784, # 28x28 pixels of the image after flatten
                             hidden_units=10,
                             output_shape=len(class_names) ) 

dummy_x = torch.rand([1,1,28,28]) # Batch = 1, ColorChannel = 1, Width = 28, Height = 28
print(f"Output of dummy_x thru the model_0: {model0(dummy_x)}") # one value per label class



