import torch
from torch import nn

# TorchVision
import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor

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

plt.show()

# Turn the data into batches
