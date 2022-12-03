from matplotlib import pyplot as plt
import torch
from torch import nn
import requests
import zipfile
from pathlib import Path
import os
import random
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Use GPU 
# Set the model to use the target device MPS (GPU)
# Set the device      
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# Get Data from Food101 Dataset (Subset)
data_path = Path("data_food101")
image_path = data_path / "pizza_steak_sushi"

if image_path.is_dir():
    print(f"{image_path} directory already exists")
else:
    print(f"{image_path} does NOT exist, creating one...")
    image_path.mkdir(parents=True, exist_ok=True)

# Donwload Food101.zip image data
# with open(data_path / "pizza_steak_sushi.zip", "wb") as f:  # wp = write binary
#     request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
#     print(f"Downloading image data...")
#     f.write(request.content)

# # Unzip downloaded data
# with zipfile.ZipFile(data_path / "pizza_steak_sushi.zip", "r") as zip_ref: # r = read permissions
#     print(f"Unzipping image data...")
#     zip_ref.extractall(image_path)

def walk_trough_dir(dir_path):
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in {dirpath}")

walk_trough_dir(image_path)

# Setup train and testing path
train_dir = image_path / "train"
test_dir = image_path / "test"

# Visualize an image
#random.seed(42)

# Get all image paths
image_path_list = list(image_path.glob("*/*/*.jpg")) # stick together /test/pizza/123.jpg
print(f"Image Data Paths: {image_path_list}")

# Pick and random image
random_image_path = random.choice(image_path_list)
print(f"Random Image Paths: {random_image_path}")

# Get the image class from the path name
image_class = random_image_path.parent.stem
print(f"Random Image Path Class/Folder name: {image_class}")

# Open image 
random_image = Image.open(random_image_path)

# Print metadata
print(f"Image height: {random_image.height}")
print(f"Image width: {random_image.width}")

image_as_array = np.asarray(random_image)
plt.imshow(image_as_array)
plt.title(f"Image class: {image_class} | Image shape: {image_as_array.shape} => [height, width, color]")
plt.axis(False)
plt.show()

# Transforming data (image to Tensor)
data_transform = transforms.Compose([
   transforms.Resize(size=(64,64)),
   transforms.RandomHorizontalFlip(p=0.5),
   transforms.ToTensor()
])


print(f"data_transform(random_image): {data_transform(random_image)}")
print(f"Shape of data_transform(random_image): {data_transform(random_image).shape}")

