import torch
from torch import nn
import requests
import zipfile
from pathlib import Path

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
with open(data_path / "pizza_steak_sushi.zip", "wb") as f:  # wp = write binary
    request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
    print(f"Downloading image data...")
    f.write(request.content)

# Unzip downloaded data
with zipfile.ZipFile(data_path / "pizza_steak_sushi.zip", "r") as zip_ref: # r = read permissions
    print(f"Unzipping image data...")
    zip_ref.extractall(image_path)
