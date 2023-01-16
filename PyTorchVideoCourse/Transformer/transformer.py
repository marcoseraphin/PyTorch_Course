import torch
import set_seed
import download_data
import data_setup
import engine
import requests
import create_writer
import utils
from pathlib import Path
import helper_functions
import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms
from torch import nn
from torchinfo import summary

# Set the device      
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# Load data
image_path = download_data.download_data(source="https://github.com/marcoseraphin/PyTorch_Course/raw/main/PyTorchVideoCourse/ZipData/pizza_steak_sushi.zip",
                                         destination="pizza_steak_sushi")

# Setup directory paths to train and test images
train_dir = image_path / "train"
test_dir = image_path / "test"

# Create image size (from Table 3 in the ViT paper) 
IMG_SIZE = 224

# Create transform pipeline manually
manual_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])           
print(f"Manually created transforms: {manual_transforms}")

# Set the batch size
BATCH_SIZE = 32 # this is lower than the ViT paper but it's because we're starting small

# Create data loaders
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=manual_transforms, # use manually created transforms
    batch_size=BATCH_SIZE
)

# Get a batch of images
# image_batch, label_batch = next(iter(train_dataloader))

# # Get a single image from the batch
# image, label = image_batch[0], label_batch[0]

# # Plot image with matplotlib
# plt.imshow(image.permute(1, 2, 0)) # rearrange image dimensions to suit matplotlib [color_channels, height, width] -> [height, width, color_channels]
# plt.title(class_names[label])
# plt.axis(False)
# plt.show()


# Use PyTorch ViT model

# Get pretrained weights for ViT-Base
pretrained_vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT # "DEFAULT" = best available

# Setup a ViT model instance with pretrained weights
pretrained_vit = torchvision.models.vit_b_16(weights=pretrained_vit_weights).to(device)

# Freeze the base parameters
for parameter in pretrained_vit.parameters():
  parameter.requires_grad = False

# Update the classifier head
set_seed.set_seeds()
pretrained_vit.heads = nn.Linear(in_features=768, out_features=len(class_names)).to(device)

summary(model=pretrained_vit,
        input_size=(1, 3, 224, 224), # (batch_size, color_channels, height, width)
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"])

# Get automtic transforms from pretrained ViT weights
vit_transforms = pretrained_vit_weights.transforms()

train_dataloader_pretrained, test_dataloader_pretrained, class_names = data_setup.create_dataloaders(train_dir=train_dir,
                                                                                                     test_dir=test_dir,
                                                                                                     transform=vit_transforms,
                                                                                                     batch_size=32) # could set a higher batch size
# Create optimizer and loss function
optimizer = torch.optim.Adam(params=pretrained_vit.parameters(),
                             lr=1e-3)

loss_fn = torch.nn.CrossEntropyLoss()

summarywriter = create_writer.create_writer("ViT_Extractor",
                                            "pretrained_vit_feature_extractor")

# Train the classifier head of pretrained ViT
pretrained_vit_results = engine.train(model=pretrained_vit,
                                      train_dataloader=train_dataloader_pretrained,
                                      test_dataloader=test_dataloader_pretrained,
                                      optimizer=optimizer,
                                      loss_fn=loss_fn,
                                      epochs=10,
                                      device=device,
                                      img_height=224,
                                      img_width=224,
                                      writer=summarywriter)
                                    
helper_functions.plot_loss_curves(pretrained_vit_results)

utils.save_model(model=pretrained_vit,
                 target_dir="models",
                 model_name="08_pretrained_vit_feature_extractor_pizza_steak_sushi.pth")

# Get the model size in bytes then convert to megabytes 
pretrained_vit_model_size = Path("models/08_pretrained_vit_feature_extractor_pizza_steak_sushi.pth").stat().st_size // (1024*1024)
print(f"Pretrained ViT feature extractor model size: {pretrained_vit_model_size} MB")

# Setup custom image path
custom_image_path = image_path / "TestImage.jpeg"

# Download the image if it doesn't already exist
if not custom_image_path.is_file():
    with open(custom_image_path, "wb") as f:
        # When downloading from GitHub, need to use the "raw" file link
        request = requests.get("https://raw.githubusercontent.com/marcoseraphin/PyTorch_Course/main/PyTorchVideoCourse/ZipData/SampleTestSushi.jpeg")
        print(f"Downloading {custom_image_path}...")
        f.write(request.content)
else:
    print(f"{custom_image_path} already exists, skipping download.")

# Predict on custom image
helper_functions.pred_and_plot_image(model=pretrained_vit,
                                     image_path=custom_image_path,
                                     class_names=class_names)
