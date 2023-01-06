import torch
import set_seed
import download_data
import data_setup
import torchvision
import engine
import requests
from pathlib import Path
import helper_functions
import create_writer
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch import nn
from torchinfo import summary

class TinyVGG(nn.Module):
    """
    Model architecture copying TinyVGG from: 
    https://poloclub.github.io/cnn-explainer/
    """
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, 
                      out_channels=hidden_units, 
                      kernel_size=3, # how big is the square that's going over the image?
                      stride=1, # default
                      padding=0), # options = "valid" (no padding) or "same" (output has same shape as input) or int for specific number 
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, 
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2) # default stride value is same as kernel_size
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # Where did this in_features shape come from? 
            # It's because each layer of our network compresses and changes the shape of our inputs data.
            # nn.Linear(in_features=hidden_units*53*53,
            #           out_features=output_shape)
            nn.Linear(in_features=hidden_units*13*13,
                      out_features=output_shape)
        )
    
    def forward(self, x: torch.Tensor):
        x = self.conv_block_1(x)
        # print(x.shape)
        x = self.conv_block_2(x)
        #print(x.shape)
        x = self.classifier(x)
        #print(x.shape)
        return x
        # return self.classifier(self.conv_block_2(self.conv_block_1(x))) # <- leverage the benefits of operator fusion

# Set the device      
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# Load data
image_path = download_data.download_data(source="https://github.com/marcoseraphin/PyTorch_Course/raw/main/PyTorchVideoCourse/ZipData/pizza_steak_sushi.zip",
                                         destination="pizza_steak_sushi")

# Setup directories
train_dir = image_path / "train"
test_dir = image_path / "test"

# Transforming data (image to Tensor)
data_transform = transforms.Compose([
   transforms.Resize(size=(64,64)),
   transforms.RandomHorizontalFlip(p=0.5),
   transforms.ToTensor()
])

# Create data loaders
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=data_transform, # use manual created transforms
    batch_size=32
)

print(f"classes: {class_names}") 

model = TinyVGG(input_shape=3, # number of color channels (3 for RGB) 
                hidden_units=10, 
                output_shape=len(class_names)).to(device)

#print(f"TinyVGG model: {model}")

summary(model, input_size=[1, 3, 64, 64]) # do a test pass through of an example input size 

set_seed.set_seeds(seed=42)

# Set number of epochs
NUM_EPOCHS = 5

# Setup loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

# Start the timer
from timeit import default_timer as timer 
start_time = timer()


summarywriter = create_writer.create_writer("Food101",
                                            "TinyVGG Model")

# Train model
results = engine.train(model=model, 
                       train_dataloader=train_dataloader,
                       test_dataloader=test_dataloader,
                       optimizer=optimizer,
                       loss_fn=loss_fn, 
                       epochs=NUM_EPOCHS,
                       device=device,
                       writer=summarywriter,
                       img_width=64,
                       img_height=64)

# End the timer and print out how long it took
end_time = timer()
print(f"Total training time: {end_time-start_time:.3f} seconds")

helper_functions.plot_loss_curves(results=results)
plt.show()

# Download custom image
data_path = Path("data")

# Setup custom image path
custom_image_path = data_path / "SampleTestPizza.jpeg"

# Download the image if it doesn't already exist
if not custom_image_path.is_file():
    with open(custom_image_path, "wb") as f:
        # When downloading from GitHub, need to use the "raw" file link
        request = requests.get("https://raw.githubusercontent.com/marcoseraphin/PyTorch_Course/main/PyTorchVideoCourse/ZipData/SampleTestPizza.jpeg")
        print(f"Downloading {custom_image_path}...")
        f.write(request.content)
else:
    print(f"{custom_image_path} already exists, skipping download.")


# Load in custom image and convert the tensor values to float32
custom_image = torchvision.io.read_image(str(custom_image_path)).type(torch.float32)

# Divide the image pixel values by 255 to get them between [0, 1]
custom_image = custom_image / 255. 

# Print out image data
# print(f"Custom image tensor:\n{custom_image}\n")
# print(f"Custom image shape: {custom_image.shape}\n")
# print(f"Custom image dtype: {custom_image.dtype}")

# Plot custom image
# plt.imshow(custom_image.permute(1, 2, 0)) # need to permute image dimensions from CHW -> HWC otherwise matplotlib will error
# plt.title(f"Image shape: {custom_image.shape}")
# plt.axis(False)

# plt.show()

# Create transform pipleine to resize image
custom_image_transform = transforms.Compose([
    transforms.Resize((64, 64)),
])

# Transform target image
custom_image_transformed = custom_image_transform(custom_image)

# Print out original shape and new shape
print(f"Original shape: {custom_image.shape}")
print(f"New shape: {custom_image_transformed.shape}")

# Try to make a prediction on image in uint8 format (this will error)
model.eval()
with torch.inference_mode():
    # Add an extra dimension to image
    custom_image_transformed_with_batch_size = custom_image_transformed.unsqueeze(dim=0)
    
    # Print out different shapes
    print(f"Custom image transformed shape: {custom_image_transformed.shape}")
    print(f"Unsqueezed custom image shape: {custom_image_transformed_with_batch_size.shape}")
    
    # Make a prediction on image with an extra dimension
    custom_image_pred = model(custom_image_transformed.unsqueeze(dim=0).to(device))

# Print out prediction logits
print(f"Prediction logits: {custom_image_pred}")

# Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
custom_image_pred_probs = torch.softmax(custom_image_pred, dim=1)
print(f"Prediction probabilities: {custom_image_pred_probs}")

# Convert prediction probabilities -> prediction labels
custom_image_pred_label = torch.argmax(custom_image_pred_probs, dim=1)
print(f"Prediction label: {custom_image_pred_label}")

# Find the predicted label
custom_image_pred_class = class_names[custom_image_pred_label.cpu()] # put pred label to CPU, otherwise will error
print(f"Prediction class: {custom_image_pred_class}")

# Pred on our custom image
helper_functions.pred_and_plot_image(model=model,
                                     image_path=custom_image_path,
                                     class_names=class_names,
                                     transform=custom_image_transform,
                                     device=device)
plt.show()
