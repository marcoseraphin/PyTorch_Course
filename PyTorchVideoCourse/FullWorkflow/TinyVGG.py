import torch
import set_seed
import download_data
import data_setup
import torchvision
import engine
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
            nn.Linear(in_features=hidden_units*13*13,
                      out_features=output_shape)
        )
    
    def forward(self, x: torch.Tensor):
        x = self.conv_block_1(x)
        # print(x.shape)
        x = self.conv_block_2(x)
        # print(x.shape)
        x = self.classifier(x)
        # print(x.shape)
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

train_data = datasets.ImageFolder(root=train_dir,
                                  transform=data_transform, # transform for the data
                                  target_transform=None)    # transform for the label/target

test_data = datasets.ImageFolder(root=test_dir,
                                 transform=data_transform, # transform for the data
                                 target_transform=None)    # transform for the label/target


class_names = train_data.classes

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
                output_shape=len(train_data.classes)).to(device)

#print(f"TinyVGG model: {model}")

summary(model, input_size=[1, 3, 64, 64]) # do a test pass through of an example input size 

set_seed.set_seeds(seed=42)

# Set number of epochs
NUM_EPOCHS = 15

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
