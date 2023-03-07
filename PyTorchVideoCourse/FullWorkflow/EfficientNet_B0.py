import torch
import set_seed
import download_data
import data_setup
import torchvision
import engine
import utils
import requests
import random
import helper_functions
import create_writer
import coremltools as ct
from pathlib import Path
#import onnx_coreml
import matplotlib.pyplot as plt
from torchvision import transforms
from torch import nn
from PIL import Image
from torchinfo import summary

# Set the device      
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# Load data
image_path = download_data.download_data(source="https://github.com/marcoseraphin/PyTorch_Course/raw/main/PyTorchVideoCourse/ZipData/pizza_steak_sushi.zip",
                                         destination="pizza_steak_sushi")

# Setup directories
train_dir = image_path / "train"
test_dir = image_path / "test"

# Setup ImageNet normalization levels (turns all images into similar distribution as ImageNet)
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

# Setup pretrained weights (plenty of these available in torchvision.models)
weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT

# Get transforms from weights (these are the transforms that were used to obtain the weights)
automatic_transforms = weights.transforms() 
print(f"Automatically created transforms: {automatic_transforms}")

# Create data loaders
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=automatic_transforms, # use automatic created transforms
    batch_size=32
)

print(f"classes: {class_names}") 

# Note: This is how a pretrained model would be created in torchvision > 0.13, it will be deprecated in future versions.
# model = torchvision.models.efficientnet_b0(pretrained=True).to(device) # OLD 

# Download the pretrained weights for EfficientNet_B2
weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT # NEW in torchvision 0.13, "DEFAULT" means "best weights available"

# Setup the model with the pretrained weights and send it to the target device
model = torchvision.models.efficientnet_b2(weights=weights).to(device)

# View the output of the model
# model

# Freeze all base layers by setting requires_grad attribute to False
for param in model.features.parameters():
    param.requires_grad = False
    
# Since we're creating a new layer with random weights (torch.nn.Linear), 
# let's set the seeds
set_seed.set_seeds(seed=42)

# Get a summary of the model (uncomment for full output)
summary(model, 
        input_size=(32, 3, 224, 224), # make sure this is "input_size", not "input_shape" (batch_size, color_channels, height, width)
        verbose=1,
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"]
)

# Update the classifier head to suit our problem
model.classifier = torch.nn.Sequential(nn.Dropout(p=0.2, inplace=True),
                                       nn.Linear(in_features=1408,  #  for efficientnet_b2 
                                       #nn.Linear(in_features=1280, #  for efficientnet_b0 
                                                 out_features=len(class_names),
                                                 bias=True).to(device))


# Define loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Set number of epochs
NUM_EPOCHS = 2

summarywriter = create_writer.create_writer("Food101",
                                            "Pretrained Model EfficientNet_B2")
results = engine.train(model=model,
                       train_dataloader=train_dataloader,
                       test_dataloader=test_dataloader,
                       optimizer=optimizer,
                       loss_fn=loss_fn,
                       epochs=NUM_EPOCHS,
                       device=device,
                       writer=summarywriter,
                       img_height=224,
                       img_width=224)

helper_functions.plot_loss_curves(results=results)
plt.show()

# Setup custom image path
data_path = Path("data/")
custom_image_path = data_path / "SampleTestSushi.jpeg"
# custom_image_path = data_path / "SampleTestPizza.jpeg"

# Download the image if it doesn't already exist
if not custom_image_path.is_file():
    with open(custom_image_path, "wb") as f:
        # When downloading from GitHub, need to use the "raw" file link
        #request = requests.get("https://raw.githubusercontent.com/marcoseraphin/PyTorch_Course/main/PyTorchVideoCourse/ZipData/SampleTestPizza.jpeg")
        request = requests.get("https://raw.githubusercontent.com/marcoseraphin/PyTorch_Course/main/PyTorchVideoCourse/ZipData/SampleTestSushi.jpeg")
        print(f"Downloading {custom_image_path}...")
        f.write(request.content)
else:
    print(f"{custom_image_path} already exists, skipping download.")

# Predict on custom image
helper_functions.pred_and_plot_image(model=model,
                    image_path=custom_image_path,
                    class_names=class_names)
plt.show()


# ONNX EXPORT
# -----------

# Save the network weights
# torch.save(model.state_dict(),
#            'pretrained_effnetb2_feature_extractor_mlcore.pth')

# # Create dummy input
# dummy_input = torch.rand(1, 3, 224, 224)

# # Define input / output names
# input_names = ["input"]
# output_names = ["output"]

# # Convert the PyTorch model to ONNX
# torch.onnx.export(model,
#                   dummy_input,
#                   "pretrained_effnetb2_feature_extractor_mlcore.onnx",
#                   verbose=True,
#                   input_names=input_names,
#                   output_names=output_names)


# # Load the ONNX model as a CoreML model
# model = onnx_coreml.convert(model='pretrained_effnetb2_feature_extractor_mlcore.onnx')

# # Save the CoreML model
# model.save('pretrained_effnetb2_feature_extractor_mlcore.mlmodel')

# CoreML Export

# Create dummy input
dummy_input = torch.rand(1, 3, 224, 224)

scripted_model = torch.jit.trace(model, dummy_input)

# mlmodel =ct.convert(scripted_model,
#     inputs=[ct.TensorType(name="test", shape=dummy_input.shape)],
#     source='auto',
#     minimum_deployment_target=ct.target.iOS15,
#     compute_units=ct.ComputeUnit.CPU_ONLY,
#     compute_precision=ct.precision.FLOAT32,
#     convert_to='mlprogram',
#     debug=True
# )

#  # Save the converted model.
# mlmodel.save('foodvisionmodel.mlpackage')

# Using image_input in the inputs parameter:
# Convert to Core ML neural network using the Unified Conversion API.
# model2 = ct.convert(
#     scripted_model,
#     inputs=[ct.TensorType(shape=dummy_input.shape)]
#  )

# 1. Load in image and convert the tensor values to float32
target_image = torchvision.io.read_image(str(custom_image_path)).type(torch.float32)

# 2. Divide the image pixel values by 255 to get them between [0, 1]
target_image = target_image / 255.0

# 4. Make sure the model is on the target device
model.to(device)

# 5. Turn on model evaluation mode and inference mode
model.eval()
with torch.inference_mode():
    # Add an extra dimension to the image
    target_image = target_image.unsqueeze(dim=0)

    # Make a prediction on image with an extra dimension and send it to the target device
    target_image_pred = model(target_image.to(device))

# 6. Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
target_image_pred_probs = torch.softmax(target_image_pred, dim=1)




# model2 = ct.convert(
#     scripted_model,
#     inputs=[ct.ImageType(shape=(1,3,224,244))])

# Set the input type as an image
# model2.input_description['InputImage'] = ('image', (3, 224, 224))
#model2.output_description[0] = ('tensor', (1, 1, 1))

# Save the converted model.
#model2.save("foodvisionmodel.mlmodel")

#mlmodel_loaded= ct.models.MLModel('foodvisionmodel.mlmodel')

#example_image = Image.open(custom_image_path).resize((224, 224))

# Make a prediction using Core ML.
#out_dict = model2.predict({input_name: example_image})

# Display its specifications
#print(mlmodel_loaded.visualize_spec)

# get model specification
# model_path = "foodvisionmodel.mlmodel"
# mlmodel = ct.models.MLModel(str(model_path))
# spec = mlmodel.get_spec()

# # get list of current output_names
# current_output_names = len(mlmodel.output_description._fd_spec)

# # rename first output in list to new_output_name
# old_name = current_output_names[0].name
# new_name = "output"
# ct.utils.rename_feature(
#     spec, old_name, new_name, rename_outputs=True
# )

# # overwite existing model spec with new renamed spec
# new_model = ct.models.MLModel(spec)
# new_model.save(model_path)

example = torch.rand(1,3,224,244)
model_cpu = model.to("cpu")

traced_model = torch.jit.trace(model_cpu, example)

from torch.utils.mobile_optimizer import optimize_for_mobile

optimized_model = optimize_for_mobile(traced_model)

model2 = ct.convert(
    optimized_model,
    inputs=[ct.ImageType(shape=(1,3,224,244))])

# Save the converted model.
model2.save("foodvisionmodel.mlmodel")