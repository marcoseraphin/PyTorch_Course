import torch
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import numpy as np

# Introduction to Tensor
# Creating a Tensor => https://pytorch.org/docs/stable/tensors.html 

# Scalar
scalar = torch.tensor(7)
print (f"Scalar => {scalar}")
print (f"scalar.ndim => {scalar.ndim}")
print (f"Scalar.item() {scalar.item()}")

# Vector
vector = torch.tensor([3,7])
print (f"vector => {vector}")
print (f"vector.ndim => {vector.ndim}")
print (f"vector.shape => {vector.shape}")

# Matrix
MATRIX = torch.tensor([[7,8],[9,10]])
print (f"MATRIX => {MATRIX}")
print (f"MATRIX.ndim => {MATRIX.ndim}")
print (f"MATRIX[1] => {MATRIX[1]}")

# TENSOR
TENSOR = torch.tensor([[[1,2,3],[3,6,8],[2,4,5]]])
print (f"TENSOR => {TENSOR}")
print (f"TENSOR.ndim => {TENSOR.ndim}")
print (f"TENSOR.shape => {TENSOR.shape}")
print (f"TENSOR[0] => {TENSOR[0]}")
print (f"TENSOR[0][1] => {TENSOR[0][1]}")

# Random tensors
random_tensor = torch.rand(3,4)
print (f"random_tensor => {random_tensor}")
print (f"random_tensor.ndim => {random_tensor.ndim}")

# Random image tensor
# Size = Heigth, Width color channel (RGB)
random_image_tensor = torch.rand(size=(224,224,3))
#print (f"random_image_tensor => {random_image_tensor}")
print (f"random_image_tensor.ndim => {random_image_tensor.ndim}")

# Zeros and ones
zero_tensor = torch.zeros(size=(3,4))
print (f"zero_tensor => {zero_tensor}")

zero_ones = torch.ones(size=(3,4))
print (f"zero_ones => {zero_ones}")

# Create a range of tensors
range_tensor = torch.arange(start=1,end=10, step=2)
print (f"range_tensor => {range_tensor}")

# Tensor Like
ten_zeros_tensor_like = torch.zeros_like(range_tensor)
print (f"ten_zeros_tensor_like => {ten_zeros_tensor_like}")

# Tensor DataType
mps_device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {mps_device}")

float_32_tensor = torch.tensor([3.0, 6.0, 9.0],dtype=None,
                                               device=None, # device = mps_device 
                                               requires_grad=False)
print (f"float_32_tensor => {float_32_tensor}")
print (f"float_32_tensor DataType => {float_32_tensor.dtype}")

float_16_tensor = torch.tensor([3.0, 6.0, 9.0],dtype=torch.float16)
print (f"float_16_tensor => {float_16_tensor}")

# Get Tensor Atttributes
some_tensor = torch.tensor([3,4], device=mps_device)
print (f"some_tensor DataType => {some_tensor.dtype}")
print (f"some_tensor Shape => {some_tensor.shape}")
print (f"some_tensor Device => {some_tensor.device}")

# Manipulating Tensors (Tensor operations)
tensor1 = torch.tensor([1,2,3])
tensor_add = tensor1 + 10
print (f"tensor_add => {tensor_add}")

tensor_mult = tensor1 * 10
print (f"tensor_mult => {tensor_mult}")
print (f"tensor_mult.mul(10) => {tensor_mult.mul(10)}")

# Matrix multiplication <=> Element wise multiplication
tensor2 = torch.tensor([1,2,3])
tensor3 = torch.tensor([4,5,6])
print (f"Tensor Element-wise multiplacation  => {tensor2 * tensor3}")
print (f"Tensor Matrix multiplication matmul => {torch.matmul(tensor2, tensor3)}")

# Manipulate shape of a tensor for matrix multiplication
# the inner dimensions must match (2x2)
tensor_a = torch.tensor([[1,2],
                         [3,4],
                         [5,6]])

tensor_b = torch.tensor([[7,10],
                         [8,11],
                         [9,12]])

print (f"tensor_b => {tensor_b}")
print (f"tensor_b Transpose => {tensor_b.T}")

# Result has the shape of the outer dimensions (3x3)
print (f"Tensor Matrix multiplication matmul after Transpose => {torch.matmul(tensor_a, tensor_b.T)}")

# Finding and Min, Max, Mean, Sum etc (tensor aggregation)
tensor_x = torch.tensor([4, 2, 6, 8, 9, 3])
print (f"tensor_x Min => {torch.min(tensor_x)}")
print (f"tensor_x Max => {tensor_x.max()}")
print (f"tensor_x Mean => {tensor_x.type(torch.float32).mean()}") # must be casted to float32 for mean()
print (f"tensor_x Max => {tensor_x.sum()}")

print (f"tensor_x ArgMin Index of Minimum => {tensor_x.argmin()}")
print (f"tensor_x ArgMax Index of Minimum => {tensor_x.argmax()}")

# Reshaping, stacking, squeezing and unsqueezing tensors
tensor_y = torch.tensor([12, 32, 56, 78, 99, 13])
print (f"tensor_y => {tensor_y}")
print (f"tensor_x Shape => {tensor_y.shape}")

# Reshape
y_reshape_tensor = tensor_y.reshape(1,6)
print (f"After Reshape: y_reshape_tensor => {y_reshape_tensor}")
print (f"After Reshape: y_reshape_tensor Shape => {y_reshape_tensor.shape}")

# Change the view (tensor_x and z_view_tensor are the same (memory))
z_view_tensor = tensor_x.view(1, 6)
print (f"After View: z_view_tensor Shape => {z_view_tensor.shape}")

# Stack tensors
tensor_stack = torch.stack([tensor_x, tensor_y])
print (f"After Stack tensor_x and tensor_y => {tensor_stack}")

# Squeeze and Unsqueeze
tensor_zz = torch.tensor([[1,2]])
tensor_s = torch.squeeze(tensor_zz)
print (f"After Squeeze tensor_zz => {tensor_s}")

tensor_s2 = tensor_zz.unsqueeze(dim=0)
print (f"After UnSqueeze tensor_zz => {tensor_s2}")

# Permute tensor
tensor_p1 = torch.randn([3,224,244])
print (f"Berfore Permute tensor_p1 => {tensor_p1.size()}")
tensor_p2 = tensor_p1.permute(1,2,0)
print (f"After Permute tensor_p1 => {tensor_p2.size()}")

# Indexing  (selecting data from a tensor)
tensor_index = torch.arange(1, 10).reshape(1,3,3)
print (f"tensor_index       => {tensor_index}")
print (f"tensor_index Shape => {tensor_index.shape}")
print (f"tensor_index [0,1] => {tensor_index[0,1]}")
print (f"tensor_index [0,1,2] => {tensor_index[0,1,2]}")
print (f"tensor_index [:,1] => {tensor_index[:,1]}") # all of first dimension
print (f"tensor_index [:,:,1] => {tensor_index[:,:,1]}") 

# Numpy array to tensor
np_array = np.arange(1.0, 8.0)
tensor_np = torch.from_numpy(np_array)
print (f"Numpy np_array     => {np_array}")
print (f"tensor_np          => {tensor_np}")

# Reproducibility (taking random out of random)

rand_tensor = torch.randn(3,3)
print (f"rand_tensor => {rand_tensor}")

RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)
rand_tensor_seed1 = torch.randn(3,3)
print (f"rand_tensor_seed1 => {rand_tensor_seed1}")

torch.manual_seed(RANDOM_SEED)
rand_tensor_seed2 = torch.randn(3,3)
print (f"rand_tensor_seed2 => {rand_tensor_seed2}")
print (f"Equal =>  {rand_tensor_seed1 == rand_tensor_seed2}")

# Running tensor and PyTorch on GPUs
# Puttin tensors (and models) on the GPU => Faster
mps_device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {mps_device}")

tensor_CPU = torch.tensor([1, 2, 3], device="cpu")
print (f"tensor_CPU Device => {tensor_CPU.device}")

tensor_GPU = torch.tensor([1, 2, 3], device="mps")
print (f"tensor_GPU        => {tensor_GPU}")
print (f"tensor_GPU Device => {tensor_GPU.device}")

tensor_GPU2 = tensor_CPU.to(mps_device)   # move tensor to GPU (Apple Metal Performance Shader = mps)
print (f"tensor_GPU2        => {tensor_GPU2}")
print (f"tensor_GPU2 Device => {tensor_GPU2.device}")

tensor_back_cpu = tensor_GPU2.cpu()
print (f"tensor_back_cpu        => {tensor_back_cpu}")
print (f"tensor_back_cpu Device => {tensor_back_cpu.device}")
