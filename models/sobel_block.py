# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 16:57:05 2024

@author: olivi
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
# Sobel filter definition
# Sobel filter definition

def get_sobel_filters():
    # Sobel filter for horizontal edges
    sobel_filter_x = torch.tensor([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Shape [1, 1, 3, 3]

    # Sobel filter for vertical edges
    sobel_filter_y = torch.tensor([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Shape [1, 1, 3, 3]

    return sobel_filter_x, sobel_filter_y

class Directional_Block(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Directional_Block, self).__init__()

        # First, define the Sobel filters
        sobel_filter_x, sobel_filter_y = get_sobel_filters()
        
        # Create convolution for Sobel filters with 3 groups: each group processes one channel
        self.conv1_Sobel = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1, bias=False, groups=3)

        # Assign filters: keep the first channel as it is, apply Sobel X to the second and Sobel Y to the third
        identity_filter = torch.tensor([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Shape [1, 1, 3, 3]

        # Assign the filters correctly
        with torch.no_grad():  
            self.conv1_Sobel.weight.data[0] = identity_filter  # Identity filter for first channel
            self.conv1_Sobel.weight.data[1] = sobel_filter_x  # Sobel X filter for second channel
            self.conv1_Sobel.weight.data[2] = sobel_filter_y  # Sobel Y filter for third channel

        # Freeze the Sobel layer parameters
        self.conv1_Sobel.weight.requires_grad = False
        
    def forward(self, x_in):
        # Pass through the Sobel filters (first channel remains original)
        x = self.conv1_Sobel(x_in)

        # Apply absolute value to Sobel filter output for second and third channels
        x = torch.abs(x)

        # Apply activation (Leaky ReLU only to second and third channels)
        x = F.leaky_relu(x)

        return x


# # Instantiate the block
# directional_block = Directional_Block()

# # # Example input image tensor
# # x_in = torch.randn(1, 3, 224, 224)  # Batch size 1, 3 channels, 224x224 image

# # # Forward pass through the block
# # output = directional_block(x_in)

# # # Output shape
# # output.shape

# # Instantiate the Directional_Block
# directional_block = Directional_Block()

# # Load the image using PIL and convert to grayscale
# image_path = r'C:\Users\olivi\Desktop\airLeakage\data_1\images\val\image_015.png'
# image = Image.open(image_path).convert('L')  # Convert to grayscale

# # Convert image to a tensor and add batch dimension
# transform = transforms.ToTensor()
# image_tensor = transform(image).unsqueeze(0)  # Shape [1, 1, H, W]

# # Repeat the grayscale image to create 3 channels (Shape [1, 3, H, W])
# image_tensor = image_tensor.repeat(1, 3, 1, 1)

# # Pass the image through the Directional_Block
# output = directional_block(image_tensor)

# # Extract the output channels (avg, sobel_x, sobel_y)
# avg_output = output[0, 0, :, :].detach().numpy()
# sobel_x_output = output[0, 1, :, :].detach().numpy()
# sobel_y_output = output[0, 2, :, :].detach().numpy()

# # Visualize the outputs
# plt.figure(figsize=(12, 4))

# # Original Image
# plt.subplot(1, 4, 1)
# plt.imshow(image, cmap='gray')
# plt.title('Original Image')

# # Averaging filter output
# plt.subplot(1, 4, 2)
# plt.imshow(avg_output, cmap='gray')
# plt.title('Averaging Filter Output')

# # Sobel X output
# plt.subplot(1, 4, 3)
# plt.imshow(sobel_x_output, cmap='gray')
# plt.title('Sobel X Output')

# # Sobel Y output
# plt.subplot(1, 4, 4)
# plt.imshow(sobel_y_output, cmap='gray')
# plt.title('Sobel Y Output')

# plt.show()