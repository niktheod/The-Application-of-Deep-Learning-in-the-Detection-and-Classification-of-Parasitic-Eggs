import torch
import torchvision
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"

weights = torchvision.models.ResNet50_Weights.DEFAULT  # Use "101" instead of "50" for ResNet101
model = torchvision.models.resnet50(weights=weights).to(device)  # Use "101" instead of "50" for ResNet101

# Freeze all the layers
for param in model.parameters():
    param.requires_grad = False

# Adapt the fully connected layers to our problem
model.fc = nn.Sequential(
    # nn.Dropout(0.2),  # Uncomment to add a Dropout layer
    nn.Linear(in_features=2048,
              out_features=11)
).to(device)
