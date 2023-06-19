import torchvision
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


# Define the train and test dir
train_dir_20_percent = "..."
test_dir = "..."

# Setup auto transform
weights = torchvision.models.ResNet50_Weights.DEFAULT  # Use "101" instead of "50" for ResNet101
transform = weights.transforms()

# Setup train and test dataset
train_dataset = ImageFolder(root=train_dir_20_percent, transform=transform)
test_dataset = ImageFolder(root=test_dir, transform=transform)

train_dataloader = DataLoader(dataset=train_dataset,
                              batch_size=4,
                              shuffle=True)

test_dataloader = DataLoader(dataset=test_dataset,
                             batch_size=4,
                             shuffle=False)
