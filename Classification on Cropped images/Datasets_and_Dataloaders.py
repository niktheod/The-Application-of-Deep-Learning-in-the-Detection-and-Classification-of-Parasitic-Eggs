from torchvision import transforms
from torch.utils.data import DataLoader
from Custom_Dataset_Class import CustomDataset

# Define the transforms for training and testing data
train_transform = transforms.Compose([
    transforms.Resize(size=(size, size)),  # Resize the image to the desired size
    transforms.ToTensor()  # Convert the image to a tensor
])

test_transform = transforms.Compose([
    transforms.Resize(size=(size, size)),  # Resize the image to the desired size
    transforms.ToTensor()  # Convert the image to a tensor
])

# Define the directories for the training and testing data
train_dir_10_percent = "..."
train_dir_20_percent = "..."
test_dir = "..."

# Create the custom datasets for training and testing
train_dataset_10_percent = CustomDataset(files_dir=train_dir_10_percent, transform=train_transform)
train_dataset_20_percent = CustomDataset(files_dir=train_dir_20_percent, transform=train_transform)
test_dataset = CustomDataset(files_dir=test_dir, transform=test_transform)

# Create the data loaders for training and testing
train_dataloader_10_percent = DataLoader(dataset=train_dataset_10_percent,
                                         batch_size=batch_size,
                                         shuffle=True)

train_dataloader_20_percent = DataLoader(dataset=train_dataset_20_percent,
                                         batch_size=batch_size,
                                         shuffle=True)

test_dataloader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             shuffle=False)
