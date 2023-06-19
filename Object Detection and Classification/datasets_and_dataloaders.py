import torch
import torchvision
from custom_dataset_class import CustomDataset
from torch.utils.data import DataLoader


def collate_fn(batch):
    """
    Custom collate function to handle variable-sized inputs in the batch.

    Args:
        batch (list): A list of samples from the dataset.

    Returns:
        images (torch.Tensor): A tensor containing the batch of images.
        targets (list): A list containing the targets for each image in the batch.
    """
    images = [item[0].unsqueeze(dim=0) for item in batch]
    images = torch.cat(images, dim=0)
    targets = [item[1] for item in batch]
    return images, targets


BATCH_SIZE = 16  # or 32

# Setup auto transform
weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
transform = weights.transforms()

# Set up train and test datasets
train_dir_10_percent = "..."
train_dir_20_percent = "..."
test_dir = "..."

train_dataset_10_percent = CustomDataset(files_dir=train_dir_10_percent,
                                         transform=transform)

train_dataset_20_percent = CustomDataset(files_dir=train_dir_20_percent,
                                         transform=transform)

test_dataset = CustomDataset(files_dir=test_dir,
                             transform=transform)

# Setup dataloaders
train_dataloader_10_percent = DataLoader(dataset=train_dataset_10_percent,
                                         batch_size=BATCH_SIZE,
                                         shuffle=True,
                                         collate_fn=collate_fn)

train_dataloader_20_percent = DataLoader(dataset=train_dataset_20_percent,
                                         batch_size=BATCH_SIZE,
                                         shuffle=True,
                                         collate_fn=collate_fn)

test_dataloader = DataLoader(dataset=test_dataset,
                             batch_size=BATCH_SIZE,
                             shuffle=False,
                             collate_fn=collate_fn)
