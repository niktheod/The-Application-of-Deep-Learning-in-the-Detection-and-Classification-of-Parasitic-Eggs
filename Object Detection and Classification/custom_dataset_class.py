import torch
import json
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from torchvision import transforms


device = "cuda" if torch.cuda.is_available() else "cpu"

labels_file_dir = "..."
with open(labels_file_dir) as f:
    labels = json.load(f)

SIZE = 224  # or 512


class CustomDataset(Dataset):
    """
    CustomDataset is a PyTorch dataset class that loads images and their corresponding annotations for object detection
    tasks.

    Args:
        files_dir (str): The directory path where the image files are stored.
        labels (dict): A dictionary containing the annotations for the images.
        transform (callable, optional): A function/transform that takes in an image and returns a transformed version.
            Default is None.

    Attributes:
        files_dir (str): The directory path where the image files are stored.
        size (int): The target size of the images after resizing.
        labels (dict): A dictionary containing the annotations for the images.
        transform (callable): A function/transform that takes in an image and returns a transformed version.
        images (list): A list of paths to the image files.
        classes (list): A list of class labels.

    Methods:
        __getitem__(self, idx): Retrieves the image and its corresponding annotations at the given index.
        __len__(self): Returns the total number of images in the dataset.
    """

    def __init__(self, files_dir, labels=labels, transform=None):
        """
        Initializes a CustomDataset instance.

        Args:
            files_dir (str): The directory path where the image files are stored.
            labels (dict): A dictionary containing the annotations for the images.
            transform (callable, optional): A function/transform that takes in an image and returns a transformed version.
                Default is None.
        """
        self.files_dir = files_dir
        self.size = SIZE
        self.labels = labels
        self.transform = transform
        self.images = [str(x) for x in Path(files_dir).glob("*/*.jpg")]
        self.classes = ["background", "Ascaris lumbricoides", "Capillaria philippinensis", "Enterobius vermicularis",
                        "Fasciolopsis buski", "Hookworm egg", "Hymenolepis diminuta", "Hymenolepis nana",
                        "Opisthorchis viverrine", "Paragonimus spp", "Taenia spp. egg", "Trichuris trichiura"]

    def __getitem__(self, idx):
        """
        Retrieves the image and its corresponding annotations at the given index.

        Args:
            idx (int): The index of the image to retrieve.

        Returns:
            img (PIL.Image.Image or torch.Tensor): The loaded image.
            target (dict): A dictionary containing the annotations for the image.
                It includes the following keys:
                - 'boxes': A tensor representing the bounding box coordinates of the object in the image.
                - 'labels': A tensor representing the label index of the object in the image.
                - 'image_id': A tensor representing the image index.
                - 'area': A tensor representing the area of the object in the image.
                - 'iscrowd': A tensor representing whether the object is a crowd.
        """
        image_path = self.images[idx]
        image_name = image_path.split("/")[-1]
        category = image_name[:-9]
        img = Image.open(image_path)
        width, height = img.size

        if self.transform:
            img = self.transform(img)

        img = transforms.Resize(size=(self.size, self.size))(img)

        for image in self.labels["images"]:
            if image["file_name"] == image_name:
                image_id = image["id"]
                break

        for annotation in self.labels["annotations"]:
            if annotation["image_id"] == image_id:
                bbox = annotation["bbox"][:]
                break

        bbox[2] += bbox[0]
        bbox[3] += bbox[1]

        if bbox[0] >= 0:
            bbox[0] /= width
        else:
            bbox[0] = 0

        if bbox[1] >= 0:
            bbox[1] /= height
        else:
            bbox[1] = 0

        if bbox[2] <= width:
            bbox[2] /= width
        else:
            bbox[2] = 1

        if bbox[3] <= height:
            bbox[3] /= height
        else:
            bbox[3] = 1

        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        bbox = torch.tensor(bbox)
        bbox *= self.size
        target = {"boxes": bbox.unsqueeze(dim=0).to(device),
                  "labels": torch.tensor(self.classes.index(category)).unsqueeze(dim=0).to(device),
                  "image_id": torch.tensor(idx),
                  "area": torch.tensor(area),
                  "iscrowd": torch.tensor(0)}

        return img, target

    def __len__(self):
        """
        Returns the total number of images in the dataset.

        Returns:
            length (int): The total number of images in the dataset.
        """
        return len(self.images)
