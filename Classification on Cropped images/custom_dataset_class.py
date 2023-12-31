import os
import pathlib
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from PIL import Image
import json


labels_file = "..."


class CustomDataset(Dataset):
    """Custom Dataset class for image classification with bounding box cropping.

    This dataset class is designed to load images and their corresponding class labels
    from a directory structure where each class has its own subdirectory. It also supports
    cropping the images based on bounding box coordinates obtained from a JSON file.

    Args:
        files_dir (str): Directory path containing the image files organized by class.
        transform (callable, optional): A function/transform to apply to the cropped image.
            Default: None.

    Attributes:
        paths (list): List of paths to the image files.
        classes (list): List of class names extracted from the directory structure.
        class_to_idx (dict): Mapping from class names to class indices.

    """

    def __init__(self, files_dir, transform=None):
        self.paths = list(pathlib.Path(files_dir).glob("*/*.jpg"))
        self.classes, self.class_to_idx = self.find_classes(files_dir)
        self.transform = transform

    @staticmethod
    def find_classes(files_dir):
        """Finds and sorts the class names from the directory structure.

        Args:
            files_dir (str): Directory path containing the image files organized by class.

        Returns:
            classes (list): List of sorted class names.
            class_to_idx (dict): Mapping from class names to class indices.

        """
        classes = []
        for filename in os.listdir(files_dir):
            classes.append(filename)

        classes = list(sorted(classes))
        class_to_idx = {classes[i]: i for i in range(len(classes))}

        return classes, class_to_idx

    @staticmethod
    def find_bbox(image_path):
        """Finds the bounding box coordinates for an image.

        Args:
            image_path (str): Path to the image file.

        Returns:
            list: List of bounding box coordinates [x, y, width, height].

        """
        filename = image_path.split("/")[-1]
        with open(labels_file) as f:
            labels = json.load(f)

        images = labels["images"]
        annotations = labels["annotations"]

        for image in images:
            if image["file_name"] == filename:
                image_id = image["id"]
                break

        for annotation in annotations:
            if annotation["image_id"] == image_id:
                return annotation["bbox"]

    def load_image(self, index):
        """Loads an image from the dataset.

        Args:
            index (int): Index of the image to load.

        Returns:
            PIL.Image: Loaded image.

        """
        image_path = self.paths[index]
        return Image.open(image_path)

    def __len__(self):
        """Returns the total number of images in the dataset.

        Returns:
            int: Number of images in the dataset.

        """
        return len(self.paths)

    def __getitem__(self, index):
        """Gets the image and its corresponding class index at the given index.

        Args:
            index (int): Index of the image to retrieve.

        Returns:
            tuple: A tuple containing the cropped image and its class index.

        """
        img = self.load_image(index)
        class_name = self.paths[index].parent.name
        class_idx = self.class_to_idx[class_name]

        bbox = self.find_bbox(str(self.paths[index]))

        cropped_image = TF.crop(img,
                                top=bbox[1],
                                left=bbox[0],
                                height=bbox[3],
                                width=bbox[2])

        if self.transform:
            img = self.transform(cropped_image)

        return img, class_idx
