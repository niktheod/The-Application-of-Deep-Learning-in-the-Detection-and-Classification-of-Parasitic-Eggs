import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

device = "cuda" if torch.cuda.is_available() else "cpu"

weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT


def get_model_with_frozen_layers():
    """
    Create a Faster R-CNN model with frozen layers.

    Returns:
        model (torch.nn.Module): Faster R-CNN model with frozen layers.
    """
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)

    # Freeze all parameters in the model
    for parameter in model.parameters():
        parameter.requires_grad = False

    # Modify the box predictor to match the number of classes in the dataset
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 12)

    return model


def get_model_without_frozen_layers():
    """
    Create a Faster R-CNN model without frozen layers.

    Returns:
        model (torch.nn.Module): Faster R-CNN model without frozen layers.
    """
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)

    # Modify the box predictor to match the number of classes in the dataset
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 12)

    return model


# Define some functions that will help to evaluate the model
def bbox_accuracy(pred_boxes, true_boxes):
    """
    Calculate the bounding box accuracy.

    Args:
        pred_boxes (list): List of predicted bounding boxes.
        true_boxes (list): List of ground truth bounding boxes.

    Returns:
        bbox_acc (float): Bounding box accuracy as a percentage.
        successful_tries (list): List indicating successful tries (True) or not (False) for each sample.
    """
    cnt = 0
    successful_tries = []
    for i in range(len(true_boxes)):
        if pred_boxes[i] is None:
            successful_tries.append(False)
            continue
        if pred_boxes[i][0] > true_boxes[i][2] or pred_boxes[i][2] < true_boxes[i][0] or \
                pred_boxes[i][1] > true_boxes[i][3] or pred_boxes[i][3] < true_boxes[i][1]:
            successful_tries.append(False)
            continue
        max_xmin = max(pred_boxes[i][0], true_boxes[i][0])
        max_ymin = max(pred_boxes[i][1], true_boxes[i][1])
        min_xmax = min(pred_boxes[i][2], true_boxes[i][2])
        min_ymax = min(pred_boxes[i][3], true_boxes[i][3])
        intersection = (min_xmax - max_xmin) * (min_ymax - max_ymin)

        area_pred = (pred_boxes[i][2] - pred_boxes[i][0]) * (pred_boxes[i][3] - pred_boxes[i][1])
        area_true = (true_boxes[i][2] - true_boxes[i][0]) * (true_boxes[i][3] - true_boxes[i][1])
        union = area_pred + area_true - intersection

        if intersection / union >= 0.5:
            cnt += 1
            successful_tries.append(True)
        else:
            successful_tries.append(False)

    return (cnt / len(true_boxes)) * 100, successful_tries


def class_accuracy(pred_labels, true_labels):
    """
    Calculate the classification accuracy.

    Args:
        pred_labels (list): List of predicted labels.
        true_labels (list): List of ground truth labels.

    Returns:
        class_acc (float): Classification accuracy as a percentage.
    """
    cnt = 0
    denominator = len(true_labels)
    for i in range(len(true_labels)):
        if pred_labels[i] is None:
            denominator -= 1
            continue
        if pred_labels[i] == true_labels[i]:
            cnt += 1

    return (cnt / denominator) * 100 if denominator > 0 else 0


# Define a train and test step function
def train_step(model, dataloader, optimizer, device=device):
    """
    Perform a single training step.

    Args:
        model (torch.nn.Module): The model to train.
        dataloader (torch.utils.data.DataLoader): Dataloader for the training data.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        device (str): Device to use for training (default is "cuda" if available, otherwise "cpu").

    Returns:
        train_loss (float): Average training loss.
    """
    train_loss = 0

    model.train()

    for images, targets in dataloader:
        images = images.to(device)

        loss = sum(model(images, targets).values())
        train_loss += loss

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    train_loss /= len(dataloader)

    print(f"\tTrain Loss: {train_loss:.4f}")
    return train_loss


def test_step(model, dataloader, bbox_acc_func, class_acc_func, device=device):
    """
    Perform a single testing/validation step.

    Args:
        model (torch.nn.Module): The model to evaluate.
        dataloader (torch.utils.data.DataLoader): Dataloader for the testing/validation data.
        bbox_acc_func (function): Function to calculate bounding box accuracy.
        class_acc_func (function): Function to calculate classification accuracy.
        device (str): Device to use for testing/validation (default is "cuda" if available, otherwise "cpu").

    Returns:
        test_bbox_acc (float): Bounding box detection accuracy as a percentage.
        test_label_acc (float): Classification accuracy as a percentage.
    """
    test_bbox_acc, test_label_acc = 0, 0

    model.eval()
    with torch.inference_mode():
        for images, targets in dataloader:
            images = images.to(device)
            true_boxes = [x["boxes"].squeeze() for x in targets]
            true_labels = [x["labels"].item() for x in targets]

            test_pred = model(images, targets)
            pred_boxes = []
            pred_labels = []
            for x in test_pred:
                if list(x["boxes"]):
                    pred_boxes.append(x["boxes"][0])
                else:
                    pred_boxes.append(None)

                if list(x["labels"]):
                    pred_labels.append(x["labels"][0])
                else:
                    pred_labels.append(None)

            bbox_acc, successful_tries = bbox_acc_func(pred_boxes, true_boxes)
            test_bbox_acc += bbox_acc

            for i in range(len(successful_tries)):
                if not successful_tries[i]:
                    pred_labels[i] = None

            test_label_acc += class_acc_func(pred_labels, true_labels)

        test_bbox_acc /= len(dataloader)
        test_label_acc /= len(dataloader)

    print(f"\tTest Bbox Detecting Accuracy: {test_bbox_acc:.2f}%, Test Classifying Accuracy: {test_label_acc:.2f}%")
    return test_bbox_acc, test_label_acc


def train(model, epochs, train_dataloader, test_dataloader, optimizer, bbox_acc_func, class_acc_func, device=device):
    """
    Train the model.

    Args:
        model (torch.nn.Module): The model to train.
        epochs (int): Number of training epochs.
        train_dataloader (torch.utils.data.DataLoader): Dataloader for the training data.
        test_dataloader (torch.utils.data.DataLoader): Dataloader for the testing/validation data.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        bbox_acc_func (function): Function to calculate bounding box accuracy.
        class_acc_func (function): Function to calculate classification accuracy.
        device (str): Device to use for training and testing/validation (default is "cuda" if available, otherwise "cpu").

    Returns:
        results (dict): Dictionary containing the training and testing/validation results.
    """
    results = {"train_loss": [], "test_bbox_acc": [], "test_label_acc": []}

    for epoch in range(epochs):
        print(f"Epoch: {epoch}")
        train_loss = train_step(model=model, dataloader=train_dataloader, optimizer=optimizer, device=device)

        test_bbox_acc, test_label_acc = test_step(model=model, dataloader=test_dataloader, bbox_acc_func=bbox_acc_func,
                                                  class_acc_func=class_acc_func, device=device)

        results["train_loss"].append(train_loss.item())
        results["test_bbox_acc"].append(test_bbox_acc)
        results["test_label_acc"].append(test_label_acc)

    return results
