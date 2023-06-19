import torch


device = "cuda" if torch.cuda.is_available() else "cpu"


def accuracy(y_pred, y_true):
    """
    Calculates the accuracy of the predicted labels compared to the true labels.

    Args:
        y_pred (torch.Tensor): Predicted labels.
        y_true (torch.Tensor): True labels.

    Returns:
        torch.Tensor: Accuracy as a percentage.

    """
    count = torch.eq(y_pred, y_true).sum()
    acc = (count / len(y_true)) * 100
    return acc


def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_func: torch.nn.Module,
               acc_func,
               optimizer: torch.optim.Optimizer,
               device: torch.device = device):
    """
    Performs a single training step on the given model.

    Args:
        model (torch.nn.Module): The model to train.
        dataloader (torch.utils.data.DataLoader): The data loader providing the training data.
        loss_func (torch.nn.Module): The loss function to calculate the training loss.
        acc_func: The accuracy function to calculate the training accuracy.
        optimizer (torch.optim.Optimizer): The optimizer to update the model's parameters.
        device (torch.device): The device on which the training will be performed. Default is 'device'.

    Returns:
        tuple: Tuple containing the training loss and training accuracy.

    """
    train_loss, train_acc = 0, 0

    model.train()

    for X_train, y_train in dataloader:
        X_train, y_train = X_train.to(device), y_train.to(device)

        train_pred = model(X_train)

        loss = loss_func(train_pred, y_train)
        train_loss += loss
        train_acc += acc_func(train_pred.argmax(dim=1), y_train)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    train_loss /= len(dataloader)
    train_acc /= len(dataloader)

    print(f"\tTraining Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.2f}%")
    return train_loss, train_acc


def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_func: torch.nn.Module,
              acc_func,
              device: torch.device = device):
    """
    Performs a single testing step on the given model.

    Args:
        model (torch.nn.Module): The model to evaluate.
        dataloader (torch.utils.data.DataLoader): The data loader providing the test data.
        loss_func (torch.nn.Module): The loss function to calculate the test loss.
        acc_func: The accuracy function to calculate the test accuracy.
        device (torch.device): The device on which the evaluation will be performed. Default is 'device'.

    Returns:
        tuple: Tuple containing the test loss and test accuracy.

    """
    test_loss, test_acc = 0, 0

    model.eval()
    with torch.inference_mode():
        for X_test, y_test in dataloader:
            X_test, y_test = X_test.to(device), y_test.to(device)

            test_pred = model(X_test)

            test_loss += loss_func(test_pred, y_test)
            test_acc += acc_func(test_pred.argmax(dim=1), y_test)

        test_loss /= len(dataloader)
        test_acc /= len(dataloader)

    print(f"\tTest Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")
    return test_loss, test_acc


def train(model, epochs, train_dataloader, test_dataloader, loss_func, optimizer, acc_func, scheduler,
          dynamic_scheduler: bool, device=device):
    """
    Trains the given model for the specified number of epochs.

    Args:
        model (torch.nn.Module): The model to train.
        epochs (int): The number of training epochs.
        train_dataloader (torch.utils.data.DataLoader): The data loader providing the training data.
        test_dataloader (torch.utils.data.DataLoader): The data loader providing the test data.
        loss_func (torch.nn.Module): The loss function to calculate the training and test losses.
        optimizer (torch.optim.Optimizer): The optimizer to update the model's parameters.
        acc_func: The accuracy function to calculate the training and test accuracies.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler.
        dynamic_scheduler (bool): Specifies whether the scheduler adjusts based on the training loss. If False, the
                                  scheduler will update after each training step based on the scheduler's step. If True,
                                  the scheduler will update after each training step based on the training loss.
        device (torch.device): The device on which the training will be performed. Default is 'device'.

    Returns:
        dict: Dictionary containing the training and test losses and accuracies for each epoch.

    """
    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []}

    for epoch in range(epochs):
        print(f"Epoch: {epoch}")
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_func=loss_func,
                                           optimizer=optimizer,
                                           acc_func=acc_func,
                                           device=device)

        if dynamic_scheduler:
            scheduler.step(train_loss)
        else:
            scheduler.step()

        test_loss, test_acc = test_step(model=model,
                                        dataloader=test_dataloader,
                                        loss_func=loss_func,
                                        acc_func=acc_func,
                                        device=device)

        results["train_loss"].append(train_loss.item())
        results["train_acc"].append(train_acc.item())
        results["test_loss"].append(test_loss.item())
        results["test_acc"].append(test_acc.item())

    return results
