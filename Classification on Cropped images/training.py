import torch


def accuracy(y_pred, y_true):
    """
    Computes the accuracy of the predicted labels compared to the true labels.

    Args:
        y_pred (torch.Tensor): Predicted labels.
        y_true (torch.Tensor): True labels.

    Returns:
        float: Accuracy in percentage.
    """
    count = torch.eq(y_pred, y_true).sum()
    acc = (count / len(y_true)) * 100
    return acc


def train_step(model, dataloader, loss_func, acc_func, optimizer, device=torch.device('cpu')):
    """
    Performs a single training step on the given model.

    Args:
        model (torch.nn.Module): The model to be trained.
        dataloader (torch.utils.data.DataLoader): DataLoader object containing the training data.
        loss_func (torch.nn.Module): Loss function to calculate the loss.
        acc_func (function): Function to calculate the accuracy.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        device (torch.device): Device to perform the computation on (default: 'cpu').

    Returns:
        tuple: Training loss and accuracy.
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


def test_step(model, dataloader, loss_func, acc_func, device=torch.device('cpu')):
    """
    Performs a single testing step on the given model.

    Args:
        model (torch.nn.Module): The model to be tested.
        dataloader (torch.utils.data.DataLoader): DataLoader object containing the testing data.
        loss_func (torch.nn.Module): Loss function to calculate the loss.
        acc_func (function): Function to calculate the accuracy.
        device (torch.device): Device to perform the computation on (default: 'cpu').

    Returns:
        tuple: Testing loss and accuracy.
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


def train(model, epochs, train_dataloader, test_dataloader, loss_func, optimizer, acc_func, lr_scheduler=None,
          device=torch.device('cpu')):
    """
    Trains the given model for the specified number of epochs.

    Args:
        model (torch.nn.Module): The model to be trained.
        epochs (int): Number of epochs to train the model for.
        train_dataloader (torch.utils.data.DataLoader): DataLoader object containing the training data.
        test_dataloader (torch.utils.data.DataLoader): DataLoader object containing the testing data.
        loss_func (torch.nn.Module): Loss function to calculate the loss.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        acc_func (function): Function to calculate the accuracy.
        lr_scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler (default: None).
        device (torch.device): Device to perform the computation on (default: 'cpu').

    Returns:
        dict: Dictionary containing the training and testing loss and accuracy for each epoch.
    """
    results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    for epoch in range(epochs):
        print(f"Epoch: {epoch}")
        train_loss, train_acc = train_step(model=model, dataloader=train_dataloader, loss_func=loss_func,
                                           acc_func=acc_func, optimizer=optimizer, device=device)

        if lr_scheduler is not None:
            lr_scheduler.step()

        test_loss, test_acc = test_step(model=model, dataloader=test_dataloader, loss_func=loss_func,
                                        acc_func=acc_func, device=device)

        results["train_loss"].append(train_loss.item())
        results["train_acc"].append(train_acc.item())
        results["test_loss"].append(test_loss.item())
        results["test_acc"].append(test_acc.item())

    return results
