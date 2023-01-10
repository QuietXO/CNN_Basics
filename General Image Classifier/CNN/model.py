# Import libraries
from . import visual

# Torch libraries
import torch
import torch.nn as nn


# TODO: if accuracy under certain percentage repeat training until hit, if falls too low STOP
def train_model(model, data_loader, criterion, optimizer, n_epochs, device=torch.device('cpu'),
                print_steps=1, print_epochs=1, loss_acc=4):
    """
    Start the training of the model parameters
    :param model: Model on which the test will be run
    :param data_loader: DataLoader you want to use for testing
    :param criterion: Type of loss you want to use
    :param optimizer: Type of optimization function
    :param n_epochs: Numer of epochs to be done
    :param device: Torch device on which the test will run (processor by default)
    :param print_steps: Number of printed steps per epoch (one per epoch by default)
    :param print_epochs: Print every n-th epoch (every epoch by default)
    :param loss_acc: Accuracy of the printed loss (4 decimal by default)
    :return: None
    """

    # Variables for epoch print
    print_epochs = (n_epochs if print_epochs is None else print_epochs)
    n_total_steps = len(data_loader)
    mean_loss = 0

    model.to(device)
    for epoch in range(n_epochs):
        for i, (images, labels) in enumerate(data_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            mean_loss += loss.item()

            # Backward & optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % print_epochs == 0:
                if (i+1) % int(n_total_steps / print_steps) == 0:
                    print(f'Epoch {epoch + 1} / {n_epochs} | Step {i+1} / {n_total_steps} | '
                          f'Loss: {(mean_loss/2000):.{loss_acc}f}')
                    mean_loss = 0

    print('--- Training Finished ---')


def load_model(model, data_loader, classification, device=torch.device('cpu'), save=None,
               class_results=True, show_wrongs=False, n_wrongs=5):
    """
    Load an accuracy test for a model and it's classes
    :param model: Model on which the test will be run
    :param data_loader: DataLoader you want to use for testing
    :param classification: List of classes
    :param device: Torch device on which the test will run (processor by default)
    :param save: Load model from a save file (None by default)
    :param class_results: Shows individual class accuracy stats (True by default)
    :param show_wrongs: Shows wrong prediction and the labels (False by default)
    :param n_wrongs: Number of wrong examples which will be shown (5 by default)
    :return: Accuracy of the model on given dataset
    """

    # Load model if necessary
    if save is not None and device == torch.device('cuda'):
        model.load_state_dict(torch.load(save, map_location='cuda'))
    if save is not None and device == torch.device('mps'):
        model.load_state_dict(torch.load(save, map_location='mps'))
    if save is not None and device == torch.device('cpu'):
        model.load_state_dict(torch.load(save, map_location='cpu'))
    model.to(device)

    # Start Testing
    wrongs = list()
    n_classes = len(classification)
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        n_class_correct = [0 for _ in range(n_classes)]
        n_class_samples = [0 for _ in range(n_classes)]
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            # max returns (value ,index)
            _, predicted = torch.max(outputs, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()

            for i in range(len(labels)):
                label = labels[i]
                pred = predicted[i]
                if label == pred:
                    n_class_correct[label] += 1
                elif show_wrongs and n_wrongs > len(wrongs) and str(images[i][0][0].numpy()) not in wrongs:
                    wrongs.append(str(images[i][0][0].numpy()))
                    print(f'\nClass: {classification[label]} | Predicted: {classification[pred]}')
                    visual.imshow(images[i])
                n_class_samples[label] += 1

        total_accuracy = 100.0 * n_correct / n_samples
        print(f'Accuracy of the Model: {total_accuracy:.2f} %')

        if class_results:
            for i in range(n_classes):
                accuracy = 100.0 * n_class_correct[i] / n_class_samples[i]
                print(f'Accuracy of {classification[i]}: {accuracy:.2f} %')
    return total_accuracy


def save(model, save_path):
    """
    Save the parameters of your model to a file
    :param model: The model you want to save
    :param save_path: Where do you want to save the file
    :return: None
    """
    torch.save(model.state_dict(), save_path)


# TODO: Take care of padding so you can input IMG_SIZE as well
class ConvNet(nn.Module):
    def __init__(self, colour_size, n_categories):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(colour_size, 6, 4)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 4)
        self.fc1 = nn.Linear(16 * 19 * 19, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_categories)

    def forward(self, x):
        x = self.pool(torch.nn.functional.relu(self.conv1(x)))
        x = self.pool(torch.nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 16 * 19 * 19)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x
