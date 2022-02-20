import numpy as np
import torch
from torch import nn

from datetime import datetime

from torch.utils.tensorboard import SummaryWriter

from architectures import CustomCNN
from dataset import get_data_loaders
import torch.optim as optim


def train(train_loader, model, optimizer, criterion, device):
    losses = []
    model.train()

    for i, data in enumerate(train_loader):
        inputs, labels = data[0].to(device), \
                         data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        losses.append(loss.item())

        if i % 10 == 0:
            print(f"{datetime.now()} - Training - Batch {i}: Loss {np.mean(losses)}")

    return np.mean(losses)


def evaluate(data_loader, model, criterion):
    losses = []
    model.eval()

    with torch.no_grad():
        for i, data in enumerate(data_loader):
            inputs, labels = data[0].to(device), \
                             data[1].to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            losses.append(loss.item())

    return np.mean(losses)


def log_epoch(writer, epoch, training_losses, validation_losses):
    try:
        writer.add_scalar(tag="Loss/Training", scalar_value=training_losses, global_step=epoch)
        writer.add_scalar(tag="Loss/Validation", scalar_value=validation_losses, global_step=epoch)
    except Exception as e:
        print("ERROR writing log...", e)


writer = SummaryWriter(log_dir="logs")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

train_loader, validation_loader, test_loader = get_data_loaders("dataset", 64)

model = CustomCNN() \
    .to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):  # loop over the dataset multiple times
    train_loss = train(train_loader, model, optimizer, criterion, device)
    validation_loss = evaluate(validation_loader, model, criterion)

    log_epoch(writer, epoch, train_loss, validation_loss)
