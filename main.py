import os

import numpy as np
import torch
from torch import nn

from datetime import datetime

from torch.utils.tensorboard import SummaryWriter

from architectures import resnet_model
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

        losses.append(loss.item())

        if i % 10 == 0:
            print(f"{datetime.now()} - Training - Batch {i}: Loss {np.mean(losses):.2f}")

    return np.mean(losses)


def evaluate(data_loader, model, criterion):
    losses = []
    corrects = 0

    model.eval()

    with torch.no_grad():
        for i, data in enumerate(data_loader):
            inputs, labels = data[0].to(device), \
                             data[1].to(device)

            outputs = model(inputs)

            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            losses.append(loss.item())
            corrects += torch.sum(preds == labels.data)


    return np.mean(losses), corrects / len(data_loader.dataset)


def save_model(model):
    torch.save(model, f"runs/{run_id}.pt")

def load_best_model():
    return torch.load(f"runs/{run_id}.pt")


def log_epoch(writer, epoch, training_losses, validation_losses, validation_acc):
    try:

        writer.add_scalar(tag="Loss/Training", scalar_value=training_losses, global_step=epoch)
        writer.add_scalar(tag="Loss/Validation", scalar_value=validation_losses, global_step=epoch)
        writer.add_scalar(tag="Acc/Validation", scalar_value=validation_acc, global_step=epoch)
        writer.flush()
    except Exception as e:
        print("ERROR writing log...", e)


def epoch_summary(train_loss, validation_loss, validation_acc, epoch, saved):
    print(
        f"Epoch: {epoch} Train Loss: {train_loss:.2f} - Validation Loss: {validation_loss:.2f} Acc: {validation_acc:.2f} {'(Saved)' if saved else ''}\n")

def summary(test_loss, test_acc):
    print(f"\nTest Loss: {test_loss:.2f} Acc: {test_acc:.2f}")

def setup_writer(comment):
    writer = SummaryWriter(log_dir=f"logs/{run_id}__{comment}",)
    return writer

run_id = datetime.now().strftime("%d-%b-%Y__%H-%M-%S")

if not os.path.isdir("runs"):
    os.mkdir("runs")


batch_size = 64
epochs = 6
learning_rate = 0.01
momentum = 0.9

log_comment = f"resnet_18_pretrained--BS{batch_size}_E{epochs}_LR{learning_rate}_MOM{momentum}"

writer = setup_writer(log_comment)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


train_loader, validation_loader, test_loader = get_data_loaders("dataset", batch_size)

model = resnet_model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

best_validation_loss = float("inf")

for epoch in range(epochs):
    train_loss = train(train_loader, model, optimizer, criterion, device)
    validation_loss, validation_acc = evaluate(validation_loader, model, criterion)

    log_epoch(writer, epoch, train_loss, validation_loss, validation_acc)

    saved = False
    if best_validation_loss > validation_loss:
        best_validation_loss = validation_loss
        save_model(model)
        saved = True

    epoch_summary(train_loss, validation_loss, validation_acc, epoch, saved)

best_model = load_best_model()

test_loss, test_acc = evaluate(test_loader, best_model, criterion)

summary(test_loss, test_acc)