import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomCNN(torch.nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=3, kernel_size=3)

        self.fc1 = nn.Linear(220*220*3, 512)
        self.fc2 = nn.Linear(512, 151)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)

        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x
