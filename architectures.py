import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


resnet_model = models.resnet18(pretrained=True)
num_ftrs = resnet_model.fc.in_features

resnet_model.fc = nn.Linear(num_ftrs, 151)


