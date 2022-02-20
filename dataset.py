import json

import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets


def get_data_loaders(data_dir: str, batch_size: int):

    mean, std = read_saved_mean_and_std("dataset_info.json")

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(0.2),
        transforms.RandomApply(transforms=[
            transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))
        ], p=0.2),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    test_validation_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    data = datasets.ImageFolder(data_dir)
    num_data_samples = len(data.targets)
    train_idx, test_idx = train_test_split(np.arange(num_data_samples),
                                           test_size=0.3,
                                           stratify=data.targets)

    num_test_data_samples = len(test_idx)
    test_idx, val_idx = train_test_split(np.arange(num_test_data_samples),
                                         test_size=0.5,
                                         stratify=np.array(data.targets)[test_idx])

    train_data = Subset(datasets.ImageFolder(data_dir, transform=train_transform), train_idx)

    validation_data = Subset(datasets.ImageFolder(data_dir, transform=test_validation_transform), val_idx)
    test_data = Subset(datasets.ImageFolder(data_dir, transform=test_validation_transform), test_idx)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    validation_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=4)

    return train_loader, validation_loader, test_loader


def read_saved_mean_and_std(filename: str) -> tuple:
    with open(filename, "r") as file:
        mean_std = json.load(file)

    return mean_std["Mean"], mean_std["Std"]