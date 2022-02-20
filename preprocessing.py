import glob
import os
import numpy as np
import json

from PIL import Image


def calculate_and_save_mean_and_std(data_dir: str):
    wildcard_path = os.path.join(data_dir, '*/*')
    data_files = glob.glob(wildcard_path, recursive=True)

    images = []
    for file in data_files:
        im = Image.open(file)
        images.append(np.asarray(im))

    images = np.asarray(images)

    std = list(images.std(axis=(0, 1, 2)) / 256)
    mean = list(images.mean(axis=(0, 1, 2)) / 256)

    print("Dataset Mean & Std")
    print(f"Mean: {mean}")
    print(f"Std:  {std}\n")

    with open("dataset_info.json", "w") as outfile:
        json.dump({
            "Mean": mean,
            "Std": std
        }, fp=outfile, indent=4)

calculate_and_save_mean_and_std("dataset")
