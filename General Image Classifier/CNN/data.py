# Import of the Libraries
import os
import csv
import pandas as pd
import random as rand
import skimage

# Torch libraries
import torch
from torch.utils.data import Dataset


def create_csv(file_path, csv_path=None, train_csv=None, test_csv=None,
               rewrite=False, split=False, test_ratio=0.2, mul=1, mul_test=False):
    """Create a csv file for your dataset"""

    translate = dict()                                  # Class : Number
    categories = os.listdir(file_path)                  # Load all categories

    # Create an indexing dictionary
    for idx in range(len(categories)):
        translate[idx] = categories[idx]

    # Create one csv file
    if not split:
        if csv_path is not None and not os.path.exists(csv_path) or rewrite:
            file = open(csv_path, 'w', newline='')      # Create the csv file
            writer = csv.writer(file)                   # Create a writer for csv file
            writer.writerow(('NaN', 'NaN'))             # DataLoader skips 1. row

            # Create a .csv file of all images & their class
            for idx in range(len(categories)):
                tmp_path = os.path.join(file_path, categories[idx])  # File path + Sub-File
                tmp_images = os.listdir(tmp_path)
                for img in tmp_images:
                    img_path = os.path.join(categories[idx], img)  # Sub-File + img
                    for _ in range(mul):                # Expand the dataset
                        writer.writerow((img_path, idx))
            file.close()                                # Close the file

    # Create train & test csv file
    if split:
        if (train_csv is not None and test_csv is not None) and \
                (not os.path.exists(train_csv) or not os.path.exists(test_csv) or rewrite):
            train_f = open(train_csv, 'w', newline='')  # Create the train csv file
            test_f = open(test_csv, 'w', newline='')    # Create the test csv file
            train_writer = csv.writer(train_f)          # Create a writer for train csv file
            test_writer = csv.writer(test_f)            # Create a writer for test csv file
            train_writer.writerow(('NaN', 'NaN'))       # DataLoader skips 1. row
            test_writer.writerow(('NaN', 'NaN'))        # DataLoader skips 1. row

            # Create the .csv files of all images & their class
            for idx in range(len(categories)):
                tmp_path = os.path.join(file_path, categories[idx])
                tmp_images = os.listdir(tmp_path)
                test_split = int(len(tmp_images) / 100 * (test_ratio * 100))
                rand_dir = sorted(rand.sample(range(0, len(tmp_images)), test_split))
                train_idx = 0
                for img in tmp_images:
                    img_path = os.path.join(categories[idx], img)
                    if train_idx in rand_dir and mul_test:
                        for _ in range(mul):
                            test_writer.writerow((img_path, idx))
                    elif train_idx in rand_dir:
                        test_writer.writerow((img_path, idx))
                    else:
                        for _ in range(mul):
                            train_writer.writerow((img_path, idx))
                    train_idx += 1
            train_f.close()                             # Close the file
            test_f.close()                              # Close the file

    return translate


class CustomDataset(Dataset):
    """Create a custom dataset"""
    def __init__(self, file_path, csv_path, transform=None):
        """Initialise the data file path, csv file path and add your transformation"""
        self.annotations = pd.read_csv(csv_path)
        self.file_path = file_path
        self.transform = transform

    def __len__(self):
        """Return length of your dataset"""
        return len(self.annotations)

    def __getitem__(self, index):
        """Get an transformed item from your dataset"""
        img_path = os.path.join(self.file_path, self.annotations.iloc[index, 0])
        image = skimage.io.imread(img_path)
        y_label = torch.tensor(self.annotations.iloc[index, 1])

        if self.transform:
            image = self.transform(image)

        return image, y_label
