# Import of the Libraries
import os
import csv
import pandas as pd
import random as rand
import skimage

# Torch libraries
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader


def create_csv(file_path, csv_path=None, train_csv=None, test_csv=None,
               rewrite=False, split=False, test_ratio=0.2, mul=1, mul_test=False):
    """
    Create a csv file for your dataset
    :param file_path: Sub-Folders location path (e.g. './dataset')
    :param csv_path: Where do you want to save the .csv file (e.g. './data/dataset.csv')
    :param train_csv: Where do you want to save the .csv train file (e.g. './data/train_dataset.csv')
    :param test_csv: Where do you want to save the .csv test file (e.g. './data/test_dataset.csv')
    :param rewrite: Do you want to rewrite the previous .csv file?
    :param split: Do you want to split the dataset into training and testing data? (False by default)
    :param test_ratio: If split is True, how much of the dataset should be used for testing? (0.2 = 20% by default)
    :param mul: Multiply the data is your dataset (No multiplication by default)
    :param mul_test: If split is True, do you want to multiply the test dataset as well? (False by default)
    :return: dictionary of indexing the classes (index : class)
    """

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


def get_normal(file_path, img_h, img_w, colour_size=1, grayscale=True, batch_size=5000):
    """
    Get the mean & std for you image dataset
    :param file_path: Sub-Folders location path (e.g. './dataset')
    :param img_h: Image height
    :param img_w: Image width
    :param colour_size: Size of the dataset colour (1 by default)
    :param grayscale: Grayscale transformation (True by default)
    :param batch_size: Loader batch_size (5000 by default)
    :return: [mean, std]
    """

    # Base Transformation
    if grayscale:
        transformer = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((img_h, img_w)),
            torchvision.transforms.Grayscale()
        ])
    else:
        transformer = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((img_h, img_w))
        ])

    # Create the .csv file
    csv_path = file_path + 'TMP.csv'
    create_csv(file_path, csv_path, rewrite=True)

    # Create the Datasets
    dataset = CustomDataset(file_path, csv_path, transform=transformer)
    os.remove(csv_path)

    # Get mean & std
    loader = DataLoader(dataset=dataset, batch_size=batch_size)
    n_pixels = len(dataset) * img_h * img_w

    total_sum = 0
    for image in loader:
        total_sum += image[0].sum()
    mean = (total_sum / n_pixels) / colour_size

    mse_sum = 0
    for image in loader:
        mse_sum += ((image[0] - mean).pow(2)).sum()
    std = torch.sqrt((mse_sum / n_pixels) / colour_size)

    return [mean, std]


def trans_normal(img_h, img_w, mean, std, grayscale=True):
    """
    Get a normalised transformation
    :param img_h: Image height
    :param img_w: Image width
    :param mean: Mean of the dataset images
    :param std: Std of the dataset images
    :param grayscale: Grayscale transformation (True by default)
    :return: normalised transformation
    """

    if grayscale:
        transformer = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((img_h, img_w)),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomVerticalFlip(),
            torchvision.transforms.Grayscale(),
            torchvision.transforms.Normalize(mean, std)
        ])
    else:
        transformer = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((img_h, img_w)),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomVerticalFlip(),
            torchvision.transforms.Normalize(mean, std)
        ])

    return transformer


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
