# Import of the Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Torch libraries
import torchvision
from torch.utils.data import DataLoader


def imshow(img):
    """
    Show a batch of images
    :param img: Image iteration from DataLoader
    :return: None
    """

    img = torchvision.utils.make_grid(img)
    img = img / 2 + 0.5  # un-normalize
    np_img = img.numpy()
    plt.imshow(np.transpose(np_img, (1, 2, 0)))
    plt.show()


def data_distribution(dataset):
    """
    Creates a dictionary of data distribution in the given dataset
    :param dataset: Dataset to analyse
    :return: dictionary ( class index : number of samples )
    """

    if dataset is not None:
        dct = dict()
        for item in dataset:
            try:
                dct[item[1].item()] += 1
            except KeyError:
                dct[item[1].item()] = 1
        return dct
    return None


def print_data_distribution(dataset, translate):
    """
    Prints the data distribution in the given dataset
    :param dataset: Dataset to analyse
    :param translate: Dict of class indexing (Hint: usually an output of the create_csv function)
    :return: None
    """

    for key, value in data_distribution(dataset).items():
        print(f'{translate[key]}: {value}')


def data_distribution_table(train_data, test_data, translate):
    """
    Show the data distribution of dataset in a Pandas table
    :param train_data: Dataset used for training the model
    :param test_data: Dataset used for testing the model
    :param translate: ict of class indexing (Hint: usually an output of the create_csv function)
    :return: Pandas table
    """

    train_dct = data_distribution(train_data)
    test_dct = data_distribution(test_data)
    index = [str(num) for num in range(len(train_dct))]
    index.append('')
    index.append('T:')
    df = pd.DataFrame(np.random.randn(len(train_dct)+2, 4),
                      columns=['Class', 'Train Data', 'Test Data', 'Total'],
                      index=index)

    idx = 0
    for key, test_value in train_dct.items():
        df.iat[idx, 0] = translate[key]
        df.iat[idx, 1] = str(test_value)
        train_value = test_dct[key] if test_dct is not None else 0
        df.iat[idx, 2] = str(train_value)
        df.iat[idx, 3] = str(test_value + train_value)
        idx += 1

    for num in range(4):
        df.iat[idx, num] = ''
    idx += 1

    df.iat[idx, 0] = str(len(train_dct))
    for num in range(1, 4):
        addition = 0
        for i in range(len(train_dct)):
            addition += int(df.iat[i, num])
        df.iat[idx, num] = str(addition)

    return df


def overview(train_data, test_data, translate, images=None, batch_size=64):
    """
    Show a batch of images, the data distribution and Pandas table of given datasets
    :param train_data: Dataset used for training the model
    :param test_data: Dataset used for testing the model
    :param translate: Dict of class indexing (Hint: usually an output of the create_csv function)
    :param images: Image iteration from DataLoader (None by default, will be generated automatically)
    :param batch_size: Number of images to show in case images is None
    :return: Pandas table
    """

    if images is None:
        loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

        # Get some random training images
        dataiter = iter(loader)
        images, labels = next(dataiter)

    imshow(images)

    params = images.shape
    print(f'Batch size: {params[0]} | Colour size: {params[1]} | Image Size: {params[2]}*{params[3]} pixels')

    return data_distribution_table(train_data, test_data, translate)
