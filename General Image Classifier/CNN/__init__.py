__all__ = ['dataset', 'model', 'visual']

import torch

from .dataset import create_csv
from .dataset import get_normal
from .dataset import trans_normal
from .dataset import CustomDataset

from .model import train_model
from .model import load_model
from .model import save_model
from .model import ConvNet

from .visual import imshow
from .visual import data_distribution
from .visual import print_data_distribution
from .visual import data_distribution_table
from .visual import overview


def pick_device(device=None):
    """
    Choose the device you want to use:\n
    - 'cpu' PyTorch will use processor
    - 'cuda' PyTorch will use NVIDIA GPU if possible
    - 'mps' PyTorch will use Apple Silicon ARM (M1/M2) if possible
    Hint: By adding ':n' you can choose a certain device (e.g. 'cuda:0')
    :param device: Device name (if no input, automatic selection of an option)
    :return: torch.device('device')
    """
    if device is None:
        gpu = torch.device('cpu')
        if torch.cuda.is_available():
            gpu = torch.device('cuda')
        if torch.has_mps:
            gpu = torch.device('mps')
        print('Using the Processor') if gpu == torch.device('cpu') else print('Using the Graphics Card')
        return gpu
    return torch.device(device)
