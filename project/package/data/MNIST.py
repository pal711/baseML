import os
import torch
from torch.utils.data import random_split
from torchvision import datasets
import torchvision.transforms as transforms
from package.data.transforms.common_transforms import FlatTensor


class MNISTData():
    def __init__(self, storage_dir, train_split=0.8, val_split=0.2):
        os.makedirs(storage_dir, exist_ok=True)
        assert train_split + val_split == 1.0

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            FlatTensor(0)
        ])

        trainval_dataset = datasets.MNIST(
            root=storage_dir, 
            train=True, 
            download=True, 
            transform=transform
        )

        generator = torch.Generator().manual_seed(711)
        self.train_dataset, self.val_dataset = random_split(
            trainval_dataset,
            lengths=[train_split, val_split],
            generator=generator
            )
        
        self.test_dataset = datasets.MNIST(
            root=storage_dir, 
            train=False, 
            download=True, 
            transform=transform
        )

    def __transform_flat(self):
        pass

    def transform_data(self):
        pass

    def get_dataset(self):
        return self.train_dataset, self.val_dataset, self.test_dataset
