import os
from torchvision import datasets
import torchvision.transforms as transforms
from package.data.transforms.common_transforms import FlatTensor


class MNISTData():
    def __init__(self, storage_dir):
        os.makedirs(storage_dir, exist_ok=True)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            FlatTensor(0)
        ])

        self.train_dataset = datasets.MNIST(
            root=storage_dir, 
            train=True, 
            download=True, 
            transform=transform
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
        return self.train_dataset, self.test_dataset
