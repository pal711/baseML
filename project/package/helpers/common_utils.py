import yaml
from torch.utils.data import DataLoader, Dataset


def read_yaml(yaml_path: str) -> dict:
    """Reads a yaml file and returns the content in a dictionary.

    Args:
        yaml_path (str): the yaml path

    Returns:
        dict: contents of yaml file
    """

    with open(yaml_path, 'r') as f:
        d_obj = yaml.safe_load(f)
    return d_obj


def dataset_to_dataloader(dataset: Dataset, batch_size: int=16, shuffle: bool=False) -> DataLoader:
    """converts a pytorch Dataset to DataLoader

    Args:
        dataset (Dataset): A pytorch Dataset
        batch_size (int, optional): Batch size of resulted Dataloader. Defaults to 16.
        shuffle (bool, optional): whether to suffle dataset records while loading in DataLoader. Defaults to False.

    Returns:
        DataLoader: the dataloader from Dataset
    """
    dl = DataLoader(dataset, batch_size, shuffle=shuffle)
    return dl