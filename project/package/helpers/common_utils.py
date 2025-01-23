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


def dataset_to_dataloader(dataset: Dataset, batch_size: int, shuffle: bool=False) -> DataLoader:
    dl = DataLoader(dataset, batch_size, shuffle=shuffle)
    return dl