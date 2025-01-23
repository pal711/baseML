from package.models.ListedModels import MODELS
from package.trainers.ListedTrainers import TRAINERS
from package.data.ListedDataset import DATASETS
from package.losses.ListedLosses import LOSSES
from package.optimizers.ListedOptimizers import OPTIMIZERS


def create_model(model_config: dict):
    """Creates the model object for training or inference from the
    model_config dictionary and returns the model.

    Args:
        model_config (dict): model configuration

    Raises:
        RuntimeError: if the model is not listed in package.models.ListedModels

    Returns:
        _type_: a model object 
    """
    model_name = model_config['model_name']
    model_params = model_config.get('model_params')
    if not model_params:
        model_params = {}

    model_class = MODELS.get(model_name)
    if not model_class:
        raise RuntimeError(f"{model_name} not listed in package.models.ListedModels")
    
    model_obj = model_class(**model_params)
    return model_obj


def create_trainer(trainer_config: dict):
    """Returns a trainer object to train a model

    Args:
        trainer_config (dict): a dictionary containing 'trainer_name' and
        'trainer_params' (optional)

    Raises:
        RuntimeError: If the class mentioned in 'trainer_name' is not listed in
        package.trainers.ListedTrainers

    Returns:
        _type_: A trainer Object
    """
    trainer_name = trainer_config['trainer_name']
    trainer_params = trainer_config.get('trainer_params')
    if not trainer_params:
        trainer_params = {}

    trainer_cls = TRAINERS.get(trainer_name)
    if not trainer_cls:
        raise RuntimeError(f"{trainer_name} not listed in package.trainers.ListedTrainers")
    
    trainer_obj = trainer_cls(**trainer_params)
    return trainer_obj


def get_dataset(dataset_config):
    dataset_name = dataset_config['dataset_name']
    dataset_params = dataset_config.get('dataset_params')
    if not dataset_params:
        dataset_params = {}

    dataset_cls = DATASETS.get(dataset_name)
    if dataset_cls is None:
        raise RuntimeError(f"{dataset_name} not listed in package.data.ListedDataset")
    
    dataset_obj = dataset_cls(**dataset_params)
    return dataset_obj.get_dataset()


def get_loss_func(loss_config):
    loss_name = loss_config['loss_name']
    loss_params = loss_config.get('loss_params')
    if not loss_params:
        loss_params = {}

    loss_cls = LOSSES.get(loss_name)
    if loss_cls is None:
        raise RuntimeError(f"{loss_cls} not listed in package.losses.ListedLosses")
    
    loss_func = loss_cls(**loss_params)
    return loss_func


def get_optimizer(model_params, optimizer_config):
    optimizer_name = optimizer_config["optimizer_name"]
    optimizer_params = optimizer_config.get('optimizer_params')
    if not optimizer_params:
        optimizer_params = {}
    optimizer_params['params'] = model_params

    optimizer_cls = OPTIMIZERS.get(optimizer_name)
    if optimizer_cls is None:
        raise RuntimeError(f"{optimizer_cls} not listed in package.optimizers.ListedOptimizers")
    
    optimizer_obj = optimizer_cls(**optimizer_params)
    return optimizer_obj

