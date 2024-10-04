import yaml
from package.models.ListedModels import MODELS
from package.trainers.ListedTrainers import TRAINERS
from package.data.ListedDataset import DATASETS
from package.losses.ListedLosses import LOSSES

def read_yaml(yaml_path):
    with open(yaml_path, 'r') as f:
        d_obj = yaml.safe_load(f)
    return d_obj

def create_model(model_config):
    model_name = model_config['model_name']
    model_params = model_config.get('model_params')
    if not model_params:
        model_params = {}

    model_class = MODELS.get(model_name)
    if not model_class:
        raise RuntimeError(f"{model_name} not listed in package.models.ListedModels")
    
    model_obj = model_class(**model_params)
    return model_obj

def create_trainer(trainer_config):
    trainer_name = trainer_config['trainer_name']
    trainer_params = trainer_config.get('trainer_params')
    if not trainer_params:
        trainer_params = {}

    trainer_cls = TRAINERS.get(trainer_name)
    if not trainer_cls:
        raise RuntimeError(f"{trainer_cls} not listed in package.trainers.ListedTrainers")
    
    trainer_obj = trainer_cls(**trainer_params)
    return trainer_obj

def get_dataset(dataset_config):
    dataset_name = dataset_config['dataset_name']
    dataset_params = dataset_config.get('dataset_params')
    if not dataset_params:
        dataset_params = {}

    dataset_cls = DATASETS.get(dataset_name)
    dataset_obj = dataset_cls(**dataset_params)
    return dataset_obj.get_dataset()

def get_loss_func(loss_config):
    loss_name = loss_config['loss_name']
    loss_params = loss_config.get('loss_params')
    if not loss_params:
        loss_params = {}

    loss_func = LOSSES[loss_name]
    return loss_func
