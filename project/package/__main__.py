import argparse
import os
import json
import logging
from logging.config import dictConfig

from package.helpers import utils
from package.helpers.test_logging import test123


logging_path = r"project\package\configs\logging_config.json"
assert os.path.exists(logging_path)

with open(logging_path, 'r') as file:
    dict_config = json.load(file)


dictConfig(dict_config)
logger = logging.getLogger(__name__)


supported_jobs = [
    'train'
]


def main():
    # Create the parser
    parser = argparse.ArgumentParser(description="A simple command line argument parser.")
    
    # Add arguments
    parser.add_argument('-c', '--config', type=str, required=True, help='Provide run_config')
    
    # Parse the arguments
    args = parser.parse_args()
    config_path = args.config
    config = utils.read_yaml(config_path)

    job_type = config['job_type']
    assert job_type in supported_jobs
    
    if job_type == "train":
        dataset_name = config['dataset']
        model_config = config['model']
        loss_config = config['loss']
        trainer_config = config['trainer']

        # logging the values
        logger.debug(f"dataset_name: {dataset_name}")
        logger.debug(f"model_config: {model_config}")
        logger.debug(f"loss_config: {loss_config}")
        logger.debug(f"trainer_config: {trainer_config}")

        train_ds, val_ds, test_ds = utils.get_dataset(dataset_name)
        ml_model = utils.create_model(model_config)
        loss_func = utils.get_loss_func(loss_config)
        trainer = utils.create_trainer(trainer_config)

        trained_model = trainer.run(
            ml_model,
            loss_func,
            train_ds,
            val_ds
            )
    
    elif job_type == "dataset_inference":
        raise NotImplementedError

    elif job_type == "webapp":
        raise NotImplementedError
    
    else:
        raise NotImplementedError

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(e)
        raise e
