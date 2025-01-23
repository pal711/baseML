import argparse
import os
import json
import logging
from logging.config import dictConfig

from package.helpers import common_utils
from package.helpers.job_object import get_job_object


logging_path = r"D:\git_repos\baseML\project\package\configs\logging_config.json"
assert os.path.exists(logging_path)

with open(logging_path, 'r') as file:
    dict_config = json.load(file)


dictConfig(dict_config)
logger = logging.getLogger(__name__)


def main():
    # Create the parser
    parser = argparse.ArgumentParser(description="A simple command line argument parser.")
    
    # Add arguments
    parser.add_argument('-c', '--config', type=str, required=True, help='Provide run_config')
    
    # Parse the arguments
    args = parser.parse_args()
    config_path = args.config
    config = common_utils.read_yaml(config_path)

    job_object = get_job_object(config)
    job_object()
    

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(e)
        raise e

# pip install -ve .
# python package -c /d/git_repos/baseML/project/package/configs/run_configs/mninst_train_ult_trainer.yaml