import argparse
import os
import json
import logging
from logging.config import dictConfig

from package.helpers import common_utils
from package.jobs.ListedJobs import SUPPORTED_JOBS


logging_path = r"D:\git_repos\baseML\project\package\configs\logging_config.json"
assert os.path.exists(logging_path)

with open(logging_path, 'r') as file:
    dict_config = json.load(file)


dictConfig(dict_config)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="A simple command line argument parser.")
    parser.add_argument('-c', '--config', type=str, required=True, help='Provide run_config')
    
    args = parser.parse_args()
    config_path = args.config
    config = common_utils.read_yaml(config_path)
    job_type = config['job_type']
    job_params = config["parameters"]
    assert job_type in SUPPORTED_JOBS, f"{job_type} is not in {SUPPORTED_JOBS}"
    job_object = SUPPORTED_JOBS[job_type](job_params)
    job_object()
    

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(e)
        raise e

# pip install -ve .
# python package -c /d/git_repos/baseML/project/package/configs/run_configs/mnist_trainer.yaml