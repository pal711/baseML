import argparse
from package.helpers import utils


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
   

if __name__ == "__main__":
    main()
