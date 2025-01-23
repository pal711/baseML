from package.helpers import module_utils
from package.trainers.ListedTrainers import TRAINERS
from package.trainers.UltimateTrainer import UltimateTrainer
from package.jobs.ListedJobs import SUPPORTED_JOBS


# SUPPORTED_JOBS = [
#     "train"
# ]


def get_job_object(job_config):
    job_type = job_config['job_type']
    assert job_type in SUPPORTED_JOBS, f"{job_type} is not in {SUPPORTED_JOBS}"

    if job_type == "train":
        trainer_config = job_config["trainer"]
        trainer_name = trainer_config['trainer_name']
        trainer_cls = TRAINERS.get(trainer_name)
        assert trainer_cls is not None, f"{trainer_name} not listed in package.trainers.ListedTrainers"
        trainer_params = trainer_config["trainer_params"]

        dataset_name = job_config['dataset']
        model_config = job_config['model']
        loss_config = job_config['loss']
        train_ds, val_ds, test_ds = module_utils.get_dataset(dataset_name)
        ml_model = module_utils.create_model(model_config)
        loss_func = module_utils.get_loss_func(loss_config)
        optim_config = job_config["optimizer"]
        optimizer_obj = module_utils.get_optimizer(ml_model.parameters(), optim_config)

        # --------- mandatory parameters for a trainer------------
        trainer_params["model"] = ml_model
        trainer_params["loss_func"] = loss_func
        trainer_params["train_dataset"] = train_ds
        trainer_params["val_dataset"] = val_ds
        trainer_params["test_dataset"] = test_ds
        trainer_params["optimizer"] = optimizer_obj
        # ---------------------------------------------------------
        
        if trainer_cls.__name__ == "UltimateTrainer":
            trainer_obj = UltimateTrainer(**trainer_params)

        # elif trainer_cls.__name__ == "SimpleTrainer":
        #     # logging the values
        #     logger.debug(f"dataset_name: {dataset_name}")
        #     logger.debug(f"model_config: {model_config}")
        #     logger.debug(f"loss_config: {loss_config}")
        #     logger.debug(f"trainer_config: {trainer_config}")

        #     train_ds, val_ds, test_ds = utils.get_dataset(dataset_name)
        #     ml_model = utils.create_model(model_config)
        #     loss_func = utils.get_loss_func(loss_config)
        #     trainer = utils.create_trainer(trainer_config)

        #     trained_model = trainer.run(
        #         ml_model,
        #         loss_func,
        #         train_ds,
        #         val_ds
        #         )

        
        return trainer_obj
