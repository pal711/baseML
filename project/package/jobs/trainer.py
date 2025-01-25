import os
import logging
import torch
from package.jobs.base_job import JobBase
from package.helpers import module_utils
from package.helpers.common_utils import dataset_to_dataloader

logger = logging.getLogger(__name__)


class Trainer(JobBase):
    """Trainer class takes care of training of a model.
    For If your project uses a tool like Sphinx for generating documentation, you can provide long-form documentation in reStructuredText or Markdown files. Then you can use the autodoc extension of Sphinx to extract docstrings from your code automatically
    """
    def __init__(
            self,
            config: dict,
    ):
        mandatory_fields = [
            "dataset", "model"
        ]
        optional_fields = [
            "loss", "batch_size", "epochs", "optimizer", "lr",
            "load_from_checkpoint", "load_checkpoint_path",
            "save_to_checkpoint", "save_checkpoint_dir",
            "save_frequency"
        ]

        super(Trainer, self).__init__(config, mandatory_fields, optional_fields)
        
        # ----------- Handling Mandatory Fields ----------
        dataset_config = self.config['dataset']
        train_dataset, val_dataset, test_dataset = module_utils.get_dataset(dataset_config)
        self.cnt_train_datapoints = len(train_dataset)
        self.cnt_val_datapoints = len(val_dataset)
        self.cnt_test_datapoints = len(test_dataset)

        model_config = self.config['model']
        self.model = module_utils.create_model(model_config)

        # ----------- Handling Optional Fields -----------

        default_loss_config = {
            "loss_name": "CrossEntropyLoss"
        }
        loss_config = self.config.get("loss", default_loss_config)
        self.loss_func = module_utils.get_loss_func(loss_config)

        default_batch_size = 16
        self.batch_size = self.config.get("batch_size", default_batch_size)

        default_epochs = 5
        self.total_epochs = self.config.get("epochs", default_epochs)

        # lr can be mentioned outside or inside optimizer
        default_lr = self.config.get("lr", 0.001)

        default_optimizer_config = {
            "optimizer_name": "adam",
            "optimizer_params": {
                "lr": default_lr
            }
        }
        if "optimizer" in self.config:
            optimizer_config = self.config["optimizer"]
            optimizer_params = optimizer_config.get("optimizer_params", {})
            if "lr" not in optimizer_params:
                optimizer_params["lr"] = default_lr
                optimizer_config["optimizer_params"] = optimizer_params
        else:
            optimizer_config = default_optimizer_config
        
        self.optimizer = module_utils.get_optimizer(
            self.model.parameters(), 
            optimizer_config
            )

        def_load_from_checkpoint = False
        load_from_checkpoint = self.config.get("load_from_checkpoint", def_load_from_checkpoint)

        def_load_checkpoint_path = ""
        load_checkpoint_path = self.config.get("load_checkpoint_path", def_load_checkpoint_path)

        default_save_to_checkpoint = True
        self.save_to_checkpoint = self.config.get(
            "save_to_checkpoint",
            default_save_to_checkpoint
        )

        default_save_checkpoint_dir = os.path.join(os.getcwd(), "saved_model")
        self.save_checkpoint_dir = self.config.get(
            "save_checkpoint_dir",
            default_save_checkpoint_dir
        )

        default_save_frequency = self.total_epochs  # default is saving after final epoch
        save_frequency = self.config.get("save_frequency", default_save_frequency)

    
        # --------------
        self.trained_epochs=0
        # modify the state if there is a load from checkpoint
        if load_from_checkpoint:
            assert os.path.exists(load_checkpoint_path), f"load_checkpoint_path: {load_checkpoint_path} does not exists"
            self.load_checkpoint_path = load_checkpoint_path
            chk_point = self.__load_from_checkpoint()
            self.model.load_state_dict(chk_point['model_state_dict'])
            self.optimizer.load_state_dict(chk_point['optimizer_state_dict'])
            self.trained_epochs = chk_point["epoch"]

        if self.save_to_checkpoint:
            os.makedirs(self.save_checkpoint_dir, exist_ok=True)
            self.save_frequency= save_frequency
            assert isinstance(self.save_frequency, int) and self.save_frequency > 0, f"save_frequency value shold be a positive integer"
            self.save_frequency = min(self.save_frequency, self.total_epochs)

        self.train_dataloader = dataset_to_dataloader(train_dataset, self.batch_size, shuffle=True)
        if val_dataset:
            self.val_dataloader = dataset_to_dataloader(val_dataset, self.batch_size)
        else:
            self.val_dataloader = None

        if train_dataset:
            self.test_dataloader = dataset_to_dataloader(test_dataset, self.batch_size)
        else:
            self.test_dataloader = None


    def __save_to_checkpoint(self, epoch, save_filepath):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': epoch
        }
        
        # Save the checkpoint
        torch.save(checkpoint, save_filepath)
        

    def __load_from_checkpoint(self):
        checkpoint = torch.load(self.load_checkpoint_path, weights_only=True)
        return checkpoint


    def __call__(self):
        for ep in range(self.trained_epochs + 1, self.total_epochs + 1):
            self.model.train()
            train_epoch_loss = 0.0
            batch_cnt = 0
            for data, labels in self.train_dataloader:
                batch_cnt += 1
                self.optimizer.zero_grad()
                outputs = self.model(data)
                loss = self.loss_func(outputs, labels)
                loss.backward()
                self.optimizer.step()
                train_epoch_loss += loss.item()
            train_batch_loss = round(train_epoch_loss / batch_cnt, 2)
            train_rec_loss = round(train_epoch_loss / self.cnt_train_datapoints, 6)
            logger.info(f"Epoch {ep}: train_loss: {train_rec_loss}")

            if self.val_dataloader:
                self.model.eval()
                val_epoch_loss = 0.0
                batch_cnt = 0
                for data, labels in self.val_dataloader:
                    batch_cnt += 1
                    self.optimizer.zero_grad()
                    outputs = self.model(data)
                    loss = self.loss_func(outputs, labels)
                    val_epoch_loss += loss.item()
                val_batch_loss = round(val_epoch_loss / batch_cnt, 2)
                val_rec_loss = round(val_epoch_loss / self.cnt_val_datapoints, 6)
                logger.info(f"Epoch {ep}: val_loss: {val_rec_loss}")

            if self.save_to_checkpoint:
                if ep % self.save_frequency == 0:
                    model_class = type(self.model).__name__
                    check_point_name = f"{model_class}_ep{ep}_train{train_rec_loss}"
                    if self.val_dataloader:
                        check_point_name += f"_val{val_rec_loss}"
                    check_point_name += ".pth"

                    chk_point_path = os.path.join(self.save_checkpoint_dir, check_point_name)
                    self.__save_to_checkpoint(ep, chk_point_path)
                    logger.info(f"Model saved at: {chk_point_path}")

        return self.model
    