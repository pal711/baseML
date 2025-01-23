import os
import logging
import torch
from torch.utils.data import Dataset
from package.helpers.common_utils import dataset_to_dataloader

logger = logging.getLogger(__name__)


class UltimateTrainer:
    def __init__(
            self,
            model,
            loss_func,
            train_dataset: Dataset,
            val_dataset: Dataset,
            test_dataset: Dataset,
            optimizer,
            batch_size: int,
            epochs: int,
            load_from_checkpoint: bool,
            load_checkpoint_path: str,
            save_to_checkpoint: bool,
            save_checkpoint_dir: str,
            save_frequency: int,
            **kwargs
    ):
        self.model = model
        self.optimizer = optimizer
        self.trained_epochs=0
        self.total_epochs = epochs
        self.loss_func = loss_func
        self.batch_size= batch_size

        # modify the state if there is a load from checkpoint
        if load_from_checkpoint:
            assert os.path.exists(load_checkpoint_path), f"load_checkpoint_path: {load_checkpoint_path} does not exists"
            self.load_checkpoint_path = load_checkpoint_path
            chk_point = self.__load_from_checkpoint()
            self.model.load_state_dict(chk_point['model_state_dict'])
            self.optimizer.load_state_dict(chk_point['optimizer_state_dict'])
            self.trained_epochs = chk_point["epoch"]

        self.save_checkpoint = save_to_checkpoint
        self.save_checkpoint_dir = save_checkpoint_dir
        if save_to_checkpoint:
            os.makedirs(save_checkpoint_dir, exist_ok=True)
            self.save_frequency= save_frequency
            assert isinstance(self.save_frequency, int) and self.save_frequency > 0, f"save_frequency value shold be a positive integer"

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
        checkpoint = torch.load(self.load_checkpoint_path)
        return checkpoint


    def __call__(self):
        for ep in range(self.trained_epochs + 1, self.total_epochs + 1):
            self.model.train()
            train_epoch_loss = 0.0
            for data, labels in self.train_dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(data)
                loss = self.loss_func(outputs, labels)
                loss.backward()
                self.optimizer.step()
                train_epoch_loss += loss.item()
            logger.info(f"Epoch {ep}: train_loss: {train_epoch_loss}")

            if self.val_dataloader:
                self.model.eval()
                val_epoch_loss = 0.0
                for data, labels in self.val_dataloader:
                    self.optimizer.zero_grad()
                    outputs = self.model(data)
                    loss = self.loss_func(outputs, labels)
                    val_epoch_loss += loss.item()
                logger.info(f"Epoch {ep}: val_loss: {val_epoch_loss}")

            if self.save_checkpoint:
                if ep == self.total_epochs or ep % self.save_frequency == 0:
                    model_class = type(self.model).__name__
                    check_point_name = f"{model_class}_ep{ep}_train{train_epoch_loss}"
                    if self.val_dataloader:
                        check_point_name += f"_val{val_epoch_loss}"
                    check_point_name += ".pth"

                    chk_point_path = os.path.join(self.save_checkpoint_dir, check_point_name)
                    self.__save_to_checkpoint(ep, chk_point_path)
                    logger.info(f"Model saved at: {chk_point_path}")

        return self.model
    