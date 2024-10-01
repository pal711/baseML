import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader


class SimpleTrainer:
    str_to_optimizer = {
        'adam': optim.Adam
    }
    
    def __init__(
            self, 
            epochs, 
            load_model_from_checkpoint=False,
            load_optimizer_from_checkpoint=False,
            check_point_path="",
            lr=0.001, 
            batch_size=16, 
            optimizer='adam', 
            save_checkpoint=True, 
            save_dir="", 
            save_frequency=10
            ):
        self.load_model_from_checkpoint = load_model_from_checkpoint
        self.load_optimizer_from_checkpoint = load_optimizer_from_checkpoint
        self.check_point_path = check_point_path
        assert isinstance(self.load_model_from_checkpoint, bool)
        assert isinstance(self.load_optimizer_from_checkpoint, bool)
        if self.load_model_from_checkpoint or self.load_optimizer_from_checkpoint:
            assert os.path.exists(self.check_point_path)

        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.optimizer = SimpleTrainer.str_to_optimizer[optimizer.lower()]
        assert isinstance(save_checkpoint, bool)
        self.save_checkpoint = save_checkpoint
        self.save_dir = save_dir
        self.save_frequency = save_frequency
        os.makedirs(self.save_dir, exist_ok=True)
        if self.save_checkpoint:
            assert isinstance(self.save_frequency, int)
            assert self.save_frequency > 0

    def __save_checkpoint(self, model, optimizer, epoch, filename):
        # Create a checkpoint dictionary
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'batch_size': self.batch_size
        }
        
        # Save the checkpoint
        torch.save(checkpoint, filename)
        print(f"Checkpoint saved to {filename}")

    def __load_model_checkpoint(self, model):
        checkpoint = torch.load(self.check_point_path)
        if self.load_model_from_checkpoint:
            model_state_dict = checkpoint['model_state_dict']
            model.load_state_dict(model_state_dict)

        return model
    
    def __load_opt_checkpoint(self, optimizer):
        checkpoint = torch.load(self.check_point_path)
        if self.load_optimizer_from_checkpoint:
            opt_state_dict = checkpoint['optimizer_state_dict']
            optimizer.load_state_dict(opt_state_dict)

        return optimizer
    
    def train(self, model, loss_criterion, train_dataset, val_dataset=None):
        if self.load_model_from_checkpoint:
            model = self.__load_model_checkpoint(model)
            
        optimizer_obj = self.optimizer(
            model.parameters(), 
            lr=self.lr
            )

        if self.load_optimizer_from_checkpoint:
            optimizer_obj = self.__load_opt_checkpoint(optimizer_obj)
        
        train_dataloader = DataLoader(
            dataset=train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True
            )
        
        if val_dataset:
            val_dataloader = DataLoader(
                dataset=val_dataset, 
                batch_size=self.batch_size, 
                shuffle=False
            )

        
        for epoch in range(1, self.epochs + 1):
            model.train()
            train_epoch_loss = 0.0
            for data, labels in train_dataloader:
                optimizer_obj.zero_grad()
                outputs = model(data)
                loss = loss_criterion(outputs, labels)
                loss.backward()
                optimizer_obj.step()
                train_epoch_loss += loss.item()
            print(f"Epoch {epoch}: train_loss: {train_epoch_loss}")

            if val_dataset:
                model.eval()
                val_epoch_loss = 0.0
                for data, labels in val_dataloader:
                    optimizer_obj.zero_grad()
                    outputs = model(data)
                    loss = loss_criterion(outputs, labels)
                    val_epoch_loss += loss.item()
                print(f"Epoch {epoch}: val_loss: {val_epoch_loss}")

            # put logging here
            print("------" * 4)

            if self.save_checkpoint:
                if epoch == self.epochs or epoch % self.save_frequency == 0:
                    model_class = type(model).__name__
                    check_point_name = f"{model_class}_ep{epoch}_train{train_epoch_loss}"
                    if val_dataset:
                        check_point_name += f"_val{val_epoch_loss}"
                    check_point_name += ".pth"

                    chk_point_path = os.path.join(self.save_dir, check_point_name)
                    self.__save_checkpoint(model, optimizer_obj, epoch, chk_point_path)

        return model
