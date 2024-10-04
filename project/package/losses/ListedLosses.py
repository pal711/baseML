import torch


LOSSES = {
    'L1Loss': torch.nn.L1Loss(),
    'MSELoss': torch.nn.MSELoss(),
    'CrossEntropyLoss': torch.nn.CrossEntropyLoss(),
    'CTCLoss': torch.nn.CTCLoss(),
    'NLLLoss': torch.nn.NLLLoss()
}
