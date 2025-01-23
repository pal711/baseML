import torch.optim as optim


OPTIMIZERS = {
    'adam': optim.Adam,
    'adamw': optim.AdamW,
    'rmsprop': optim.RMSprop,
    'sgd': optim.SGD
}