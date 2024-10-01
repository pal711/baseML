from functools import partial
import torch.nn.functional as F
import torch.nn as nn
from package.models.components.FeedForward import SimpleNN

class MNISTFFN(nn.Module):
    def __init__(self):
        super(MNISTFFN, self).__init__()
        self.layers = SimpleNN(
            784,
            10,
            2,
            [300, 100],
            [
                F.relu,
                F.relu,
                partial(F.sofrmax, dim=1)
            ]
        )

    def forward(self, x):
        x = self.layers(x)
        return x
        
    def save_model(self, model_path):
        pass

    def load_model(self, model_path):
        pass