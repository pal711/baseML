import torch


class FlatTensor:
    def __init__(self, start_dim, end_dim=-1):
        self.start_sim = start_dim
        self.end_dim = end_dim

    def __call__(self, img):
        img = torch.flatten(img, self.start_sim, self.end_dim)
        return img
