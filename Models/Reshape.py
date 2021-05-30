import torch.nn as nn

class Reshape(nn.Module):
    def __init__(self, dims):
        super(Reshape, self).__init__()
        self.shape = tuple(dims)

    def forward(self, x):
        return x.view(self.shape)