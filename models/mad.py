import torch
from torch import nn
import numpy as np

class MAD(nn.Module):
    def __init__(
            self,
            p=0.5,
    ):
        super(MAD, self).__init__()
        self.p = p

    def forward(self, layers):
        with torch.no_grad():
            num_layers = len(layers)
            layer_to_drop = np.random.choice(num_layers, 1, p=self.p) 
            layers[layer_to_drop[0]].weight = nn.Parameter(torch.zeros_like(layers[layer_to_drop[0]]))

        return layers 