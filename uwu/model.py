from torch import nn
import torch
from .config import Config


class AudioGenerator(nn.Module):
    def __init__(self, config: Config):
        self.model = nn.Sequential()

    def forward(self, x):
        pass
