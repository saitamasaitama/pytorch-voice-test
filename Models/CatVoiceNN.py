import typing
import wave
import struct
import torch.nn as nn
import torch.nn.functional as F
import torch

class CatVoiceNet(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        x = (nn.Linear(1, 3))(x)
        print(x.grad)
        return x

    

