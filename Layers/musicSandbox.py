import torch as tf
import torch.nn as nn
import torch.optim as optim
import math
import struct
import wave
import os
import AnySinLayer
import tensorToWAV

class IntegrationLayer(nn.Module):
    def __init__(self):
        super(IntegrationLayer, self).__init__()
        self.sin = AnySinLayer.SinLayer(2200, 2)
        self.save = tensorToWAV.tensorToWAVLayer()
    def forward(self):
        self.save(self.sin(), 'sandBox')
        return True

if __name__ == "__main__":
    model = IntegrationLayer()
    model.forward()