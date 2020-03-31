import torch as tf
import torch.nn as nn
import torch.optim as optim
import math
import struct
import wave
import os
import AnySinLayer
import tensorToWAV
import ArrayToSinLayer
import midiToTextLayer

class IntegrationLayer(nn.Module):
    def __init__(self):
        super(IntegrationLayer, self).__init__()
        self.sin = AnySinLayer.SinLayer()
        self.save = tensorToWAV.tensorToWAVLayer()
    def forward(self):
        self.save(self.sin(2200, 2), 'sandBox')
        return True

class midiToTextToSinToWavLayer(nn.Module):
    def __init__(self):
        super(midiToTextToSinToWavLayer, self).__init__()
        self.sins = ArrayToSinLayer.ArrayToSinLayer()
        self.midis = midiToTextLayer.midiToTextLayer()
        self.save = tensorToWAV.tensorToWAVLayer()
    def forward(self, name: str):
        out = self.midis(name)
        print(out)
        out = self.sins(out)
        self.save(out, 'midiTowav')
        return True

if __name__ == "__main__":
    model = midiToTextToSinToWavLayer()
    model.forward("../midi/kuma")