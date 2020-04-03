import torch as tf
import torch.nn as nn
import torch.optim as optim
import math
import struct
import wave
import os
import AnySinLayer
import matplotlib.pyplot as plt

class ArrayToDataLayer(nn.Module):
    def __init__(self):
      super(ArrayToDataLayer, self).__init__()
    def forward(self, arr):
      out = []
      grad = 0
      lastHz = arr[0][0]
      lastms = 0
      for Hz, ms in arr:
        if ms - lastms == 0:
          grad = 0
        else:
          grad = (Hz - lastHz)/(ms - lastms)
        out.append((Hz, ms, grad))
        lastHZ = Hz
        lastms = ms
      return out
      



if __name__ == "__main__":
    pass