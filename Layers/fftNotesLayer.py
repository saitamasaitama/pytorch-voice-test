import torch as tf
import torch.nn as nn
import torch.optim as optim
import math
import struct
import wave
import os
import AnySinLayer
import matplotlib.pyplot as plt
import fftLayer

class catLayer(nn.Module):
    def __init__(self, sampleRate=44100):
      super(catLayer, self).__init__()
      self.conv1 = nn.Conv1d(1, 32, 6)
      self.conv2 = nn.Conv1d(32, 64, 6)
      self.fc1 = nn.Linear()
    def forward(self, tensor:tf.tensor)->tf.Tensor:
      



if __name__ == "__main__":
    pass