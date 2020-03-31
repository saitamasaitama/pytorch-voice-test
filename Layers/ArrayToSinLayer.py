import torch as tf
import torch.nn as nn
import torch.optim as optim
import math
import struct
import wave
import os
import AnySinLayer

class ArrayToSinLayer(nn.Module):
    def __init__(self):
      super(ArrayToSinLayer, self).__init__()
      self.sin = AnySinLayer.SinLayer()

    def forward(self, arr)->tf.Tensor:
        out = tf.tensor([], dtype=tf.float32)
        #print(arr)
        for f, ms in arr:
          #print("{}".format(self.sin(f, ms/1000)))
          out = tf.cat([out, self.sin(f, ms/1000)])
        print(out)
        return out
        
        #スケール取得




if __name__ == "__main__":
  pass