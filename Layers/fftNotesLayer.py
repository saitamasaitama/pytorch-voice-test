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

class fftNotesLayer(nn.Module):
    def __init__(self, sampleRate=44100):
      super(fftNotesLayer, self).__init__()
      self.fft = fftLayer.fftLayer(sampleRate)
    def forward(self, tensor:tf.tensor)->tf.Tensor:
      out = []
      maxF = 0
      Ftime = 0
      i = 0
      for i in range(0, len(tensor)-442, 4):
        x = self.fft(tensor[i: i+442], 0.01)
        x = tf.t(x[:221])
        x = tf.abs(x[0]) + tf.abs(x[1])
        tmp = int(tf.argmax(x))*100
        if abs(maxF - tmp) > maxF * 0.0594: # 1音階ずれたら
          out.append((maxF, (i - Ftime)*1000/44100))
          maxF = tmp
          Ftime = i
      out.append((maxF, (i - Ftime)*1000/44100))
      return out




if __name__ == "__main__":
    layer = fftNotesLayer()
    '''
    DIR=os.path.dirname(os.path.abspath(__file__))

    


    voice=wave.open( f"{DIR}/../Data/song_kei_recuerdo.wav","rb")
    
    print(f"""
    CHANNELS={voice.getnchannels()}
    SAMPLEWIDTH={voice.getsampwidth()*8}
    FRAMERATE={voice.getframerate()}
    FRAMES={voice.getnframes()}
    """)
    FRAME_COUNT=voice.getnframes()


    buff=voice.readframes(voice.getnframes())
    print(f"LEN={len(buff)}")
    
    unpacked=tf.Tensor(struct.unpack(f"{voice.getnframes()}h",buff))
    '''
    sampleRate = 44100
    time = 0.03
    y1 = tf.sin(
      2 * math.pi * tf.arange(int(time * sampleRate))/ sampleRate * 10000
    ) * (math.pow(2, 15) - 1)
    
    y2 = tf.cos(
      2 * math.pi * tf.arange(int(time * sampleRate))/ sampleRate * 1000
    ) * (math.pow(2, 15) - 1)
    y = tf.cat([y1 + y2])
    #y = y1
    out=layer(y)
    print(out)
    ''''
    z = tf.t(z)
    plt.xscale("log")
    plt.xlim(50, 11000)
    plt.plot(tf.abs(z[0]) + tf.abs(z[1]))
    plt.show()
    '''