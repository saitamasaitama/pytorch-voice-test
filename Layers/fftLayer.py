import torch as tf
import torch.nn as nn
import torch.optim as optim
import math
import struct
import wave
import os
import AnySinLayer
import matplotlib.pyplot as plt

class fftLayer(nn.Module):
    def __init__(self, sampleRate=44100):
      super(fftLayer, self).__init__()
      self.sampleRate = 44100
    def forward(self, tensor:tf.tensor, len=1)->tf.Tensor:

        x = tf.arange(0, 1 * len, 1/self.sampleRate, dtype=tf.float32)
        
        x = tf.stack([x, tensor[:int(self.sampleRate * len)]], 1)
        out = tf.fft(x, 1)

        return out




if __name__ == "__main__":
    layer = fftLayer()
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
    time = 0.1
    y1 = tf.sin(
      2 * math.pi * tf.arange(int(time * sampleRate))/ sampleRate * 10000
    ) * (math.pow(2, 15) - 1)
    
    y2 = tf.cos(
      2 * math.pi * tf.arange(int(time * sampleRate))/ sampleRate * 100
    ) * (math.pow(2, 15) - 1)
    
    out = tf.cat([y1 + y2])
    #out = y1
    for i in range(5):
      z=layer(out[i*100: (i+10)*100], 0.01)
      plt.plot(z)
      plt.show()
    ''''
    z = tf.t(z)
    plt.xscale("log")
    plt.xlim(50, 11000)
    plt.plot(tf.abs(z[0]) + tf.abs(z[1]))
    plt.show()
    '''