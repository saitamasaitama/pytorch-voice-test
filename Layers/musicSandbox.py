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
import fftNotesLayer
import ArrayToDataLayer

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

class fftSandboxLayer(nn.Module):
    def __init__(self):
        super(fftSandboxLayer, self).__init__()
        self.fft = fftNotesLayer.fftNotesLayer()
        self.sins = ArrayToSinLayer.ArrayToSinLayer()
        self.save = tensorToWAV.tensorToWAVLayer()
        self.data = ArrayToDataLayer.ArrayToDataLayer()
    def forward(self, tensor):
        ff = self.fft(tensor)
        out = self.sins(ff)
        self.save(out, 'sandBox')
        #ff = self.data(ff)
        return ff


if __name__ == "__main__":
    DIR=os.path.dirname(os.path.abspath(__file__))

    #model = midiToTextToSinToWavLayer()
    #model.forward("../midi/kuma")
    '''
    time = 0.5
    sampleRate = 44100
    y1 = tf.sin(
      2 * math.pi * tf.arange(int(time * sampleRate))/ sampleRate * 10000
    ) * (math.pow(2, 15) - 1)
    
    y2 = tf.cos(
      2 * math.pi * tf.arange(int(time * sampleRate))/ sampleRate * 1000
    ) * (math.pow(2, 15) - 1)
    y = tf.cat([y1, y2])
    #y = y1
    '''
    name = 'km'
    voice=wave.open( f"{DIR}/../Data/{name}.wav","rb")
    
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
    layer = fftSandboxLayer()
    out=layer(unpacked)
    #print(out)