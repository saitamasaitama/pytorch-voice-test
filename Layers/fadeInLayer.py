import torch as tf
import torch.nn as nn
import torch.optim as optim
import math
import wave
import struct
import os

DIR=os.path.dirname(os.path.abspath(__file__))

SampleRate = 44100

class FadeInLayer(nn.Module):
    samplingBit=16
    def __init__(self, seconds):
        super(FadeInLayer, self).__init__()
        self.seconds = seconds
    def forward(self, input:tf.Tensor)->tf.Tensor:
        l = min(len(input), int(self.seconds * SampleRate))
        filt = tf.ones([len(input)], dtype=tf.float32)
        filt[:l] = tf.arange(l, dtype=tf.float32)/l
        out = input.mul(filt).mul(filt)
        return out

        
layer=FadeInLayer(10)


if __name__ == "__main__":
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

    serialized=((layer(unpacked)).int())
    SERIAL_MAX=serialized[tf.argmax(serialized.abs())]
    print(f"SERIAL_MAX=:{SERIAL_MAX}")
    print(serialized.size())

    s= serialized[224].item()

    #
    # 
    # 保存する
    shortes=serialized.tolist()
    #print(struct.pack('42624h',*shortes) )
    #exit()
    writer=wave.open(f"{DIR}/../Data/test-Fadein.wav","wb")

    writer.setnchannels(1)
    writer.setsampwidth(2)
    writer.setframerate(SampleRate)


    #writer.writeframes(struct.pack(text,serialized.tolist()))
    text=f"{(FRAME_COUNT)}h"
    print(text)
    writer.writeframes(struct.pack(text,*shortes))