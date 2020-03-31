import torch as tf
import torch.nn as nn
import torch.optim as optim
import math
import struct
import wave
import os



class tensorToWAVLayer(nn.Module):

    def __init__(self, sampleRate=44100):
        super(tensorToWAVLayer, self).__init__()
        self.sampleRate = 44100
        self.DIR=os.path.dirname(os.path.abspath(__file__))

        
    def forward(self, tensor, name='test'):
        try:
            serialized=(tensor.int())
            count = len(serialized)
            shortes=serialized.tolist()
            writer=wave.open(f"{self.DIR}/../Data/{name}.wav","wb")
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(self.sampleRate)
            text=f"{count}h"
            writer.writeframes(struct.pack(text,*shortes))
            return True
        except:
            return False
        

if __name__ == "__main__":
    fileName = 'song_kei_recuerdo'
    DIR=os.path.dirname(os.path.abspath(__file__))
    voice=wave.open( f"{DIR}/../Data/{fileName}.wav","rb")

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
    model = tensorToWAVLayer()
    model.forward(unpacked, 'ok')