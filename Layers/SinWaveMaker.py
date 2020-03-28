print(1)
import torch as tf
print(2)
import torch.nn as nn
print(3)
import torch.optim as optim
print(4)
import math
print(5)
import struct
print(6)
import os
print(7)

DIR=os.path.dirname(__file__)
sampleRate = 44100
"""
  受け取った波の音量の最大値/最小値を1.0にする。
  ・音の情報量減少＝なし
  ※入力は16bit intであること
"""
class SinLayer(nn.Module):
    samplingBit=16
    def __init__(self, f:float, s:float):
      super(SinLayer, self).__init__()
      self.f = f
      self.s = s

    def forward(self)->tf.Tensor:
        out = tf.sin(
          2 * math.pi * tf.arange(int(self.s * sampleRate)) * self.f
        )
        return out
        
        #スケール取得


layer=SinLayer(1000, 2)


if __name__ == "__main__":
    '''
    voice=wave.open( f"{DIR}/../Data/cat-cry3.wav","rb")

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
    serialized=((layer()).int())
    count = len(serialized)
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
    writer=wave.open(f"{DIR}/../Data/sin.wav","wb")

    writer.setnchannels(1)
    writer.setsampwidth(2)
    writer.setframerate(44100)


    #writer.writeframes(struct.pack(text,serialized.tolist()))
    text=f"{count}h"
    writer.writeframes(struct.pack(text,*shortes))