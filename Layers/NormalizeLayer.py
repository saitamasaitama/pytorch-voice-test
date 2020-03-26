import torch as tf
import torch.nn as nn
import torch.optim as optim
import math
import wave
import struct
import os

DIR=os.path.dirname(os.path.abspath(__file__))


"""
  受け取った波の音量の最大値/最小値を1.0にする。
  ・音の情報量減少＝なし
  ※入力は16bit intであること
"""
class NormalizeLayer(nn.Module):
    samplingBit=16

    def forward(self, input:tf.Tensor)->tf.Tensor:
        max=tf.abs(input[tf.argmax(tf.abs(input))])
        print(max)
        mul=(math.pow(2,15))/max
        print(f"NORMALIZE={mul}" )
        return input.mul(mul)

        
        #スケール取得


layer=NormalizeLayer()


if __name__ == "__main__":
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

    unpacked=tf.Tensor(struct.unpack(f"{voice.getnframes()}h",buff[:175104]))

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
    writer=wave.open(f"{DIR}/../Data/cat-cry3-serialize.wav","wb")

    writer.setnchannels(1)
    writer.setsampwidth(2)
    writer.setframerate(44100)


    #writer.writeframes(struct.pack(text,serialized.tolist()))
    text=f"{FRAME_COUNT}h"
    writer.writeframes(struct.pack(text,*shortes))
  
  
