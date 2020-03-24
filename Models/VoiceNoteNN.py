from typing import Tuple,List,Dict,NamedTuple
import wave
import struct
import torch.nn as nn
import torch.nn.functional as F
import torch as tf


class VoiceNote(NamedTuple):
    start:float
    height:float
    width:float
    voice:str

class VoiceText(NamedTuple):
    text:str



class WaveData():


    pass

#一つのセリフにあたるものを生成する
class VoiceScoreNet(nn.Module):

    def __init__(self):
        super(VoiceScoreNet, self).__init__()


    def forward(self,notes:List[VoiceNote])->WaveData:
        #結合する
        for note in notes:
            #音符をwaveに変換

            
            pass
        return WaveData()


#一つの音符にあたるものを生成する
class VoiceNoteNet(nn.Module):
    actor:str="Cat"

    def __init__(self,actor):                
        super(VoiceNoteNet, self).__init__()
        self.actor=actor
        self.main=nn.Sequential(
            nn.Linear(1,2)
        )
        self.L1=nn.Linear(1,2)
        self.B=123

    def forward(self, note:VoiceNote)->WaveData:
        """
            Tupleの中身
            X : 開始時間 →Noteでは必要ない
            Y : 音の高さ
            Z : 音データ
            W : 長さ
            [返り値は Tuple  ]
        """

        #音データはactor
        
        #x = (nn.Linear(1, 3))(x)
        #           print(note)
        #x=self.L1(tf.Tensor[1])
        #waveを返す
        return WaveData()
        #return tf.Tensor([1,2,3])



    

model=VoiceNoteNet("aaa")

#t=model(tf.Tensor([2]))
t=model(VoiceNote(
    start=0,
    height=0,
    width=1.0,
    voice="meow"
))

print(f"{model}")
print(f"{t}")
