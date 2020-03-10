import wave
from typing import Callable
import math

#
def then(a:Callable[[],bool],b:Callable[[],None]):
    return b() if a() else None

def isPrime(n:int)->bool:
    if n < 2 :
        return False;
    if( n % 2 == 0):
        return False
    #3から順に割り切れるか判断
    for f in range(3,n,2):
        if n % f == 0:
            return False
    return True


#サイン波フレームを出力
def sinHz(hz,framerate):
  #実質何frame分なのか
  frames=int(framerate / hz)
  print(f"SIN({hz}hz) FRAMES={frames}")
  #1hz分の長さ = -1PI
  length =  hz / framerate 
  for i in range(frames):
      sin=math.sin((i/frames) * (-2 * math.pi))
      #print(f"S[{i}]{sin}")


hz=1000
totalsec=1.0
framerate=44100
totalframe=framerate*totalsec



#for i in range((int)(hz*totalsec)):
#  print(f"FRAME:{i}")
#  sinHz(hz,framerate)
#  #当hzに対しての波を書き込み

#sinHz(440,44100)
stackbuff=list()
AA=12345

#素数だったら実行するやつ
for i in range(50,4000):
    then(lambda :isPrime(i),lambda:{
        #print(f"PRIME= {i}"),
        #AA=AA+1
        stackbuff.append(sinHz(i,framerate))
    })
out = wave.open("100hz.wav","wb")
out.setnchannels(1)
out.setsampwidth(2)
out.setframerate(44100)
out.close()
