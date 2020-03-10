import typing
import wave
import struct
import torch.nn as nn
import torch.nn.functional as F
import torch


class WaveBit:

    Begin: float = 0
    Width: float = 0
    V: float = 0.0
    H: float = 0.0

    def Y(self, now):
        return self.H + (self.V * self.X(now))

    def X(self, now):
        return now - self.Begin


def PlayWaveBits(datas: typing.List[WaveBit]) -> int:
    return 12345


# 22100 Hz -> 100Hz ～
# つまり1音に最大200Nの情報が含まれている。
# 221Hzで検索すれば
# 1音の長さはN

# 44100 * N から畳み込む
# 評価する際に 44100 * N にまず変換する

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        x = (nn.Linear(1, 3))(x)
        print(x.grad)

        return x

    def trainAndSave(self):
        self.train()
        opt=torch.optim.Adam([1,2], lr=0.0001)
        #opt=torch.optim.Adadelta(self.parameters(), lr=1e-10)
        output = model(data)
        loss = F.nll_loss(output, target)

        loss.backward()
        opt.step()

        torch.save(self.state_dict(), "file-name.model")
        print("Save Conmplate")


    def batchTrain(self) -> int:
        opt: torch.optimizer = torch.optim.Adadelta(self.parameters(), lr=1e-2)
        opt.zero_grad()
        pass

#自作optim
class MyReLU(torch.autograd.Function):

    #forwardの活性化関数とbackwardの計算のみ記述すれば良い
    def forward(self, input):
        #値の記憶
        self.save_for_backward(input)
        #ReLUの定義部分
        #x.clamp(min=0) <=> max(x, 0)
        return input.clamp(min=0)

    #backpropagationの記述
    #勾配情報を返せば良い
    def backward(self, grad_output):
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input

# 自作loss関数


class TripletMarginLoss(nn.Module):

    def __init__(self, margin):
        super(TripletMarginLoss, self).__init__()
        self.margin: float = margin

    def forward(self, anchor, positive, negative):
        dist = torch.sum(
                torch.pow((anchor - positive), 2) -
                torch.pow((anchor - negative), 2),
                dim=1) + self.margin
        dist_hinge = torch.clamp(dist, min=0.0)  # max(dist, 0.0)と等価
        loss = torch.mean(dist_hinge)
        return loss
# 自作loss関数　ここまで


w: wave.Wave_read = wave.open("./info-lady1-bottonwo1.wav", "rb")
print(f"""
CHANNEL={w.getnchannels()}
SAMPLE={w.getsampwidth()*8}bit
FRAME={w.getframerate()}
SEC={format(w.getnframes()/w.getframerate(),"4.2f")}
""")
w.readframes(4410)
for i in range(100):
    print(struct.unpack(">h", w.readframes(1)))
    # print(int.from_bytes(w.readframes(1),'little'))
print(w)

N: Model = Model()
N.eval()
result = N(torch.ones([1]))


N.trainAndSave()

print(nn)
print(result)
