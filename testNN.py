import torch
import torch.nn as nn
import torch.optim as optim

#テスト用レイヤ
class MyLayer(nn.Module):
    def __init__(self, input, output):
    def __init__(self):
        super(MyLayer, self).__init__()
        self.input = input
        self.output = output
        self.weight = nn.Parameter(
                torch.randn(input, output)
        )

    def forward(self, input:torch.Tensor):
        return torch.ones([1,1])
#        return torch.matmul(input, self.weight).mul(2)

#最終的なSpeakerレイヤ
#16bit 
class SpeakerLayer(nn.Module):
    def forward(self,input:torch.Tensor):
        flat=torch.flatten(input,0)
        print(flat)
        #平べったくする
        return flat.int().clamp(min=0,max=1<<16)

#Loss関数
class TripletMarginLoss(nn.Module):
    def __init__(self, margin):
        super(TripletMarginLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        dist = torch.sum(
            torch.pow((anchor - positive), 2) -
            torch.pow((anchor - negative), 2),
            dim=1) + self.margin
        dist_hinge = torch.clamp(dist, min=0.0)  # max(dist, 0.0)と等価
        loss = torch.mean(dist_hinge)

        return loss

def origin_loss(input:nn.Module,target:torch.Tensor)->float:
    return 2.1

#自作Loss ここまで

#自作Model
class M(nn.Module):

    def __init__(self):
        super(M, self).__init__()
        #self.conv1 = nn.Conv2d(1, 32, 3, 1)
        #self.L2=nn.Linear(1,1)
        #ここで宣言しないと学習されない
        self.L1=MyLayer()
        self.L2=SpeakerLayer()
        self.M=MyLayer()
        print(f"conv1={self.L1}")

    def forward(self, a: torch.Tensor):
        a=self.L1(a)
        print(f"L1={a}" )
        a=self.L2(a)
        print(f"L2={a}" )
        return a.mul(2)

#ここから

print( M() ) 
