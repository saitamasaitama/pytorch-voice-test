import torch as tf
import torch.nn as nn
import torch.optim as optim

#重み付きsinwave

class SinWaveLayer(nn.Module):
    def __init__(self,actor):
        super(SinWaveLayer, self).__init__()
        self.A = 1
        self.weight=nn.Parameter(
                tf.ones([3])
        )
    
    def forward(self,hz:float):
        return hz * 3

class TES(nn.Module):

    def __init__(self):
        super(TES, self).__init__()
        self.L1=SinWaveLayer("aa")

    def forward(self,x):
        print("TES FORWARD")
        x=self.L1(x)
        return x

if __name__ == "__main__" :
    print("ready train")

    device = tf.device("cpu")
    model=TES().to(device)
    parameters=model.parameters()

    #for param in parameters:
    #    print(param)

    optimizer = optim.Adadelta(model.parameters(), lr=1.0e-1)
    loss_func = nn.MSELoss()
    
    print("OK train start")
    model.train()
    
    for i in range(10000):
        data=tf.Tensor([1,2])
        data.to(device)

        label=tf.Tensor([3])
        label.to(device)

        optimizer.zero_grad()

        output=model(22)
        print("OUT")

        loss=loss_func(tf.Tensor([output,output,output]),tf.ones([3]))

        # 必須
        loss.backward()
        optimizer.step()
    
    print("learn complete")

    model.eval()

    print(model(34))
