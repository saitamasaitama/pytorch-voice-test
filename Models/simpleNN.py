import torch as tf
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random



class SumNet(nn.Module):

    def __init__(self):
        super(SumNet,self).__init__()
        self.weight=[1,1]
        self.L1=nn.Linear(2,1,True)


    def forward(self,x):
        x=self.L1(x)

        return x


if __name__ == "__main__" :

    
    device = tf.device("cpu")
    model=SumNet().to(device)
    #tf.manual_seed(1)
    optimizer = optim.Adadelta(model.parameters(), lr=1.0e-1)
    loss_func = nn.MSELoss()
    model.train()

    for i in range(10000):
        q=[]
        answer=[]

        #学習用のバッチデータを作成
        for j in range(125):
            A=random.randint(1,999)
            B=random.randint(1,999)
            C=A+B

            q.append([A,B])
            answer.append([C])

        data=tf.Tensor(q)
        data.to(device)

        label=tf.Tensor(answer)
        label.to(device)

        optimizer.zero_grad()
        output=model(data)

        loss=loss_func(output,label)

        # 必須
        loss.backward()
        optimizer.step()



    model.eval()
    A=random.randint(1000,9999)
    B=random.randint(1000,9999)

    o=model(tf.Tensor([A,B]))
    print(f"{A} + {B} ={A+B} ANSWER={o[0]} ")
    print(o)

    #保存
    tf.save(model.state_dict(),"tashizan.pt")

    #optimizer = optim.Adadelta(model.parameters(), lr=1.0e-5)

