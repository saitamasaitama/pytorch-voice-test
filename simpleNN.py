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
     #   output = F.log_softmax(x, dim=1)
        return x


if __name__ == "__main__" :

    device = tf.device("cpu")
    model=SumNet().to(device)
    #tf.manual_seed(1)
    optimizer = optim.Adadelta(model.parameters(), lr=1.0e-1)
    model.train()


    for i in range(10000):
        q=[]
        answer=[]
        for j in range(125):
            A=random.randint(1,99)
            B=random.randint(1,99)
            C=A+B

            q.append([A,B])
            answer.append([C])

        data=tf.Tensor(q);
        data.to(device)
        target=tf.Tensor(answer)
        target.to(device)

        optimizer.zero_grad()
        output=model(data)
    #    loss= F.l1_loss(output, target)
        loss= F.mse_loss(output, target)

        # 必須
        loss.backward()
        #print(f"loss={loss}")
        optimizer.step()


    #保存



    model.eval()


    A=random.randint(10,90)
    B=random.randint(10,90)

    o=model(tf.Tensor([A,B]))
    print(f"{A} + {B} ={A+B} ANSWER={o[0]} ")
    print(o)

    #保存
    tf.save(model.state_dict(),"tashizan.pt")

    #optimizer = optim.Adadelta(model.parameters(), lr=1.0e-5)

