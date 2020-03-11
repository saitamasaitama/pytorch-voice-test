import typing
import wave
import struct
import torch.nn as nn
import torch.nn.functional as F
import torch

class HumanoidNoneNet(nn.Module):

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

