import torch
import simpleNN

#サンプル用 ptから読み込むやつ

model=simpleNN.SumNet()
model.load_state_dict(torch.load("./tashizan.pt"))

model.eval()

pred=model(torch.Tensor([10,20]))

print ("simpleNN")
print (model)
print (pred)


