import torch

a = torch.tensor([], dtype=torch.int)
b = torch.tensor([1, 2], dtype=torch.int)

print(torch.cat([a, b]))