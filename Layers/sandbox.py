'''
import torch
import matplotlib.pyplot as plt
import math

y1 = torch.sin(3 * 2 * math.pi * x)
x = torch.arange(0, 1, 0.01)
y2 = torch.cos(5 * 2 * math.pi * x)
y = torch.stack([x, y1 + y2], 1)
#plt.plot(x, y)
#plt.show()
print(y)
z = torch.fft(y, 1)
l = 10
plt.plot(torch.arange(l), torch.abs(z[:l]))
plt.show()
'''

print(2**(1/12))