import torch as tf
import torch.nn as nn
import torch.optim as optim
import glob
import mido



class midiToTextLayer(nn.Module):
    def __init__(self):
        super(midiToTextLayer, self).__init__()

    def calcHZ(self, x):
        return 440 * 2**((x-69)/12)
    
    def forward(self, fileName):
        out = []
        mid = mido.MidiFile(fileName + '.mid')
        for message in mid.tracks[2]:
            if message.type == 'note_on' and message.time:
                out.append((self.calcHZ(message.note), message.time))
        return out


if __name__ == "__main__" :

    model = midiToTextLayer()

    print(model.forward('../midi/kiyosi'))