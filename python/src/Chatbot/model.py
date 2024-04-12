import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):

    def __init__(self,input_size,hidden_size,num_class):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, num_class)
        self.relu = nn.ReLU()

    def forward(self,x):
        out = self.layer1(x)
        out = self.relu(out)
        out = self.layer2(out)
        out = self.relu(out)
        out = self.layer3(out)
        # no activation and no softmax at the end
        return out