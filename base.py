import torch
from torch import nn
import torch.nn.functional as F

import math

class LSTM(nn.Module):
    
    def __init__(self,
                input_size,
                hidden_size,
                use_bias=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = use_bias
        self.x = nn.Linear(input_size, hidden_size*4, bias=use_bias)
        self.h = nn.Linear(input_size, hidden_size*4, bias=use_bias)
        self.c = torch.Tensor(hidden_size*3)
        self.reset()
    
    def forward(self, input, hidden):
        h, c = hidden
        input = input.view(-1, input.size(1))
        gates = self.x(input) + self.h(input)
        gates = gates.squeeze()
        c_i, c_f, c_o = self.c.unsqueeze(0).chunk(3,1)
        in_gate, forget_gate, cell_gate, out_gate = gates.chunk(4,1)
        in_gate = torch.sigmoid(in_gate + (c_i*c))
        forget_gate = torch.sigmoid(forget_gate + (c_f*c))
        cell_gate = forget_gate*c + torch.tanh(cell_gate)*in_gate
        out_gate = torch.sigmoid(out_gate + (c_o*cell_gate))
        
        return ((out_gate*F.tanh(cell_gate)), cell_gate)
       
    def reset(self):
        dev = 1/math.sqrt(self.hidden_size)
        
        for params in self.parameters():
            params.data.uniform_(-dev, +dev)
            
            
class LSTMModel(nn.Module):
    
    def __init__(
        self,
        input_size,
        hidden_size,
        layer_size,
        output_size,
        use_bias=False
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layer_size = layer_size
        self.lstm = LSTM(input_size, hidden_size, layer_size)
        self.feedforward = nn.Linear(hidden_size, output_size)
        
    def forward(self, inputs):
        
        x = torch.zeros(self.layer_size, inputs.size(0), self.hidden_size)
        
        h, c = torch.Tensor(x), torch.Tensor(x)
        
        outputs = []
        __h, __c = h[0,:,:], c[0,:,:]
        
        for i in range(x.size(1)):
            __h, __c = self.lstm(x[:, i, :], (__h,__c))
            outputs.append(__h)
            
        return self.feedforward(outputs[-1].squeeze())