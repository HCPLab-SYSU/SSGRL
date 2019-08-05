import math

import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.nn as nn

class Element_Wise_Layer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Element_Wise_Layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(in_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        for i in range(self.in_features):
            self.weight[i].data.uniform_(-stdv, stdv)
        if self.bias is not None:
            for i in range(self.in_features):
                self.bias[i].data.uniform_(-stdv, stdv)


    def forward(self, input):
        #print('input_size: {}'.format(input.size()))
        #(class_num, feature_dim)
        #print('weight size: {}'.format(self.weight.size()))
        x = input * self.weight
        #(class_num, 1)
        x = torch.sum(x,2)
        #print('after reducing(sum): {}'.format(x.size()))
        if self.bias is not None:
            x = x + self.bias
        return x

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None)
