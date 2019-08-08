import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F

from networks.resnet import resnet101
from semantic import semantic
from ggnn import GGNN
from element_wise_layer import Element_Wise_Layer

class SSGRL(nn.Module):
    def __init__(self, image_feature_dim, output_dim, time_step,
                 adjacency_matrix, word_features, num_classes=80, word_feature_dim = 300):
        super(SSGRL, self).__init__()
        self.resnet_101 = resnet101()

        self.num_classes = num_classes
        self.word_feature_dim = word_feature_dim
        self.image_feature_dim = image_feature_dim
        
        self.word_semantic = semantic(num_classes= self.num_classes,
                                      image_feature_dim = self.image_feature_dim,
                                      word_feature_dim=self.word_feature_dim)

        self.word_features = word_features
        self._word_features = self.load_features()
        self.adjacency_matrix = adjacency_matrix
        self._in_matrix, self._out_matrix = self.load_matrix()
        self.time_step = time_step
        
        self.graph_net = GGNN(input_dim=self.image_feature_dim,
                              time_step=self.time_step,
                              in_matrix=self._in_matrix,
                              out_matrix=self._out_matrix)

        self.output_dim = output_dim
        self.fc_output = nn.Linear(2*self.image_feature_dim, self.output_dim)
        self.classifiers = Element_Wise_Layer(self.num_classes, self.output_dim)

    def forward(self, x):
        batch_size = x.size()[0]
        img_feature_map = self.resnet_101(x)
        graph_net_input = self.word_semantic(batch_size,
                                             img_feature_map,
                                             torch.tensor(self._word_features).cuda())
        graph_net_feature = self.graph_net(graph_net_input)

        output = torch.cat((graph_net_feature.view(batch_size*self.num_classes,-1), graph_net_input.view(-1, self.image_feature_dim)), 1)
        output = self.fc_output(output)
        output = torch.tanh(output)
        output = output.contiguous().view(batch_size, self.num_classes, self.output_dim)
        result = self.classifiers(output)
        return result 

    def load_features(self):
        return Variable(torch.from_numpy(np.load(self.word_features).astype(np.float32))).cuda()

    def load_matrix(self):
        mat = np.load(self.adjacency_matrix)
        _in_matrix, _out_matrix = mat.astype(np.float32), mat.T.astype(np.float32)
        _in_matrix = Variable(torch.from_numpy(_in_matrix), requires_grad=False).cuda()
        _out_matrix = Variable(torch.from_numpy(_out_matrix), requires_grad=False).cuda()
        return _in_matrix, _out_matrix
