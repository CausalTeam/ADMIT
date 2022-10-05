import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np

class Truncated_power(nn.Module):
    def __init__(self, degree, knots):
        super(Truncated_power, self).__init__()
        self.degree = degree
        self.knots = knots
        self.num_of_basis = self.degree + 1 + len(self.knots)
        self.relu = nn.ReLU(inplace=True)

        if self.degree == 0:
            print('Degree should not set to be 0!')
            raise ValueError

        if not isinstance(self.degree, int):
            print('Degree should be int')
            raise ValueError

    def forward(self, x):
        x = x.squeeze()
        out = torch.zeros(x.shape[0], self.num_of_basis, device=x.device)
        for _ in range(self.num_of_basis):
            if _ <= self.degree:
                out[:, _] = x**_
            else:
                out[:, _] = (self.relu(x - self.knots[_ - self.degree - 1])) ** self.degree

        return out 

class MLP_treatnet(nn.Module):
    def __init__(self, num_out, n_hidden=10, num_in=4) -> None:
        super(MLP_treatnet, self).__init__()
        self.num_in = num_in
        self.hidden1 = torch.nn.Linear(num_in, n_hidden)            
        self.hidden2 = torch.nn.Linear(n_hidden, n_hidden)    
        self.predict = torch.nn.Linear(n_hidden, num_out)
        self.act = nn.ReLU()
    
    def forward(self, x):
        x_mix = torch.zeros([x.shape[0], 3])
        x_mix = x_mix.to(x.device)
        x_mix[:, 0] = 0
        
        x_mix[:, 1] = torch.cos(x * np.pi)
        x_mix[:, 2] = torch.sin(x * np.pi)
        h = self.act(self.hidden1(x_mix))     # relu
        y = self.predict(h)

        return y

class Dynamic_FC(nn.Module):
    def __init__(self, ind, outd, degree, knots, act='relu', isbias=1, islastlayer=0, dynamic_type='power'):
        super(Dynamic_FC, self).__init__()
        self.ind = ind
        self.outd = outd
        self.degree = degree
        self.knots = knots

        self.islastlayer = islastlayer

        self.isbias = isbias

        if dynamic_type == 'power':
            self.spb = Truncated_power(degree, knots)
            self.d = self.spb.num_of_basis # num of basis
        else:
            self.spb = MLP_treatnet(num_out=10, num_in=3)
            self.d = 10

        self.weight = nn.Parameter(torch.rand(self.ind, self.outd, self.d), requires_grad=True)

        if self.isbias:
            self.bias = nn.Parameter(torch.rand(self.outd, self.d), requires_grad=True)
        else:
            self.bias = None

        if act == 'relu':
            self.act = nn.ReLU()
        elif act == 'soft':
            self.act = nn.Softplus()
        elif act == 'tanh':
            self.act = nn.Tanh()
        elif act == 'sigmoid':
            self.act = nn.Sigmoid()
        else:
            self.act = None

    def forward(self, x):
        # x: batch_size * (treatment, other feature)
        x_feature = x[:, 1:]
        x_treat = x[:, 0]

        x_feature_weight = torch.matmul(self.weight.T, x_feature.T).T # bs, outd, d

        x_treat_basis = self.spb(x_treat) # bs, d
        x_treat_basis_ = torch.unsqueeze(x_treat_basis, 1)
        out = torch.sum(x_feature_weight * x_treat_basis_, dim=2) # bs, outd

        if self.isbias:
            out_bias = torch.matmul(self.bias, x_treat_basis.T).T
            out = out + out_bias

        if self.act is not None:
            out = self.act(out)
        if not self.islastlayer:
            out = torch.cat((torch.unsqueeze(x_treat, 1), out), 1)

        return out