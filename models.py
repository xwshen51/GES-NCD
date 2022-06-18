import torch.nn as nn
from torch.nn.utils import spectral_norm
import torch.nn.init as init

class Test(nn.Module):
    '''Test function'''
    def __init__(self, input_size=5, num_layer=4, hidden_size=20):
        super().__init__()
        net = [
            snlinear(input_size, hidden_size, bias=False),
            nn.ReLU(True),
        ]
        for i in range(num_layer - 2):
            net.append(snlinear(hidden_size, hidden_size, bias=False))
            net.append(nn.ReLU(True))
        net.append(snlinear(hidden_size, 1, bias=False))
        self.net = nn.Sequential(*net)
        self.sample()

    def sample(self, ini_type='kaiming_unif'):
        if ini_type == 'kaiming_unif':
            self.apply(init_kaiming_unif)
        elif ini_type == 'kaiming_norm':
            self.apply(init_kaiming_norm)
        else: # normal
            self.apply(init_norm)

    def forward(self, x):
        return self.net(x)

class NLR(nn.Module):
    '''Nonlinear regressor'''
    def __init__(self, input_size=5, num_layer=4, hidden_size=20):
        super().__init__()
        net = [
            snlinear(input_size, hidden_size, bias=False),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(True),
        ]
        for i in range(num_layer - 2):
            net.append(snlinear(hidden_size, hidden_size, bias=False))
            net.append(nn.BatchNorm1d(hidden_size))
            net.append(nn.ReLU(True))
        net.append(snlinear(hidden_size, 1, bias=False))
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)

def snlinear(in_features, out_features, bias=False):
    return spectral_norm(nn.Linear(in_features=in_features, out_features=out_features, bias=bias))

def init_kaiming_unif(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        init.kaiming_uniform_(m.weight)

def init_kaiming_norm(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        init.kaiming_normal_(m.weight)

def init_norm(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        init.normal_(m.weight, mean=0, std=1)
