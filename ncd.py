import torch
from models import *


class NCD:

    def __init__(self, test_depth=2, test_width=20, lr_test=1e-2,
                 reg_depth=3, reg_width=20, lr_reg=1e-2, num_iter_nlr=5, n_epoch=20,
                 device=None, dataset=None, label=None,
                 tau=1e-3):
        '''NCD measure

        Args:
            test_depth, test_width: the depth and width of test function networks
            reg_depth, reg_width: the depth and width of regressor networks
            lr_test, lr_reg: learning rates of test functions and regressors
            num_iter_nlr: training steps of nonlinear regression (T_r)
            n_epoch: training steps of test function (T_t)
            tau: threshold. The larger the sparser, the smaller the denser.
            dataset: numpy array.
            label: for multi-dimensional case.
        '''
        self.d = dataset.shape[1]
        self.dataset = torch.FloatTensor(dataset).to(device)
        self.label = label
        self.num_iter_nlr = num_iter_nlr
        self.n_epoch = n_epoch
        self.tau = tau
        self.device = device
        self.f = Test(self.d, num_layer=test_depth, hidden_size=test_width).to(device)
        self.g = Test(self.d, num_layer=test_depth, hidden_size=test_width).to(device)
        self.optim_f = torch.optim.Adam(self.f.parameters(), lr=lr_test)
        self.optim_g = torch.optim.Adam(self.g.parameters(), lr=lr_test)
        self.phi = NLR(self.d, num_layer=reg_depth, hidden_size=reg_width).to(device)
        self.psi = NLR(self.d, num_layer=reg_depth, hidden_size=reg_width).to(device)
        self.optim_phi = torch.optim.Adam(self.phi.parameters(), lr=lr_reg)
        self.optim_psi = torch.optim.Adam(self.psi.parameters(), lr=lr_reg)


    def local_score_diff_parents(self, node1, node2, parents):#todo: delete nds

        # multi-dimensional
        if self.label is not None:
            if len(parents) > 0:
                parents = self.label[0][parents][0].astype(int)
            node1 = self.label[0][node1][0].astype(int)
            node2 = self.label[0][node2][0].astype(int)

        data_pa = torch.zeros(self.dataset.shape, dtype=torch.float, device=self.device)
        data_pa[:, parents] = self.dataset[:, parents]
        data1 = data_pa.clone()
        data2 = data_pa.clone()
        data1[:, node1] = self.dataset[:, node1]
        data2[:, node2] = self.dataset[:, node2]
        loss_reg = nn.MSELoss()

        for epoch in range(self.n_epoch):
            self.f.train()
            self.g.train()
            self.f.zero_grad()
            self.g.zero_grad()
            target1 = self.f(data1)
            target2 = self.g(data2)

            if len(parents) == 0:
                score = cc_square(target1, target2)
            else:
                self.phi.train()
                self.psi.train()
                for _ in range(self.num_iter_nlr):
                    self.phi.zero_grad()
                    self.psi.zero_grad()
                    ls_f = loss_reg(target1.detach(), self.phi(data_pa))
                    ls_g = loss_reg(target2.detach(), self.psi(data_pa))
                    ls_f.backward()
                    ls_g.backward()
                    self.optim_phi.step()
                    self.optim_psi.step()
                res1 = target1 - self.phi(data_pa)
                res2 = target2 - self.psi(data_pa)
                score = cc_square(res1, res2)

            loss = -score
            loss.backward()
            self.optim_f.step()
            self.optim_g.step()

        return score.item() - self.tau

    def local_score_diff(self, node1, node2):
        return self.local_score_diff_parents(node1, node2, [])


def cc_square(x, y):
    '''Square of Pearson correlation coefficient
    x, y: n*1 (centralized) data (torch Tensor)'''
    cov = (x * y).mean() - x.mean() * y.mean()
    var_x = x.var(unbiased=False)
    var_y = y.var(unbiased=False)
    return cov**2 / (var_x * var_y)
