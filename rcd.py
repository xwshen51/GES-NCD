import numpy as np
from rpy2.robjects.packages import importr
# One needs to install R package 'FOCI' in advance
importr('FOCI', lib_loc="~/R/x86_64-pc-linux-gnu-library/3.6")
from rpy2.robjects import FloatVector
from rpy2.robjects import r

class RCD:
    def __init__(self, dataset=None, tau=1e-2):
        self.dataset = dataset
        self.tau = tau

    def local_score_diff_parents(self, node1, node2, parents):
        data1 = self.dataset[:, node1]
        data2 = self.dataset[:, node2]
        data1_r = FloatVector(data1)
        data2_r = FloatVector(data2)
        if len(parents) == 0:
            score = r['codec'](data1_r, data2_r)
        else:
            datap = self.dataset[:, parents]
            nr, nc = datap.shape
            datap_r = r['matrix'](FloatVector(datap.reshape(nr*nc)), nrow=nr, ncol=nc, byrow=1)
            score = r['codec'](data1_r, data2_r, datap_r)

        return float(np.array(score)) - self.tau

    def local_score_diff(self, node1, node2):
        return self.local_score_diff_parents(node1, node2, [])
