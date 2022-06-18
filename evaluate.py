import copy
import numpy as np
import pandas as pd
from cdt.metrics import SHD_CPDAG, SID_CPDAG


class MetricsDAG(object):
    """
    Compute various accuracy metrics for B_est.
    precision: TP/(TP + FP)
    recall: TP/(TP + FN)
    F1: 2*(recall*precision)/(recall+precision)
    Parameters
    ----------
    B_est: np.ndarray
        [d, d] estimate, {0, 1, -1}, -1 is undirected edge in CPDAG.
    B_true: np.ndarray
        [d, d] ground truth graph, {0, 1}.
    """

    def __init__(self, B_est, B_true):
        self.B_est = copy.deepcopy(B_est)
        self.B_true = copy.deepcopy(B_true)

        self.metrics = MetricsDAG._count_accuracy(self.B_est, self.B_true)

    @staticmethod
    def _count_accuracy(B_est, B_true):
        """
        Parameters
        ----------
        B_est: np.ndarray
            [d, d] estimate, {0, 1, -1}, -1 is undirected edge in CPDAG.
        B_true: np.ndarray
            [d, d] ground truth graph, {0, 1}.
        """

        # trans diagonal element into 0
        for i in range(len(B_est)):
            if B_est[i, i] == 1:
                B_est[i, i] = 0
            if B_true[i, i] == 1:
                B_true[i, i] = 0

        shd_c = SHD_CPDAG(B_true, B_est)
        sid_l, sid_u = SID_CPDAG(B_true, B_est)

        W_p = pd.DataFrame(B_est)
        W_true = pd.DataFrame(B_true)

        precision, recall, F1 = MetricsDAG._cal_precision_recall(W_p, W_true)

        return precision, recall, F1, shd_c, sid_l, sid_u

    @staticmethod
    def _cal_precision_recall(W_p, W_true):
        """
        Parameters
        ----------
        W_p: pd.DataDrame
            [d, d] estimate, {0, 1, -1}, -1 is undirected edge in CPDAG.
        W_true: pd.DataDrame
            [d, d] ground truth graph, {0, 1}.

        Return
        ------
        precision: float
            TP/(TP + FP)
        recall: float
            TP/(TP + FN)
        F1: float
            2*(recall*precision)/(recall+precision)
        """

        assert (W_p.shape == W_true.shape and W_p.shape[0] == W_p.shape[1])
        TP = (W_p + W_true).applymap(lambda elem: 1 if elem == 2 else 0).sum(axis=1).sum()
        TP_FP = W_p.sum(axis=1).sum()
        TP_FN = W_true.sum(axis=1).sum()
        precision = TP / TP_FP
        recall = TP / TP_FN
        F1 = 2 * (recall * precision) / (recall + precision)

        return precision, recall, F1

if __name__ == "__main__":
    A = np.load('./data/pnl_mult/p_10_e_10_n_5000_mult_DAG1.npy').astype(float)
    A_est = np.loadtxt('./results/est_mult_rcd_1_DAG.csv')
    met = MetricsDAG(A_est, A)
    precision, recall, F1, shd_c, sid_l, sid_u = met.metrics
    print('shd_c {}, sid_l {}, sid_u {}, F1 {:.4f}, precision {:.4f}, recall {:.4f}'.format(
        shd_c, sid_l, sid_u, F1, precision, recall))
