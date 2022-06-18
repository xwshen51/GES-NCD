import numpy as np
from rcd import *
from ncd import *
from fges import *
from scipy.io import loadmat
import argparse

parser = argparse.ArgumentParser(
    description="Run the reframed GES with nonparametric conditional dependence measure.")
parser.add_argument("--measure", default='rcd', type=str, choices=['ncd', 'rcd'],
                    help='Type of conditional dependence measure')
parser.add_argument("--dataset", default='mult', type=str,
                    choices=['mult', 'gp', 'multidim', 'syntren'],
                    help='Type of data set')
parser.add_argument("--id", default='1', type=str,
                    help='Data set ID')
parser.add_argument("--dim", type=int, default=10,
                    help='Number of nodes')
parser.add_argument("--num_sample", type=int, default=5000,
                    help='Number of samples')
parser.add_argument("--save_name", default='./results/none', type=str)
parser.add_argument("--checkpoint_frequency", type=int, default=0,
                    help="Frequency to checkpoint work (in seconds). \
                          Defaults to 0 to turn off checkpointing.")
parser.add_argument("--max_parents", type=int, default=None)
parser.add_argument("--tau", default=0.05, type=float)
parser.add_argument("--reg_width", type=int, default=40,
                    help='Network width of regressor')
parser.add_argument("--reg_depth", type=int, default=3,
                    help='Network depth of regressor')


def load_file(data_file):
    return np.loadtxt(data_file, skiprows = 0)

def main():
    args = parser.parse_args()

    if args.dataset == 'multdim':
        data_save = loadmat('./data/multidim.mat')
        id = int(args.id)
        A = data_save['G_save'][0][id]
        dataset = data_save['Data_save'][0][id]
        label = data_save['d_label_save'][0][id] - 1
    else:
        label = None
        if args.dataset == 'mult':
            dataset = np.load('./data/pnl_mult/p_10_e_10_n_5000_mult_data' + args.id + '.npy')
            A = np.load('./data/pnl_mult/p_10_e_10_n_5000_mult_DAG' + args.id + '.npy').astype(float)
        elif args.dataset == 'gp':
            dataset = np.load('./data/pnl_gp/p_10_e_10_n_5000_GP_data' + args.id + '.npy')
            A = np.load('./data/pnl_gp/p_10_e_10_n_5000_GP_DAG' + args.id + '.npy').astype(float)
        elif args.dataset == 'syntren':
            dataset = np.load('./data/syntren/data' + args.id + '.npy')
            A = np.load('./data/syntren/DAG' + args.id + '.npy')
        else:
            raise NotImplementedError("Not found data set.")
        dataset = dataset[:args.num_sample, :]

    if args.measure == 'rcd':
        score = RCD(dataset, args.tau)
    elif args.measure == 'ncd':
        device = torch.device('cuda')
        score = NCD(dataset=dataset, tau=args.tau, device=device,
                    reg_depth=args.reg_depth, reg_width=args.reg_width,
                    label=label)
    else:
        raise NotImplementedError("Not supported CD measure.")

    variables = list(range(args.dim))
    print("Running reframed GES with " + args.measure + " measure.")
    fges = FGES(variables, score, score_type=args.measure,
                filename=args.dataset,
                checkpoint_frequency=args.checkpoint_frequency,
                save_name=args.save_name,
                verbose=False,
                max_parents=args.max_parents)
    result = fges.search()

    save_dir = './results/{}_{}_{}_DAG.csv'.format(args.dataset, args.measure, args.id)
    print('Done! Estimated graph saved at: ' + save_dir)
    np.savetxt(save_dir, result['adjacency'], fmt='%i')

if __name__ == "__main__":
    main()
