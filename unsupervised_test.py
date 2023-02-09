import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard.writer import SummaryWriter
import torch_geometric

import numpy as np

import gpytorch
from gpytorch.mlls import VariationalELBO
from gpytorch.likelihoods import GaussianLikelihood

from ignite.engine import Events, Engine
from ignite.metrics import Average, Loss
from ignite.contrib.handlers import ProgressBar

from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from gnn.model import IGCNet, BayesianIGCNet, MCDropoutIGCNet
from bnn.model import BayesianRegressor, evaluate_regression
from mcdropout.model import MCDropoutRegressor
from svgp.model import MultitaskGPModel, ApproximateGPModel
from due.dkl import DKL, GP, initial_values
from due.sngp import Laplace
from due.fc_resnet import FCResNet
from graphdkl.model import GraphDKL

from lib.datasets import get_wGaussian_graph_data
from lib.evaluate_uncertainty import loss_divided_by_MCoutput, loss_divided_by_uncertainty

def main(hparams):
    batch_size = hparams.batch_size
    # X, y = load_boston(return_X_y=True)
    if hparams.task == 'sum_rate_maximization':
        K = 10  # number of users
        num_train = 10000  # number of training samples
        num_test = 2000  # number of testing  samples
        trainseed = 0  # set random seed for training set
        testseed = 7  # set random seed for test set
        print('Gaussian IC Case: K=%d, Total Samples: %d' % (K, num_train))
        var_db = 10
        var = 1 / 10 ** (var_db / 10)
        ds_train = get_wGaussian_graph_data(K, num_train, seed=trainseed, var_noise=var, WMMSE_eval=False)
        ds_test = get_wGaussian_graph_data(K, num_test, seed=testseed, var_noise=var, WMMSE_eval=True)
        dl_train = torch_geometric.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True)
        dl_test = torch_geometric.data.DataLoader(ds_test, batch_size=64, shuffle=False)
    else:
        raise Exception("Invalid task!")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    method = hparams.method  # 'GNN' , 'BGNN'

    lr = hparams.learning_rate
    epochs = hparams.epochs
    results_dir = hparams.output_dir
    # writer = SummaryWriter(log_dir=str(results_dir))

    input_dim = hparams.input_dim
    hidden_dim = hparams.hidden_dim
    output_dim = hparams.output_dim
    dropout_rate = hparams.dropout_rate

    def sr_loss(data, out, K, reduction='sum'):
        if len(out.size()) == 2:
            power = out[:, -1]
        else:
            power = out
        power = torch.reshape(power, (-1, K, 1))
        abs_H = data.y
        abs_H_2 = torch.pow(abs_H, 2)
        rx_power = torch.mul(abs_H_2, power)
        mask = torch.eye(K)
        mask = mask.to(device)
        valid_rx_power = torch.sum(torch.mul(rx_power, mask), 1)
        interference = torch.sum(torch.mul(rx_power, 1 - mask), 1) + var
        rate = torch.log(1 + torch.div(valid_rx_power, interference))
        w_rate = torch.mul(data.pos, rate)
        sum_rate = torch.sum(w_rate, 1)
        if reduction == 'sum':
            loss = torch.neg(torch.mean(sum_rate))
        elif reduction == 'none':
            loss = torch.neg(sum_rate)
        else:
            raise Exception("Invalid reduction")
        return loss

    if method == 'GNN' or method == 'BGNN' or method == 'MCDropoutGNN':
        if method == 'GNN':
            model = IGCNet()
        elif method == 'BGNN':
            model = BayesianIGCNet()
        elif method == 'MCDropoutGNN':
            model = MCDropoutIGCNet()
    elif method == 'GraphDKL':
        feature_extractor = IGCNet()
        n_inducing_points = hparams.n_inducing_points
        gp = ApproximateGPModel(n_inducing_points)
        model = GraphDKL(feature_extractor, gp)
    else:
        raise Exception('Invalid method!')

    state_dict = torch.load(hparams.output_dir+'/model.pt')
    model.load_state_dict(state_dict)
    model.to(device)

    def test_step(batch):
        if method != 'MCDropoutGNN':
            model.eval()

        x = batch
        x = x.to(device)
        # y = y.to(device)

        if method == 'GNN':
            out = model(x)
            loss = sr_loss(x, out, K)
            loss = - loss
        elif method == 'BGNN':
            # complexity_cost_weight = 1 / num_train
            # complexity_cost_weight = complexity_cost_weight \
            #                          * len(dl_train) * 2 ** (len(dl_train) - engine.state.iteration) \
            #                          / (2 ** (len(dl_train) - 1))
            y_hat, _, likelihood_loss, _ = model.my_sample_elbo_detailed_loss(x,
                                                               sr_loss,
                                                               sample_nbr=hparams.sample_nbr,
                                                               reduction='none',
                                                               complexity_cost_weight=1,
                                                               K=K)
            power = y_hat[:,:,-1]
            power = np.reshape(power, (power.shape[0], -1, K))
            std = np.std(power, axis=0)
            std = np.sum(std, axis=-1)
            loss = -np.mean(likelihood_loss, axis=0)
            low_uncertainty_loss, high_uncertainty_loss = loss_divided_by_uncertainty(std,
                                                                                   loss,
                                                                                   low_uncertainty_ratio=0.3,
                                                                                   high_uncertainty_ratio=0.3)
            print(np.mean(low_uncertainty_loss), np.mean(high_uncertainty_loss))
        elif method == 'MCDropoutGNN':
            y_hat, loss = model.sample_detailed_loss(x,
                                                 sr_loss,
                                                 sample_nbr=hparams.sample_nbr,
                                                 reduction='none',
                                                 K=K)
            power = y_hat[:, :, -1]
            power = np.reshape(power, (power.shape[0], -1, K))
            std = np.std(power, axis=0)
            std = np.sum(std, axis=-1)
            loss = -np.mean(loss, axis=0)
            low_uncertainty_loss, high_uncertainty_loss = loss_divided_by_uncertainty(std,
                                                                                      loss,
                                                                                      low_uncertainty_ratio=0.3,
                                                                                      high_uncertainty_ratio=0.3)
            print(np.mean(low_uncertainty_loss), np.mean(high_uncertainty_loss))
        elif method == 'GraphDKL':
            y_pred, loss, _ = model.detailed_loss(x,
                                             sr_loss,
                                             reduction='none',
                                             labels=None,
                                             K=K)
            std = y_pred.stddev.cpu().detach().numpy()
            std = np.reshape(std, (-1, K))
            std = np.sum(std, axis=-1)
            loss = -loss
            low_uncertainty_loss, high_uncertainty_loss = loss_divided_by_uncertainty(std,
                                                                                      loss,
                                                                                      low_uncertainty_ratio=0.3,
                                                                                      high_uncertainty_ratio=0.3)
            print(np.mean(low_uncertainty_loss), np.mean(high_uncertainty_loss))

        else:
            raise Exception('Invalid method!')

        return loss, low_uncertainty_loss, high_uncertainty_loss, std

    std, loss = np.array([]), np.array([])
    for batch in dl_test:
        _loss, _low_uncertainty_loss, _high_uncertainty_loss, _std = test_step(batch)
        if len(std) == 0:
            loss, std = _loss, _std
        else:
            loss = np.concatenate([loss, _loss], axis=0)
            std = np.concatenate([std, _std], axis=0)
    np.save('./test_result/1023/' + hparams.method + '_loss_std.npy', {'loss':loss, 'std':std})

if __name__ == "__main__":
    class Dotdict(dict):
        __setattr__ = dict.__setitem__
        __getattr__ = dict.__getitem__

    import json
    # root_path = 'runs/sum_rate_maximization/BGNN/2022-10-20-Thursday-17-14-16'
    root_path = 'runs/sum_rate_maximization/GraphDKL/2022-10-23-Sunday-15-34-46'
    with open(root_path + '/hparams.json') as file:
        hparams_dict = json.load(file)
    hparams = Dotdict(hparams_dict)
    main(hparams)