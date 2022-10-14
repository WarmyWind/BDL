import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard.writer import SummaryWriter
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

from bnn.model import BayesianRegressor, evaluate_regression
from mcdropout.model import MCDropoutRegressor
from due.dkl import DKL, GP, initial_values
from due.sngp import Laplace
from due.fc_resnet import FCResNet
from svgp.model import MultitaskGPModel

from lib.evaluate_uncertainty import loss_divided_by_MCoutput, loss_divided_by_uncertainty
from lib.plot_func import *
from matplotlib import pyplot as plt

def main(hparams):
    Data_set = np.load(hparams.dataset_path, allow_pickle=True).tolist()
    X = Data_set['x']
    y = Data_set['y']
    X = StandardScaler().fit_transform(X)
    y = StandardScaler().fit_transform(y)

    # 5% for test
    X_test = X[np.int_(np.floor(X.shape[0] * 0.8)):, :]
    y_test = y[np.int_(np.floor(y.shape[0] * 0.8)):, :]
    X_test, y_test = torch.tensor(X_test).float(), torch.tensor(y_test).float()
    ds_test = torch.utils.data.TensorDataset(X_test, y_test)
    dl_test = torch.utils.data.DataLoader(ds_test, batch_size=512, shuffle=True)

    X_train = X[:np.int_(np.floor(X.shape[0]*0.8)), :]
    y_train = y[:np.int_(np.floor(y.shape[0]*0.8)), :]
    X_train, _, y_train, _ = train_test_split(X_train, y_train, test_size=.2, random_state=42)
    X_train, y_train = torch.tensor(X_train).float(), torch.tensor(y_train).float()
    ds_train = torch.utils.data.TensorDataset(X_train, y_train)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    method = hparams.method  # 'BNN' , 'MCDropout', 'DUE', 'SNGP'
    lr = hparams.learning_rate
    epochs = hparams.epochs
    results_dir = hparams.output_dir
    # writer = SummaryWriter(log_dir=str(results_dir))

    input_dim = hparams.input_dim
    hidden_dim = hparams.hidden_dim
    output_dim = hparams.output_dim
    dropout_rate = hparams.dropout_rate
    if method =='BNN':
        model = BayesianRegressor(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, hidden_depth=hparams.hidden_depth)
    elif method == 'MCDropout':
        model = MCDropoutRegressor(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, hidden_depth=hparams.hidden_depth, pdrop=dropout_rate)
        # loss_fn = F.mse_loss
    elif method == 'SVGP':
        model = MultitaskGPModel(num_tasks=output_dim, num_latents=output_dim, inducing_points=hparams.n_inducing_points)
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=output_dim).to(device)
        # elbo_fn = VariationalELBO(likelihood, model, num_data=len(ds_train))
        # loss_fn = lambda x, y: -elbo_fn(x, y)
    elif method == 'DUE':
        features = hparams.hidden_dim
        depth = hparams.hidden_depth
        spectral_normalization = hparams.spectral_normalization
        coeff = hparams.coeff
        n_power_iterations = hparams.n_power_iterations
        dropout_rate = hparams.dropout_rate

        feature_extractor = FCResNet(
            input_dim=input_dim,
            features=features,
            depth=depth,
            spectral_normalization=spectral_normalization,
            coeff=coeff,
            n_power_iterations=n_power_iterations,
            dropout_rate=dropout_rate
        )
        n_inducing_points = hparams.n_inducing_points
        kernel = "RBF"

        initial_inducing_points, initial_lengthscale = initial_values(
            ds_train, feature_extractor, n_inducing_points
        )

        gp = GP(
            num_outputs=output_dim,
            initial_lengthscale=initial_lengthscale,
            initial_inducing_points=initial_inducing_points,
            kernel=kernel,
        )

        model = DKL(feature_extractor, gp)
    else:
        raise Exception('Invalid method!')

    state_dict = torch.load(hparams.output_dir+'/model.pt')
    model.load_state_dict(state_dict)
    model.to(device)

    def test_step(batch):
        if method != 'MCDropout':
            model.eval()

        if method == 'SVGP':
            likelihood.eval()

        x, y = batch
        x = x.to(device)
        y = y.to(device)

        if method == 'MCDropout':
            criterion = torch.nn.MSELoss(reduction='none')
            y_hat, loss = model.sample_detailed_loss(inputs=x,
                                                 labels=y,
                                                 criterion=criterion,
                                                 sample_nbr=5)
            std = np.std(y_hat, axis=0)
            low_uncertainty_loss, high_uncertainty_loss = loss_divided_by_MCoutput(y_hat,
                                                                                   loss,
                                                                                   low_uncertainty_ratio=0.3,
                                                                                   high_uncertainty_ratio=0.3)
            # print(low_uncertainty_loss < high_uncertainty_loss)
        elif method == 'BNN':
            criterion = torch.nn.MSELoss(reduction='none')
            y_hat, _, loss, _ = model.sample_elbo_detailed_loss(inputs=x,
                                                   labels=y,
                                                   criterion=criterion,
                                                   sample_nbr=5,
                                                   complexity_cost_weight=1 / X_train.shape[0])
            std = np.std(y_hat, axis=0)
            low_uncertainty_loss, high_uncertainty_loss = loss_divided_by_MCoutput(y_hat,
                                                                                   loss,
                                                                                   low_uncertainty_ratio=0.3,
                                                                                   high_uncertainty_ratio=0.3)
            # print(low_uncertainty_loss < high_uncertainty_loss)
        elif method == 'SVGP':
            criterion = torch.nn.MSELoss(reduction='none')
            y_hat, loss, std = model.detailed_loss(inputs=x,
                                            labels=y,
                                            criterion=criterion)

            low_uncertainty_loss, high_uncertainty_loss = loss_divided_by_uncertainty(std,
                                                                                      loss,
                                                                                      low_uncertainty_ratio=0.3,
                                                                                      high_uncertainty_ratio=0.3)
            # print(low_uncertainty_loss < high_uncertainty_loss)
        elif method == 'DUE':
            criterion = torch.nn.MSELoss(reduction='none')
            y_hat, loss, std = model.detailed_loss(inputs=x,
                                            labels=y,
                                            criterion=criterion)
            low_uncertainty_loss, high_uncertainty_loss = loss_divided_by_uncertainty(std,
                                                                                      loss,
                                                                                      low_uncertainty_ratio=0.3,
                                                                                      high_uncertainty_ratio=0.3)
            # print(low_uncertainty_loss < high_uncertainty_loss)
        else:
            raise Exception('Invalid method!')
        return loss, low_uncertainty_loss, high_uncertainty_loss, std

    # std, loss = np.array([]), np.array([])
    # for batch in dl_test:
    #     _loss, _low_uncertainty_loss, _high_uncertainty_loss, _std = test_step(batch)
    #     if len(std) == 0:
    #         loss, std = _loss.cpu().detach().numpy(), _std
    #     else:
    #         loss = np.concatenate([loss, _loss.cpu().detach().numpy()], axis=0)
    #         std = np.concatenate([std, _std], axis=0)
    # np.save('./test_result/1009/' + hparams.method + '_loss_std.npy', {'loss':loss, 'std':std})


    idx = 178
    X_test_sample, y_test_sample = X_test[idx].unsqueeze(0).to(device), y_test[idx].unsqueeze(0).to(device)
    if method == 'BNN' or method == 'MCDropout':
        if method == 'BNN':
            model.eval()
            criterion = torch.nn.MSELoss(reduction='none')
            y_hat, _, loss, _ = model.sample_elbo_detailed_loss(inputs=X_test_sample,
                                                                labels=y_test_sample,
                                                                criterion=criterion,
                                                                sample_nbr=5,
                                                                complexity_cost_weight=1 / X_train.shape[0])
            y_std = np.std(y_hat, axis=0)
            fig, ax = plot_uncertainty(np.array(X_test_sample.cpu().detach()), np.array(y_test_sample.cpu().detach()), np.mean(y_hat, axis=0), y_std)
        else:
            criterion = torch.nn.MSELoss(reduction='none')
            y_hat, loss = model.sample_detailed_loss(inputs=X_test_sample,
                                                     labels=y_test_sample,
                                                     criterion=criterion,
                                                     sample_nbr=5)
            y_std = np.std(y_hat, axis=0)
            fig, ax = plot_uncertainty(np.array(X_test_sample.cpu().detach()), np.array(y_test_sample.cpu().detach()), np.mean(y_hat, axis=0), y_std)
    elif method == 'SVGP':
        model.eval()
        criterion = torch.nn.MSELoss(reduction='none')
        y_hat, loss, std = model.detailed_loss(inputs=X_test_sample,
                                        labels=y_test_sample,
                                        criterion=criterion)
        fig, ax = plot_uncertainty(np.array(X_test_sample.cpu().detach()), np.array(y_test_sample.cpu().detach()), y_hat, std)
    elif method == 'DUE':
        model.eval()
        criterion = torch.nn.MSELoss(reduction='none')
        y_hat, loss, std = model.detailed_loss(inputs=X_test_sample,
                                               labels=y_test_sample,
                                               criterion=criterion)
        fig, ax = plot_uncertainty(np.array(X_test_sample.cpu().detach()), np.array(y_test_sample.cpu().detach()), y_hat, std)
    plt.show()


if __name__ == "__main__":
    class Dotdict(dict):
        __setattr__ = dict.__setitem__
        __getattr__ = dict.__getitem__

    import json
    root_path = 'runs/MCDropout/2022-10-08-Saturday-21-33-07'
    with open(root_path + '/hparams.json') as file:
        hparams_dict = json.load(file)
    hparams = Dotdict(hparams_dict)
    main(hparams)