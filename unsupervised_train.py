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

from gnn.model import IGCNet
from bnn.model import BayesianRegressor, evaluate_regression
from mcdropout.model import MCDropoutRegressor
from svgp.model import MultitaskGPModel, ApproximateGPModel
from due.dkl import DKL, GP, initial_values
from due.sngp import Laplace
from due.fc_resnet import FCResNet


from lib.datasets import get_wGaussian_graph_data




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
        ds_test = get_wGaussian_graph_data(K+2, num_test, seed=testseed, var_noise=var, WMMSE_eval=True)
        dl_train = torch_geometric.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True)
        dl_test = torch_geometric.data.DataLoader(ds_test, batch_size=1, shuffle=False)
    else:
        raise Exception("Invalid task!")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    method = hparams.method  # 'GNN' , 'GNN+GP'

    lr = hparams.learning_rate
    epochs = hparams.epochs
    results_dir = hparams.output_dir
    writer = SummaryWriter(log_dir=str(results_dir))

    input_dim = hparams.input_dim
    hidden_dim = hparams.hidden_dim
    output_dim = hparams.output_dim
    dropout_rate = hparams.dropout_rate
    if method =='GNN':
        model = IGCNet()
        def sr_loss(data, out, K):
            power = out[:, 2]
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
            sum_rate = torch.mean(torch.sum(w_rate, 1))
            loss = torch.neg(sum_rate)
            return loss
    elif method == 'GNN+GP':
        n_inducing_points = hparams.n_inducing_points
        kernel = "RBF"
        gp = ApproximateGPModel(n_inducing_points)


    else:
        raise Exception('Invalid method!')
    model.to(device)

    parameters = [
        {"params": model.parameters(), "lr": lr},
    ]

    # if method == 'DUE' or method == 'SVGP':
    #     parameters.append({"params": likelihood.parameters(), "lr": lr})

    optimizer = optim.Adam(parameters, lr=lr)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)
    pbar = ProgressBar()


    def step(engine, batch):
        model.train()
        # if method == 'DUE':
        #     likelihood.train()

        optimizer.zero_grad()

        x = batch
        x = x.to(device)
        # y = y.to(device)
        if method == 'GNN':
            out = model(x)
            loss = sr_loss(x, out, K)
        else:
            raise Exception('Invalid method!')

        loss.backward()
        optimizer.step()
        # scheduler.step()

        return loss.item()


    def eval_step(engine, batch):
        model.eval()
        # if method == 'DUE':
        #     likelihood.eval()

        x = batch
        x = x.to(device)
        # y = y.to(device)
        # y.squeeze_()

        if method == 'GNN':
            out = model(x)
            loss = sr_loss(x, out, K+2)
        else:
            raise Exception('Invalid method!')

        return loss


    trainer = Engine(step)
    evaluator = Engine(eval_step)

    metric = Average()
    metric.attach(trainer, "loss")
    pbar.attach(trainer)

    metric = Average()
    metric.attach(evaluator, "loss")


    @trainer.on(Events.EPOCH_COMPLETED(every=int(5)))
    def log_results(trainer):
        evaluator.run(dl_test)
        print(f"Results - Epoch: {trainer.state.epoch} - "
              f"Test Mse loss: {evaluator.state.metrics['loss']:.4f} - "
              f"Train Loss: {trainer.state.metrics['loss']:.4f}")
        writer.add_scalar("Loss/train", trainer.state.metrics['loss'], trainer.state.epoch)
        writer.add_scalar("Loss/eval", evaluator.state.metrics['loss'], trainer.state.epoch)

    # if method == 'SNGP':
    #     @trainer.on(Events.EPOCH_STARTED)
    #     def reset_precision_matrix(trainer):
    #         model.reset_precision_matrix()

    trainer.run(dl_train, max_epochs=epochs)

    ######################################## Save #######################################

    torch.save(model.state_dict(), results_dir + "/model.pt")
    # if method == 'DUE' or method == 'SVGP':
    #     torch.save(likelihood.state_dict(), results_dir + "/likelihood.pt")

    writer.close()

    hparams.save(results_dir + "/hparams.json")

if __name__ == "__main__":
    from unsupervised_hyperpara import hparams
    main(hparams)