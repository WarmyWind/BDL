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

from bnn.model import BayesianRegressor, evaluate_regression
from mcdropout.model import MCDropoutRegressor
from svgp.model import MultitaskGPModel
from due.dkl import DKL, GP, initial_values
from due.sngp import Laplace
from due.fc_resnet import FCResNet

from lib.datasets import get_wGaussian_graph_data




def main(hparams):
    batch_size = hparams.batch_size
    # X, y = load_boston(return_X_y=True)
    if hparams.task == 'large_channel_predict':
        Data_set = np.load(hparams.dataset_path, allow_pickle=True).tolist()
        X = Data_set['x']
        y = Data_set['y']
        X = StandardScaler().fit_transform(X)
        y = StandardScaler().fit_transform(y)

        # 80% data for training, 20% for model evaluation
        X = X[:np.int_(np.floor(X.shape[0]*0.8)), :]
        y = y[:np.int_(np.floor(y.shape[0]*0.8)), :]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
        X_train, y_train = torch.tensor(X_train).float(), torch.tensor(y_train).float()
        X_test, y_test = torch.tensor(X_test).float(), torch.tensor(y_test).float()

        ds_train = torch.utils.data.TensorDataset(X_train, y_train)
        dl_train = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True)
        ds_test = torch.utils.data.TensorDataset(X_test, y_test)
        dl_test = torch.utils.data.DataLoader(ds_test, batch_size=512, shuffle=True)
    else:
        raise Exception('Invalid task!')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    method = hparams.method  # 'BNN' , 'MCDropout', 'DUE', 'SNGP' or 'SVGP'

    lr = hparams.learning_rate
    epochs = hparams.epochs
    results_dir = hparams.output_dir
    writer = SummaryWriter(log_dir=str(results_dir))

    input_dim = hparams.input_dim
    hidden_dim = hparams.hidden_dim
    output_dim = hparams.output_dim
    dropout_rate = hparams.dropout_rate
    if method =='BNN':
        model = BayesianRegressor(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, hidden_depth=hparams.hidden_depth)
    elif method == 'MCDropout':
        model = MCDropoutRegressor(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, pdrop=dropout_rate, hidden_depth=hparams.hidden_depth)
        loss_fn = F.mse_loss
    elif method == 'SVGP':
        model = MultitaskGPModel(num_tasks=output_dim, num_latents=output_dim, inducing_points=hparams.n_inducing_points)
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=output_dim).to(device)
        elbo_fn = VariationalELBO(likelihood, model, num_data=len(ds_train))
        loss_fn = lambda x, y: -elbo_fn(x, y)
    else:  # DUE or SNGP
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
        if method == 'DUE':
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
            likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=output_dim).to(device)
            elbo_fn = VariationalELBO(likelihood, model.gp, num_data=len(ds_train))
            loss_fn = lambda x, y: -elbo_fn(x, y)

        elif method == 'SNGP':

            num_gp_features = 128
            num_random_features = 1024
            normalize_gp_features = True
            feature_scale = 2
            ridge_penalty = 1

            model = Laplace(feature_extractor,
                            features,
                            num_gp_features,
                            normalize_gp_features,
                            num_random_features,
                            output_dim,
                            len(ds_train),
                            batch_size,
                            ridge_penalty=ridge_penalty,
                            feature_scale=feature_scale
                            )

            loss_fn = F.mse_loss

        else:
            raise Exception('Invalid method!')
    model.to(device)


    parameters = [
        {"params": model.parameters(), "lr": lr},
    ]

    if method == 'DUE' or method == 'SVGP':
        parameters.append({"params": likelihood.parameters(), "lr": lr})

    optimizer = optim.Adam(parameters, lr=lr)
    pbar = ProgressBar()


    def step(engine, batch):
        model.train()
        if method == 'DUE':
            likelihood.train()

        optimizer.zero_grad()

        x, y = batch
        x = x.to(device)
        y = y.to(device)
        # y.squeeze_()
        if method == 'DUE' or method == 'SNGP':
            y_pred = model(x)
            loss = loss_fn(y_pred, y)

        elif method == 'MCDropout':
            y_pred = model(x)
            loss = loss_fn(y_pred, y)

        elif method == 'BNN':
            criterion = torch.nn.MSELoss()

            complexity_cost_weight = 1 / X_train.shape[0]
            complexity_cost_weight = complexity_cost_weight \
                                     * len(dl_train) * 2**(len(dl_train)-engine.state.iteration) \
                                     / (2**(len(dl_train)-1))
            # complexity_cost_weight = complexity_cost_weight * 20 * np.log2(1/engine.state.epoch + 1)

            loss = model.sample_elbo(inputs=x,
                                     labels=y,
                                     criterion=criterion,
                                     sample_nbr=hparams.sample_nbr,
                                     complexity_cost_weight=complexity_cost_weight)
        elif method == 'SVGP':
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
        else:
            raise Exception('Invalid method!')

        loss.backward()
        optimizer.step()

        return loss.item()


    def eval_step(engine, batch):
        if method != 'MCDropout':
            model.eval()

        if method == 'DUE':
            likelihood.eval()

        x, y = batch
        x = x.to(device)
        y = y.to(device)
        # y.squeeze_()

        if method == 'DUE' or method == 'SNGP':
            y_pred = model(x)

            if method == 'DUE':  # DUE loss
                # loss = - likelihood.expected_log_prob(y, y_pred).mean()
                y_pred = likelihood(model(x))
                y_mean = y_pred.mean
                loss = F.mse_loss(y_mean, y)

            else:  # SNGP loss
                loss = loss_fn(y_pred, y)

        elif method == 'MCDropout':
            criterion = torch.nn.MSELoss()
            _, loss = model.sample_detailed_loss(inputs=x,
                                                 labels=y,
                                                 criterion=criterion,
                                                 sample_nbr=hparams.sample_nbr,)

        elif method == 'BNN':
            criterion = torch.nn.MSELoss()
            complexity_cost_weight = 1 / X_train.shape[0]
            _, _, loss, _ = model.sample_elbo_detailed_loss(inputs=x,
                                                   labels=y,
                                                   criterion=criterion,
                                                   sample_nbr=hparams.sample_nbr,
                                                   complexity_cost_weight=complexity_cost_weight)

        elif method == 'SVGP':
            y_pred = likelihood(model(x))
            y_mean = y_pred.mean
            loss = F.mse_loss(y_mean, y)

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

    if method == 'SNGP':
        @trainer.on(Events.EPOCH_STARTED)
        def reset_precision_matrix(trainer):
            model.reset_precision_matrix()

    trainer.run(dl_train, max_epochs=epochs)

    ######################################## Save #######################################

    torch.save(model.state_dict(), results_dir + "/model.pt")
    if method == 'DUE' or method == 'SVGP':
        torch.save(likelihood.state_dict(), results_dir + "/likelihood.pt")

    writer.close()

    hparams.save(results_dir + "/hparams.json")

if __name__ == "__main__":
    from supervised_hyperpara import hparams
    main(hparams)