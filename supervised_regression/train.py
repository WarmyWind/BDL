import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard.writer import SummaryWriter
# import torch_geometric
import numpy as np

from ignite.engine import Events, Engine
from ignite.metrics import Average, Loss
from ignite.contrib.handlers import ProgressBar

from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from bnn.model import BayesianRegressor
from mcdropout.model import MCDropoutRegressor, log_gaussian_loss
from ensemble.model import BootstrapEnsemble
from edl.model import EvidentialRegressor

from Dataset.load_dataset import *

def main(hparams):
    if hparams.task == 'large_channel_predict':
        dl_train, dl_valid, dl_test = load_large_channel(hparams.dataset_path, hparams.input_dim, hparams.output_dim,
                                                         test_size=0.2, valid_size=0.2, train_batch_size=hparams.batch_size,
                                                         valid_batch_size=hparams.batch_size, seed=hparams.seed)
    elif hparams.task == 'uci_wine':
        dl_train, dl_valid, dl_test = load_uci_wine(test_size=0.25, valid_size=0,
                                                    train_batch_size=hparams.batch_size,
                                                    valid_batch_size=hparams.batch_size, seed=hparams.seed)
    elif hparams.task == 'uci_crime':
        dl_train, dl_valid, dl_test = load_uci_crime(test_size=0.25, valid_size=0,
                                                     train_batch_size=hparams.batch_size,
                                                     valid_batch_size=hparams.batch_size, seed=hparams.seed)
    else:
        raise Exception('Invalid task!')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    method = hparams.method  # 'BNN' , 'MCDropout'

    lr = hparams.learning_rate
    epochs = hparams.epochs
    results_dir = hparams.output_dir
    writer = SummaryWriter(log_dir=str(results_dir))

    input_dim = hparams.input_dim
    hidden_dim = hparams.hidden_dim
    output_dim = hparams.output_dim
    dropout_rate = hparams.dropout_rate
    if method =='BNN':
        if hparams.noise_estimation == 'none':
            model = BayesianRegressor(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
                                      hidden_depth=hparams.hidden_depth, hetero_noise_est=False)
        elif hparams.noise_estimation == 'hetero':
            model = BayesianRegressor(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
                                      hidden_depth=hparams.hidden_depth, hetero_noise_est=True)
        else:
            raise Exception("Unsupported noise estimation: \"{}\"!".format(hparams.noise_estimation))
        if model.hetero_noise_est == False:
            criterion = torch.nn.MSELoss()
        else:
            criterion = log_gaussian_loss

    elif method == 'MCDropout':
        if hparams.noise_estimation == 'none':
            model = MCDropoutRegressor(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
                                       pdrop=dropout_rate, hidden_depth=hparams.hidden_depth, hetero_noise_est=False)
        elif hparams.noise_estimation == 'hetero':
            model = MCDropoutRegressor(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
                                       pdrop=dropout_rate, hidden_depth=hparams.hidden_depth, hetero_noise_est=True)
        else:
            raise Exception("Unsupported noise estimation: \"{}\"!".format(hparams.noise_estimation))
        # loss_fn = F.mse_loss
    elif method == 'Ensemble':
        num_net = hparams.num_net
        if hparams.noise_estimation == 'none':
            model = BootstrapEnsemble(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
                                      hidden_depth=hparams.hidden_depth, learn_rate=lr, weight_decay=1e-4,
                                      num_net=num_net, hetero_noise_est=False)
        elif hparams.noise_estimation == 'hetero':
            model = BootstrapEnsemble(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
                                      hidden_depth=hparams.hidden_depth, learn_rate=lr, weight_decay=1e-4,
                                      num_net=num_net, hetero_noise_est=True)
        else:
            raise Exception("Unsupported noise estimation: \"{}\"!".format(hparams.noise_estimation))
    elif method == 'EDL':
        model = EvidentialRegressor(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
                                  hidden_depth=hparams.hidden_depth)
    else:
        raise Exception('Invalid method!')

    model.to(device)



    if method != 'Ensemble':
        parameters = [
            {"params": model.parameters(), "lr": lr},
        ]
        optimizer = optim.Adam(parameters, lr=lr)
    pbar = ProgressBar()

    def step(engine, batch):
        model.train()
        if method != 'Ensemble':
            optimizer.zero_grad()

        x, y = batch
        x = x.to(device)
        y = y.to(device)
        # y.squeeze_()

        if method == 'MCDropout':
            # outputs = model(x)
            # loss = model.loss_func(outputs[:,:output_dim], y, outputs[:,output_dim:].exp())
            loss = model.get_loss(x, y)
            loss.backward()
            optimizer.step()

        elif method == 'BNN':
            # complexity_cost_weight = 1 / X_train.shape[0]
            # complexity_cost_weight = complexity_cost_weight \
            #                          * len(dl_train) * 2**(len(dl_train)-engine.state.iteration) \
            #                          / (2**(len(dl_train)-1))
            total_iteration = len(dl_train)
            complexity_cost_weight = 2**(total_iteration-engine.state.iteration) / (2**total_iteration-1)
            loss = model.get_loss(inputs=x,
                                labels=y,
                                criterion=criterion,
                                sample_nbr=hparams.sample_nbr,
                                complexity_cost_weight=complexity_cost_weight)
            loss.backward()
            optimizer.step()

        elif method == 'Ensemble':
            idx_list = model.sample_batch_idx(len(x), hparams.seed, ratio=0.2)
            loss = 0
            for ii, net in enumerate(model.net_list):
                net.optimizer.zero_grad()
                _idx = idx_list[ii]
                _x = x[_idx]
                _y = y[_idx]
                # outputs = net(_x)
                # _loss = net.loss_func(outputs[:, :output_dim], _y, outputs[:, output_dim:].exp())
                _loss = net.get_loss(_x, _y)
                _loss.backward()
                net.optimizer.step()
                loss += _loss

            loss /= model.num_net

        elif method == 'EDL':
            outputs = model(x)
            loss = model.loss_func(outputs, y)
            loss.backward()
            optimizer.step()

        else:
            raise Exception('Invalid method!')

        return loss.item()

    def eval_step(engine, batch):
        if method != 'MCDropout':
            model.eval()

        x, y = batch
        x = x.to(device)
        y = y.to(device)

        if method == 'MCDropout':
            _, _, _, mse = model.sample_detailed_loss(inputs=x,
                                                 labels=y,
                                                 sample_nbr=hparams.sample_nbr)

        elif method == 'BNN':
            complexity_cost_weight = 1 / len(dl_train)
            _, _, _, mse = model.sample_detailed_loss(inputs=x,
                                                 labels=y,
                                                 criterion=criterion,
                                                 sample_nbr=hparams.sample_nbr,
                                                 complexity_cost_weight=complexity_cost_weight)

        elif method == 'Ensemble':
            _, _, _, mse = model.sample_detailed_loss(inputs=x, labels=y)

        elif method == 'EDL':
            _, _, _, mse = model.sample_detailed_loss(inputs=x, labels=y)

        else:
            raise Exception('Invalid method!')

        rmse = mse ** 0.5
        return rmse


    trainer = Engine(step)
    evaluator = Engine(eval_step)

    metric = Average()
    metric.attach(trainer, "loss")
    pbar.attach(trainer)

    metric = Average()
    metric.attach(evaluator, "RMSE")

    best_rmse = np.inf
    @trainer.on(Events.EPOCH_COMPLETED(every=int(1)))
    def log_results(trainer):
        evaluator.run(dl_valid)
        rmse_valid = evaluator.state.metrics['RMSE']
        print(f"Results - Epoch: {trainer.state.epoch} - "
              f"Valid RMSE loss: {rmse_valid:.4f} - "
              f"Train Loss: {trainer.state.metrics['loss']:.4f}")

        nonlocal best_rmse, method
        if rmse_valid < best_rmse:
            print("\033[5;33mBest rmse before:{:.4f}, now:{:.4f} Saving model\033[0m".format(best_rmse, rmse_valid))
            best_rmse = rmse_valid
            if method != 'Ensemble':
                torch.save(model.state_dict(), results_dir + "/model.pt")
            else:
                model.save(results_dir)


        writer.add_scalar("Loss/train", trainer.state.metrics['loss'], trainer.state.epoch)
        writer.add_scalar("Loss/eval", evaluator.state.metrics['RMSE'], trainer.state.epoch)


    trainer.run(dl_train, max_epochs=epochs)

    ######################################## Save #######################################

    # torch.save(model.state_dict(), results_dir + "/model.pt")

    writer.close()
    hparams.save(results_dir + "/hparams.json")

if __name__ == "__main__":
    from hyperpara import hparams
    main(hparams)