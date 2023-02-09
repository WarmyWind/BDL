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

from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from dnn.model import DNNClassifier, DNNBinaryClassifier
from bnn.model import BayesianClassifier
from mcdropout.model import MCDropoutClassifier
from ensemble.model import BootstrapEnsemble
from edl.model import EvidentialClassifier
from edl.loss.discrete import edl_digamma_loss, edl_log_loss, edl_mse_loss
print(torch.cuda.is_available())
def main(hparams):
    if hparams.task == 'HO_predict':
        Data_set = np.load(hparams.dataset_path, allow_pickle=True).tolist()
        X = Data_set['x']
        y = Data_set['y']
        X = X[:, -hparams.input_dim:]
        y = y[:, :hparams.output_dim]
        X = StandardScaler().fit_transform(X)
        y = y.flatten()
        # y = StandardScaler().fit_transform(y)

        # 80% data for training process, 20% for model calibration
        X, X_calibration, y, y_calibration = train_test_split(X, y, test_size=.2, random_state=hparams.seed)

        # In the 80%, 70% for training/evaluating, 10% for testing
        X, X_test, y, y_test = train_test_split(X, y, test_size=.1 / 0.8, random_state=hparams.seed)

        # In the 70%, 60% for training, 10% for validation
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=.1 / 0.7, random_state=hparams.seed)

        X_train, y_train = torch.tensor(X_train).float(), torch.tensor(y_train).float()
        X_valid, y_valid = torch.tensor(X_valid).float(), torch.tensor(y_valid).float()
        X_test, y_test = torch.tensor(X_test).float(), torch.tensor(y_test).float()

        ds_train = torch.utils.data.TensorDataset(X_train, y_train)
        dl_train = torch.utils.data.DataLoader(ds_train, batch_size=hparams.batch_size, shuffle=True)
        ds_valid = torch.utils.data.TensorDataset(X_valid, y_valid)
        dl_valid = torch.utils.data.DataLoader(ds_valid, batch_size=512, shuffle=True)
        ds_test = torch.utils.data.TensorDataset(X_test, y_test)
        dl_test = torch.utils.data.DataLoader(ds_test, batch_size=len(ds_test))
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
        model = BayesianClassifier(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
                                  hidden_depth=hparams.hidden_depth)
        criterion = F.cross_entropy

    elif method == 'MCDropout':
        model = MCDropoutClassifier(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
                                   pdrop=dropout_rate, hidden_depth=hparams.hidden_depth)

        # loss_fn = F.mse_loss
    elif method == 'Ensemble':
        num_net = hparams.num_net

        model = BootstrapEnsemble(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
                                  hidden_depth=hparams.hidden_depth, learn_rate=lr, weight_decay=1e-4,
                                  num_net=num_net, task='classify')

    elif method == 'EDL':
        model = EvidentialClassifier(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
                                     hidden_depth=hparams.hidden_depth, annealing_step=hparams.annealing_step)
        if hparams.edl_loss == 'digamma':
            criterion = edl_digamma_loss
        elif hparams.edl_loss == 'log':
            criterion = edl_log_loss
        elif hparams.edl_loss == 'mse':
            criterion = edl_mse_loss

    elif method == 'DNN':
        if not hparams.binary_classifier:
            loss_func = F.cross_entropy
            model = DNNClassifier(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
                                  hidden_depth=hparams.hidden_depth, learn_rate=lr, weight_decay=1e-4,
                                  loss_func=loss_func)
        else:
            loss_func = nn.BCEWithLogitsLoss()
            model = DNNBinaryClassifier(input_dim=input_dim, hidden_dim=hidden_dim, hidden_depth=hparams.hidden_depth,
                                        learn_rate=lr, weight_decay=1e-4, loss_func=loss_func)

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
            loss = model.get_loss(x, y)
            loss.backward()
            optimizer.step()

        elif method == 'BNN':
            # complexity_cost_weight = 1 / X_train.shape[0]
            # complexity_cost_weight = complexity_cost_weight \
            #                          * len(dl_train) * 2 ** (len(dl_train) - engine.state.iteration) \
            #                          / (2 ** (len(dl_train) - 1))

            total_iteration = len(dl_train)
            complexity_cost_weight = 2**(total_iteration-engine.state.iteration) / (2**total_iteration-1)

            # complexity_cost_weight = x.shape[0] / X_train.shape[0]

            loss = model.get_loss(inputs=x,
                                labels=y,
                                criterion=criterion,
                                sample_nbr=hparams.sample_nbr,
                                complexity_cost_weight=complexity_cost_weight)
            loss.backward()
            optimizer.step()

        elif method == 'Ensemble':
            idx_list = model.sample_batch_idx(len(x), hparams.seed, ratio=hparams.dataset_ratio)
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
            # outputs = model(x)
            loss = model.get_loss(x, y, criterion, engine.state.epoch)
            loss.backward()
            optimizer.step()

        elif method == 'DNN':
            loss = model.get_loss(x, y)
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
            _, _, _, acc = model.sample_detailed_loss(inputs=x,
                                                 labels=y,
                                                 sample_nbr=hparams.sample_nbr)

        elif method == 'BNN':
            complexity_cost_weight = 1 / X_train.shape[0]
            _, _, _, acc = model.sample_detailed_loss(inputs=x,
                                                 labels=y,
                                                 criterion=criterion,
                                                 sample_nbr=hparams.sample_nbr,
                                                 complexity_cost_weight=complexity_cost_weight)

        elif method == 'Ensemble':
            _, _, _, acc = model.sample_detailed_loss(inputs=x, labels=y)

        elif method == 'EDL':
            _, _, _, acc = model.sample_detailed_loss(inputs=x, labels=y, criterion=criterion, epoch=engine.state.epoch)

        elif method == 'DNN':
            _, _, acc = model.sample_detailed_loss(inputs=x, labels=y)
        else:
            raise Exception('Invalid method!')

        return acc


    trainer = Engine(step)
    evaluator = Engine(eval_step)

    metric = Average()
    metric.attach(trainer, "loss")
    pbar.attach(trainer)

    metric = Average()
    metric.attach(evaluator, "acc")

    best_acc = 0
    @trainer.on(Events.EPOCH_COMPLETED(every=int(1)))
    def log_results(trainer):
        evaluator.run(dl_valid)
        acc_valid = evaluator.state.metrics['acc']
        print(f"Results - Epoch: {trainer.state.epoch} - "
              f"Valid Acc: {acc_valid:.4f} - "
              f"Train Loss: {trainer.state.metrics['loss']:.4f}")

        nonlocal best_acc, method
        if acc_valid > best_acc:
            print("\033[5;33mBest acc before:{:.4f}, now:{:.4f} Saving model\033[0m".format(best_acc, acc_valid))
            best_acc = acc_valid
            if method != 'Ensemble':
                torch.save(model.state_dict(), results_dir + "/model.pt")
            else:
                model.save(results_dir)


        writer.add_scalar("Loss/train", trainer.state.metrics['loss'], trainer.state.epoch)
        writer.add_scalar("Loss/eval", evaluator.state.metrics['acc'], trainer.state.epoch)


    trainer.run(dl_train, max_epochs=epochs)

    ######################################## Save #######################################

    # torch.save(model.state_dict(), results_dir + "/model.pt")

    writer.close()
    hparams.save(results_dir + "/hparams.json")

if __name__ == "__main__":
    from hyperpara import hparams
    main(hparams)