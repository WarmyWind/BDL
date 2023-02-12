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
from Dataset.load_dataset import *
from utils import build_model
from cnn.model import LeNet5
from dnn.model import DNNClassifier, DNNBinaryClassifier
from bnn.model import BayesianClassifier
from mcdropout.model import MCDropoutClassifier
from ensemble.model import BootstrapEnsemble
from edl.model import EvidentialClassifier
from edl.loss.discrete import edl_digamma_loss, edl_log_loss, edl_mse_loss


def main(hparams):
    print("Use CUDA: ", torch.cuda.is_available())
    if hparams.task == 'HO_predict':
        dl_train, dl_valid, dl_test = load_handover(hparams.dataset_path, hparams.input_dim, hparams.output_dim,
                                                         test_size=0.2, valid_size=0.2,
                                                         train_batch_size=hparams.batch_size,
                                                         valid_batch_size=hparams.batch_size, seed=hparams.seed)

    elif hparams.task == 'cifar10':
        dl_train, dl_valid, dl_test = load_cifar10(valid_size=0.2, train_batch_size=hparams.batch_size,
                                                   seed=hparams.seed)
        classes = ('plane', 'car', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck')
    else:
        raise Exception('Invalid task!')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    method = hparams.method
    lr = hparams.learning_rate
    epochs = hparams.epochs
    results_dir = hparams.output_dir
    writer = SummaryWriter(log_dir=str(results_dir))

    model = build_model(hparams)
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

        if method == 'MCDropout':
            loss = model.get_loss(x, y)
            loss.backward()
            optimizer.step()

        elif method == 'BNN':
            total_iteration = len(dl_train)
            complexity_cost_weight = 2**(total_iteration-engine.state.iteration) / (2**total_iteration-1)

            loss = model.get_loss(inputs=x,
                                labels=y,
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
            loss = model.get_loss(x, y, engine.state.epoch)
            loss.backward()
            optimizer.step()

        elif method == 'CNN':
            loss = model.get_loss(x, y)
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
            complexity_cost_weight = 1 / len(dl_train)
            _, _, _, acc = model.sample_detailed_loss(inputs=x,
                                                 labels=y,
                                                 sample_nbr=hparams.sample_nbr,
                                                 complexity_cost_weight=complexity_cost_weight)

        elif method == 'Ensemble':
            _, _, _, acc = model.sample_detailed_loss(inputs=x, labels=y)

        elif method == 'EDL':
            _, _, _, acc = model.sample_detailed_loss(inputs=x, labels=y, epoch=engine.state.epoch)
        elif method == 'CNN':
            _, _, _, acc = model.sample_detailed_loss(inputs=x, labels=y)
        elif method == 'DNN':
            _, _, _, acc = model.sample_detailed_loss(inputs=x, labels=y)
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