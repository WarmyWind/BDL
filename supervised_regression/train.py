import torch.optim as optim
from torch.utils.tensorboard.writer import SummaryWriter
# import torch_geometric
import numpy as np

from ignite.engine import Events, Engine
from ignite.metrics import Average, Loss
from ignite.contrib.handlers import ProgressBar

from Dataset.load_dataset import *
from utils import build_model, get_dataloader

def main(hparams):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dl_train, dl_valid, dl_test = get_dataloader(hparams)
    model = build_model(hparams)
    model.to(device)

    method = hparams.method
    lr = hparams.learning_rate
    epochs = hparams.epochs
    results_dir = hparams.output_dir
    writer = SummaryWriter(log_dir=str(results_dir))
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
            total_iteration = len(dl_train)
            complexity_cost_weight = 2**(total_iteration-engine.state.iteration) / (2**total_iteration-1)
            loss = model.get_loss(inputs=x,
                                labels=y,
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