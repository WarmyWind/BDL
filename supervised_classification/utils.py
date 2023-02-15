from cnn.model import LeNet5
from dnn.model import DNNClassifier, DNNBinaryClassifier
from bnn.model import BayesianClassifier, BayesianLeNet5
from mcdropout.model import MCDropoutClassifier
from ensemble.model import BootstrapEnsemble
from edl.model import EvidentialClassifier
from edl.loss.discrete import edl_digamma_loss, edl_log_loss, edl_mse_loss
import torch.nn as nn
import torch.nn.functional as F

def build_model(hparams):
    method = hparams.method
    task = hparams.task
    lr = hparams.learning_rate
    input_dim = hparams.input_dim
    hidden_dim = hparams.hidden_dim
    output_dim = hparams.output_dim
    dropout_rate = hparams.dropout_rate
    if method == 'BNN':
        if task != 'cifar10':
            criterion = F.cross_entropy
            model = BayesianClassifier(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
                                   hidden_depth=hparams.hidden_depth, criterion=criterion)
        else:
            model = BayesianLeNet5()

    elif method == 'MCDropout':
        model = MCDropoutClassifier(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
                                    pdrop=dropout_rate, hidden_depth=hparams.hidden_depth)

    elif method == 'Ensemble':
        num_net = hparams.num_net
        if hparams.task == 'cifar10':
            task = 'cifar10'
        else:
            task = 'classify'
        model = BootstrapEnsemble(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
                                  hidden_depth=hparams.hidden_depth, learn_rate=lr, weight_decay=1e-4,
                                  num_net=num_net, task=task)

    elif method == 'EDL':
        if hparams.edl_loss == 'digamma':
            criterion = edl_digamma_loss
        elif hparams.edl_loss == 'log':
            criterion = edl_log_loss
        elif hparams.edl_loss == 'mse':
            criterion = edl_mse_loss
        else:
            raise Exception("Invalid EDL loss!")
        model = EvidentialClassifier(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
                                     hidden_depth=hparams.hidden_depth, annealing_step=hparams.annealing_step,
                                     criterion=criterion)

    elif method == 'CNN':
        if hparams.task == 'cifar10' or 'mnist':
            model = LeNet5(task=hparams.task)
        else:
            raise Exception('Now CNN only implements LeNet5 for cifar10 and MNIST')

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

    return model

