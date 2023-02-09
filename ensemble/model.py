import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from lib.utils import get_device, one_hot_embedding

def log_gaussian_loss(output, target, sigma, no_dim=1):
    exponent = -0.5 * (target - output) ** 2 / sigma ** 2
    log_coeff = -no_dim * torch.log(sigma) - 0.5 * no_dim * np.log(2 * np.pi)

    return - (log_coeff + exponent).sum()


class DNNRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, hidden_depth, learn_rate, weight_decay, loss_func,
                 hetero_noise_est):
        super(DNNRegressor, self).__init__()

        self.loss_func = loss_func

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hetero_noise_est = hetero_noise_est
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        # self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for i in range(hidden_depth)]
        )
        if hetero_noise_est:
            self.output_layer = nn.Linear(hidden_dim, 2 * output_dim)
        else:
            self.output_layer = nn.Linear(hidden_dim, output_dim)

        self.act = nn.ReLU(inplace=True)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learn_rate, weight_decay=weight_decay)

    def forward(self, x, sample=True):
        mask = self.training or sample  # if training or sampling, mc dropout will apply random binary mask
        # Otherwise, for regular test set evaluation, we can just scale activations

        x = x.view(-1, self.input_dim)  # view(batch_size, input_dim)
        # -----------------
        x = self.input_layer(x)
        # x = MC_dropout(x, p=self.pdrop, mask=mask)
        x = self.act(x)
        # -----------------
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
            # x = MC_dropout(x, p=self.pdrop, mask=mask)
            x = self.act(x)
        # -----------------
        y = self.output_layer(x)

        return y

    def get_loss(self, inputs, labels):
        outputs = self(inputs)
        if self.hetero_noise_est:
            loss = self.loss_func(outputs[:, :self.output_dim], labels, outputs[:, self.output_dim:].exp())
        else:
            loss = self.loss_func(outputs[:, :self.output_dim], labels, torch.tensor(1))
        return loss


class DNNClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, hidden_depth, learn_rate, weight_decay, loss_func):
        super(DNNClassifier, self).__init__()

        self.loss_func = loss_func

        self.input_dim = input_dim
        self.output_dim = output_dim
        # self.hetero_noise_est = hetero_noise_est
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        # self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for i in range(hidden_depth)]
        )
        self.output_layer = nn.Linear(hidden_dim, output_dim)

        self.act = nn.ReLU(inplace=True)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learn_rate, weight_decay=weight_decay)

    def forward(self, x, sample=True):
        mask = self.training or sample  # if training or sampling, mc dropout will apply random binary mask
        # Otherwise, for regular test set evaluation, we can just scale activations

        x = x.view(-1, self.input_dim)  # view(batch_size, input_dim)
        # -----------------
        x = self.input_layer(x)
        # x = MC_dropout(x, p=self.pdrop, mask=mask)
        x = self.act(x)
        # -----------------
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
            # x = MC_dropout(x, p=self.pdrop, mask=mask)
            x = self.act(x)
        # -----------------
        y = self.output_layer(x)

        return y

    def get_loss(self, inputs, labels):
        outputs = self(inputs)
        y = one_hot_embedding(labels, self.output_dim)
        loss = self.loss_func(outputs, y.float(), reduction='sum')

        return loss

class BootstrapEnsemble(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, hidden_depth, learn_rate, weight_decay, num_net,
                 hetero_noise_est=True, task='regress'):
        super(BootstrapEnsemble, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_classes = output_dim
        self.learn_rate = learn_rate
        self.weight_decay = weight_decay
        self.num_net = num_net
          # changable according to specific task
        self.task = task
        if task == 'regress':
            self.hetero_noise_est = hetero_noise_est
            self.loss_func = log_gaussian_loss
            self.net_list = [DNNRegressor(input_dim, hidden_dim, output_dim, hidden_depth,
                                 learn_rate, weight_decay, self.loss_func, hetero_noise_est) for _ in range(num_net)]
        elif task == 'classify':
            self.loss_func = F.cross_entropy
            self.net_list = [DNNClassifier(input_dim, hidden_dim, output_dim, hidden_depth,
                                          learn_rate, weight_decay, self.loss_func) for _ in range(num_net)]

    def forward(self, inputs):
        means = []
        for net in self.net_list:
            outputs = net(inputs)
            means.append(outputs[:, :self.output_dim][..., None])
        means = torch.cat(means, dim=-1)
        mean = means.mean(dim=-1)
        return mean

    def sample_batch_idx(self, batch_len, seed, ratio):
        np.random.seed(seed)
        idx_list = []
        for i in range(self.num_net):
            _idx = np.random.choice(np.arange(0, batch_len), size=(int(batch_len * ratio),), replace=True)
            idx_list.append(_idx)
        return idx_list

    def to(self, device):
        for net in self.net_list:
            net.to(device)

    def train(self, mode=True):
        for net in self.net_list:
            net.train()

    def eval(self):
        for net in self.net_list:
            net.eval()

    def save(self, dir):
        for i, net in enumerate(self.net_list):
            torch.save(net.state_dict(), dir + "/model{}.pt".format(i))

    def load(self, dir):
        for i, net in enumerate(self.net_list):
            state_dict = torch.load(dir + '/model{}.pt'.format(i))
            net.load_state_dict(state_dict)

    def sample_detailed_loss(self, inputs, labels):
        if self.task == 'regress':
            means, noise_stds = [], []

            for net in self.net_list:
                outputs = net(inputs)
                means.append(outputs[:, :self.output_dim][..., None])
                if self.hetero_noise_est:
                    noise_stds.append(outputs[:, self.output_dim:].exp()[..., None])

            means = torch.cat(means, dim=-1)
            mean = means.mean(dim=-1)
            if self.hetero_noise_est:
                noise_stds = torch.cat(noise_stds, dim=-1)

            std = means.var(dim=-1)
            if self.hetero_noise_est:
                std = (std + noise_stds.mean(dim=-1) ** 2) ** 0.5

            # std = means.var(dim=-1)
            if self.hetero_noise_est:
                loss = self.loss_func(mean, labels, noise_stds.mean(dim=-1))
            else:
                loss = self.loss_func(mean, labels, torch.tensor(1))

            mse = ((mean - labels) ** 2).mean()

            return np.array(mean.detach().cpu()), \
                   np.array(std.detach().cpu()), \
                   loss.detach().cpu(), mse.detach().cpu()

        elif self.task == 'classify':
            means = []
            for net in self.net_list:
                outputs = net(inputs)
                means.append(outputs[..., None])

            means = torch.cat(means, dim=-1)
            probs = F.softmax(means, dim=1)
            prob = probs.mean(dim=-1)
            mean = means.mean(dim=-1)

            _, preds = torch.max(mean, 1)
            # prob = F.softmax(mean, dim=1)
            y = one_hot_embedding(labels, self.output_dim)
            loss = self.loss_func(mean, y.float(), reduction='sum')

            match = torch.reshape(torch.eq(preds, labels).float(), (-1, 1))
            acc = torch.mean(match)

            return np.array(mean.detach().cpu()), \
                np.array(prob.detach().cpu()), \
                loss.detach().cpu(), acc.detach().cpu()

    def get_accuracy_matrix(self, inputs, labels):
        means, noise_stds = [], []

        for _ in range(1):
            outputs = self(inputs)
            means.append(outputs[:, :self.output_dim][..., None])

        means = torch.cat(means, dim=-1)
        mean = means.mean(dim=-1)

        mse = ((mean - labels) ** 2).mean()
        mape = ((mean - labels) / (labels+1e-10)).abs().mean()

        return mse.detach().cpu(), mape.detach().cpu()
