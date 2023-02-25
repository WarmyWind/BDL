import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from lib.utils import get_device, one_hot_embedding


def log_gaussian_loss(output, target, sigma, no_dim=1):
    target = target.view(output.shape)
    exponent = -0.5 * (target - output) ** 2 / (sigma ** 2 + 1e-12)
    log_coeff = -no_dim * torch.log(sigma) - 0.5 * no_dim * np.log(2 * np.pi)

    return - (log_coeff + exponent).sum()


def MC_dropout(act_vec, p, mask=True):
    return F.dropout(act_vec, p=p, training=mask, inplace=True)


class MCDropoutRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, hidden_depth, pdrop, hetero_noise_est):
        super(MCDropoutRegressor, self).__init__()

        self.pdrop = pdrop
        self.loss_func = log_gaussian_loss

        self.hetero_noise_est = hetero_noise_est
        self.input_dim = input_dim
        self.output_dim = output_dim

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
        # self.act = nn.PReLU()



    def forward(self, x, sample=True):
        mask = self.training or sample  # if training or sampling, mc dropout will apply random binary mask
        # Otherwise, for regular test set evaluation, we can just scale activations

        x = x.view(-1, self.input_dim)  # view(batch_size, input_dim)
        # -----------------
        x = self.input_layer(x)
        x = MC_dropout(x, p=self.pdrop, mask=mask)
        x = self.act(x)
        # -----------------
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
            x = MC_dropout(x, p=self.pdrop, mask=mask)
            x = self.act(x)
        # -----------------
        y = self.output_layer(x)

        return y

    def get_loss(self, inputs, labels):
        outputs = self(inputs)
        labels = labels.view(labels.shape[0], -1)
        if self.hetero_noise_est:
            loss = self.loss_func(outputs[:, :self.output_dim], labels, outputs[:, self.output_dim:].exp())
        else:
            loss = self.loss_func(outputs[:, :self.output_dim], labels, torch.tensor(1))
        return loss


    def sample_detailed_loss(self, inputs, labels, sample_nbr):
        # loss = 0
        # y_hat = []
        means, noise_stds = [], []

        for _ in range(sample_nbr):
            outputs = self(inputs)
            means.append(outputs[:, :self.output_dim][..., None])
            if self.hetero_noise_est:
                noise_stds.append(outputs[:, self.output_dim:].exp()[..., None])

        means = torch.cat(means, dim=-1)
        mean = means.mean(dim=-1)
        if self.hetero_noise_est:
            noise_stds = torch.cat(noise_stds, dim=-1)
            alea = noise_stds.mean(dim=-1)
            std = (means.std(dim=-1) ** 2 + alea ** 2) ** 0.5

        else:
            std = means.std(dim=-1)
            alea = torch.zeros_like(std).to(get_device())


        # std = means.var(dim=-1)
        if self.hetero_noise_est:
            loss = self.loss_func(mean, labels, noise_stds.mean(dim=-1))
        else:
            loss = self.loss_func(mean, labels, torch.tensor(1))

        mse = ((mean - labels.view(mean.shape)) ** 2).mean()

        return np.array(mean.detach().cpu()), \
               np.array(std.detach().cpu()), \
               np.array(alea.detach().cpu()), \
               loss.detach().cpu(), mse.detach().cpu()

    def get_accuracy_matrix(self, inputs, labels, sample_nbr):
        means, noise_stds = [], []

        for _ in range(sample_nbr):
            outputs = self(inputs)
            means.append(outputs[:, :self.output_dim][..., None])

        means = torch.cat(means, dim=-1)
        mean = means.mean(dim=-1)

        labels = labels.view(mean.shape)
        mse = ((mean - labels) ** 2).mean()
        mape = ((mean - labels) / (labels+1e-10)).abs().mean()

        return mse.detach().cpu(), mape.detach().cpu()


    def save(self, dir):
        checkpoint = {'model': self.state_dict(), 'norm_mean':self.norm_mean, 'norm_std':self.norm_std}
        torch.save(checkpoint, dir)

    def load(self, dir):
        checkpoint = torch.load(dir + '/model.pt')
        self.load_state_dict(checkpoint['model'])
        self.norm_mean = checkpoint['norm_mean']
        self.norm_std = checkpoint['norm_std']


class MCDropoutClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, hidden_depth, pdrop):
        super(MCDropoutClassifier, self).__init__()

        self.pdrop = pdrop
        self.loss_func = F.cross_entropy

        self.input_dim = input_dim
        self.num_classes = output_dim

        self.input_layer = nn.Linear(input_dim, hidden_dim)
        # self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for i in range(hidden_depth)]
        )

        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.act = nn.ReLU(inplace=True)


    def forward(self, x, sample=True):
        mask = self.training or sample  # if training or sampling, mc dropout will apply random binary mask
        # Otherwise, for regular test set evaluation, we can just scale activations

        x = x.view(-1, self.input_dim)  # view(batch_size, input_dim)
        # -----------------
        x = self.input_layer(x)
        x = MC_dropout(x, p=self.pdrop, mask=mask)
        x = self.act(x)
        # -----------------
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
            x = MC_dropout(x, p=self.pdrop, mask=mask)
            x = self.act(x)
        # -----------------
        y = self.output_layer(x)

        return y

    def get_loss(self, inputs, labels):
        outputs = self(inputs)
        y = one_hot_embedding(labels, self.num_classes)
        loss = self.loss_func(outputs, y.float(), reduction='sum')

        return loss


    def sample_detailed_loss(self, inputs, labels, sample_nbr):
        means = []
        for _i in range(sample_nbr):
            outputs = self(inputs)
            # y = one_hot_embedding(labels, self.num_classes)
            # _, preds = torch.max(outputs, 1)
            means.append(outputs[..., None])


        means = torch.cat(means, dim=-1)
        probs = F.softmax(means, dim=1)
        prob = probs.mean(dim=-1)
        mean = means.mean(dim=-1)

        _, preds = torch.max(mean, 1)
        # prob = F.softmax(mean, dim=1)
        y = one_hot_embedding(labels, self.num_classes)
        loss = self.loss_func(mean, y.float(), reduction='sum')

        match = torch.reshape(torch.eq(preds, labels).float(), (-1, 1))
        acc = torch.mean(match)

        return np.array(mean.detach().cpu()), \
            np.array(prob.detach().cpu()), \
            loss.detach().cpu(), acc.detach().cpu()

    # def save(self, dir):
    #     checkpoint = {'model': self.state_dict(), 'norm_mean':self.norm_mean, 'norm_std':self.norm_std}
    #     torch.save(checkpoint, dir)
    #
    # def load(self, dir):
    #     checkpoint = torch.load(dir)
    #     self.load_state_dict(checkpoint['model'])
    #     self.norm_mean = checkpoint['norm_mean']
    #     self.norm_std = checkpoint['norm_std']