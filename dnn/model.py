import torch
from torch.nn import Linear
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from lib.utils import get_device, one_hot_embedding

class DNNBinaryClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_depth, learn_rate, weight_decay, loss_func):
        super(DNNBinaryClassifier, self).__init__()

        self.loss_func = loss_func
        self.input_dim = input_dim
        self.output_dim = 1
        self.num_classes = 2
        # self.hetero_noise_est = hetero_noise_est
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        # self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for i in range(hidden_depth)]
        )
        self.output_layer = nn.Linear(hidden_dim, self.output_dim)

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
        # y = one_hot_embedding(labels, self.output_dim)
        y = labels[..., None]
        loss = self.loss_func(outputs, y.float())

        return loss

    def sample_detailed_loss(self, inputs, labels, sample_nbr=1):
        means = []
        for _i in range(sample_nbr):
            outputs = self(inputs)
            # y = one_hot_embedding(labels, self.num_classes)
            # _, preds = torch.max(outputs, 1)
            means.append(outputs[..., None])

        means = torch.cat(means, dim=-1)
        mean = means.mean(dim=-1)

        preds = torch.squeeze(mean >= 0)
        probs = F.sigmoid(mean)
        probs = torch.concatenate([1-probs, probs], dim=1)
        # y = one_hot_embedding(labels, self.num_classes)
        # loss = self.model.loss_func(mean, y.float(), reduction='sum')

        match = torch.reshape(torch.eq(preds, labels).float(), (-1, 1))
        acc = torch.mean(match)

        return np.array(mean.detach().cpu()), \
            np.array(probs.detach().cpu()), \
            acc.detach().cpu()

class DNNClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, hidden_depth, learn_rate, weight_decay, loss_func):
        super(DNNClassifier, self).__init__()

        self.loss_func = loss_func
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_classes = output_dim
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

    def sample_detailed_loss(self, inputs, labels, sample_nbr=1):
        means = []
        for _i in range(sample_nbr):
            outputs = self(inputs)
            # y = one_hot_embedding(labels, self.num_classes)
            # _, preds = torch.max(outputs, 1)
            means.append(outputs[..., None])

        means = torch.cat(means, dim=-1)
        mean = means.mean(dim=-1)

        _, preds = torch.max(mean, 1)
        probs = F.softmax(mean, dim=1)
        y = one_hot_embedding(labels, self.num_classes)
        # loss = self.model.loss_func(mean, y.float(), reduction='sum')

        match = torch.reshape(torch.eq(preds, labels).float(), (-1, 1))
        acc = torch.mean(match)

        return np.array(mean.detach().cpu()), \
            np.array(probs.detach().cpu()), \
            acc.detach().cpu()