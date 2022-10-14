import torch.nn.functional as F
import torch.nn as nn
import numpy as np

def MC_dropout(act_vec, p, mask=True):
    return F.dropout(act_vec, p=p, training=mask, inplace=True)


class MCDropoutRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, hidden_depth, pdrop):
        super(MCDropoutRegressor, self).__init__()

        self.pdrop = pdrop

        self.input_dim = input_dim
        self.output_dim = output_dim

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


    def sample_detailed_loss(self, inputs, labels, criterion, sample_nbr):
        loss = 0
        y_hat = []
        for _ in range(sample_nbr):
            outputs = self(inputs)
            y_hat.append(outputs.cpu().detach().numpy())
            loss += criterion(outputs, labels)

        return np.array(y_hat), loss / sample_nbr

