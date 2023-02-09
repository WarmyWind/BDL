import numpy as np
from matplotlib import pyplot as plt

def loss_divided_by_uncertainty(y_std, loss, low_uncertainty_ratio, high_uncertainty_ratio):
    try:
        loss = loss.cpu().detach().numpy()
    except:
        pass

    if len(y_std.shape) == 1:
        y_std = y_std[:, np.newaxis]
        loss = loss[:, np.newaxis]

    idx = np.argsort(y_std, axis=0)
    low_uncertainty_num = int(y_std.shape[0] * low_uncertainty_ratio)
    high_uncertainty_num = int(y_std.shape[0] * high_uncertainty_ratio)

    low_uncertainty_loss = np.array([loss[idx[:low_uncertainty_num,i], i] for i in range(idx.shape[1])]).swapaxes(0,1)
    high_uncertainty_loss = np.array([loss[idx[-high_uncertainty_num:, i], i] for i in range(idx.shape[1])]).swapaxes(0, 1)
    # return np.mean(low_uncertainty_loss, axis=0), np.mean(high_uncertainty_loss, axis=0)
    return low_uncertainty_loss, high_uncertainty_loss

def loss_divided_by_MCoutput(y_hat, loss, low_uncertainty_ratio, high_uncertainty_ratio):
    y_std = np.std(y_hat, axis=0)

    return loss_divided_by_uncertainty(y_std, loss, low_uncertainty_ratio, high_uncertainty_ratio)




