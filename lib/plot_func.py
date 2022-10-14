import numpy as np
from matplotlib import pyplot as plt


def plot_uncertainty(x, y, y_hat, std):
    x, y, y_hat, std = x.squeeze(), y.squeeze(), y_hat.squeeze(), std.squeeze()

    f, ax = plt.subplots(1, 1, figsize=(4, 3))
    # Get upper and lower confidence bounds
    lower, upper = y_hat - std*3, y_hat + std*3
    # Plot training data as black stars

    ax.plot(np.arange(len(x)+len(y)), np.concatenate([x, y]), 'k*-')
    # Plot predictive means as blue line
    ax.plot(np.arange(len(x), len(x)+len(y)), y_hat, 'b')
    # Shade between the lower and upper confidence bounds
    ax.fill_between(np.arange(len(x), len(x)+len(y)), lower, upper, alpha=0.5)
    # ax.set_ylim([-3, 3])
    ax.legend(['True Data', 'Prediction', 'Uncertainty'])

    return f, ax


def plot_bar(data, xlabel, ylabel, para_list, label_list, ax=None):
    if ax == None:
        fig, ax = plt.subplots()

    # 创建分组柱状图，需要自己控制x轴坐标
    xticks = np.arange(len(para_list))

    # 注意控制柱子的宽度，这里选择0.25
    for i in range(len(label_list)):
        ax.bar(xticks + i * 0.25, data[i], width=0.25, label=label_list[i])

    if xlabel != '':
        ax.set_xlabel(xlabel)
    if ylabel != '':
        ax.set_ylabel(ylabel)
    ax.legend()

    # 最后调整x轴标签的位置
    ax.set_xticks(xticks + 0.25*(len(label_list)-1)/2)
    ax.set_xticklabels(para_list)

    return ax