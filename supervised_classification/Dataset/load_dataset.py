import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_wine
from sklearn.feature_selection import RFE, RFECV
import torch
import torchvision
import torchvision.transforms as transforms

def split_data_set(X, y, test_size, valid_size, train_batch_size, valid_batch_size, seed, norm, feature_selector='none'):
    X, X_test, y, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    if valid_size != 0:
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=valid_size / (1 - test_size), random_state=seed)
    else:
        X_train, y_train = X, y
        X_valid, y_valid = X_train, y_train
    if norm == True:
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_valid = sc.transform(X_valid)
        X_test = sc.transform(X_test)

    X_train, y_train = torch.tensor(X_train).float(), torch.tensor(y_train).float()
    X_valid, y_valid = torch.tensor(X_valid).float(), torch.tensor(y_valid).float()
    X_test, y_test = torch.tensor(X_test).float(), torch.tensor(y_test).float()

    ds_train = torch.utils.data.TensorDataset(X_train, y_train)
    train_batch_size = train_batch_size if train_batch_size < len(ds_train) else len(ds_train)
    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=train_batch_size, shuffle=True)

    ds_valid = torch.utils.data.TensorDataset(X_valid, y_valid)
    valid_batch_size = valid_batch_size if valid_batch_size < len(ds_valid) else len(ds_valid)
    dl_valid = torch.utils.data.DataLoader(ds_valid, batch_size=valid_batch_size, shuffle=True)

    ds_test = torch.utils.data.TensorDataset(X_test, y_test)
    dl_test = torch.utils.data.DataLoader(ds_test, batch_size=len(ds_test))

    return dl_train, dl_valid, dl_test


def load_handover(dataset_path, input_dim, output_dim, test_size, valid_size, train_batch_size, valid_batch_size, seed=42, norm=True):
    Data_set = np.load(dataset_path, allow_pickle=True).tolist()
    X = Data_set['x']
    y = Data_set['y']
    X = X[:, -input_dim:]
    y = y[:, :output_dim]
    y = y.flatten()
    # if norm == True:
    #     X = StandardScaler().fit_transform(X)

    # Split dataset into train, valid and test set
    return split_data_set(X, y, test_size, valid_size, train_batch_size, valid_batch_size, seed, norm)


def load_cifar10(valid_size, train_batch_size, seed, cali_dataset_num=-1):
    # classes = ('plane', 'car', 'bird', 'cat', 'deer',
    #            'dog', 'frog', 'horse', 'ship', 'truck')
    """
    Load Cifar10 Dataset
    """
    # Use the Google recommended mean and std for cifar10
    CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
    CIFAR10_STD = (0.2470, 0.2435, 0.2616)  # instead of (0.2023, 0.1994, 0.2010)

    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./Dataset', train=True, download=True, transform=transform_train)

    torch.manual_seed(seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # 为GPU设置随机种子

    # 随机划分训练集和验证集
    if valid_size!=0:
        train_dataset, valid_dataset = torch.utils.data.random_split(trainset,
                                                                 [int((1-valid_size)*50000), int(valid_size*50000)],
                                                                 )
        trainloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=2)
        validloader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=train_batch_size, shuffle=False, num_workers=2)
    else:
        train_dataset = trainset
        trainloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=2)
        valid_dataset = trainset
        validloader = trainloader

    testset = torchvision.datasets.CIFAR10(
        root='./Dataset', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)


    if cali_dataset_num == -1:
        return trainloader, validloader, testloader
    else:
        _, cali_dataset = torch.utils.data.random_split(valid_dataset,
                                                      [int(len(valid_dataset)-cali_dataset_num), int(cali_dataset_num)],
                                                      )
        caliloader = torch.utils.data.DataLoader(cali_dataset, batch_size=100, shuffle=False, num_workers=2)
        return trainloader, caliloader, testloader