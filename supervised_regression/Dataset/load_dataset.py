import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_wine
from sklearn.feature_selection import RFE, RFECV
import torch

def split_data_set(X, y, test_size, valid_size, train_batch_size, valid_batch_size, seed, norm, noisy_feature=False):
    if noisy_feature == True:
        temp_X = np.random.rand(X.shape[0],X.shape[1],X.shape[2])
        temp_X[:,:,-1] = X[:,:,-1]
        X = temp_X

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
        y_train = sc.fit_transform(y_train)
        y_valid = sc.transform(y_valid)
        y_test = sc.transform(y_test)



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
    dl_test = torch.utils.data.DataLoader(ds_test, batch_size=int(len(ds_test)/5))

    return dl_train, dl_valid, dl_test


def load_uci_energy(dataset_path, input_dim, output_dim, test_size, valid_size, train_batch_size, valid_batch_size,
                    seed=42, gap=1, norm=False, noisy_feature=False):
    Data_set = np.load(dataset_path, allow_pickle=True).tolist()
    X = Data_set['x']
    y = Data_set['y']
    if len(X.shape) == 2:
        X = X[:, -input_dim*gap::gap]
    elif len(X.shape) == 3:
        input_dim = int(input_dim/X.shape[2])
        X = X[:, -input_dim*gap::gap, :]
    else:
        raise Exception('Unsupported X shape')
    y = y[..., :output_dim:gap]

    # if norm == True:
    #     X = StandardScaler().fit_transform(X)
    #     y = StandardScaler().fit_transform(y)

    # Split dataset into train, valid and test set
    return split_data_set(X, y, test_size, valid_size, train_batch_size, valid_batch_size, seed, norm, noisy_feature)


def load_uci_crime(test_size, valid_size, train_batch_size, valid_batch_size, seed=42, norm=True):
    '''
    Load UCI crime dataset, for regression with 100 inputs and 1 output
    '''
    attrib = read_csv('Dataset/uci_crime/attributes.csv', delim_whitespace=True)
    data = read_csv('Dataset/uci_crime/communities.data', names=attrib['attributes'])

    # Remove non-predictive features
    data = data.drop(columns=['state', 'county',
                              'community', 'communityname',
                              'fold'], axis=1)

    data = data.replace('?', np.nan)
    feat_miss = data.columns[data.isnull().any()]

    # 'OtherPerCap' has only one missing value
    # Impute mean values
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputer = imputer.fit(data[['OtherPerCap']])
    data[['OtherPerCap']] = imputer.transform(data[['OtherPerCap']])

    # Drop other attributes with missing value
    data = data.dropna(axis=1)
    # print(data.shape)

    X = data.iloc[:, 0:100].values
    y = data.iloc[:, 100].values[:, np.newaxis]
    # print(X.shape)
    # print(y.shape)

    # Split dataset into train, valid and test set
    return split_data_set(X, y, test_size, valid_size, train_batch_size, valid_batch_size, seed, norm)


def load_uci_wine(test_size, valid_size, train_batch_size, valid_batch_size, seed=42, norm=True):
    '''
    Load UCI wine data set, for regression with 13 inputs and 1 output
    '''
    Data_set = load_wine()
    X = Data_set.data
    y = Data_set.target[:, np.newaxis]
    # if norm == True:
    #     X = StandardScaler().fit_transform(X)
    #     y = StandardScaler().fit_transform(y)

    # Split dataset into train, valid and test set
    return split_data_set(X, y, test_size, valid_size, train_batch_size, valid_batch_size, seed, norm)


def load_large_channel(dataset_path, input_dim, output_dim, test_size, valid_size, train_batch_size, valid_batch_size, seed=42, norm=True):
    Data_set = np.load(dataset_path, allow_pickle=True).tolist()
    X = np.array(Data_set['x'])
    y = np.array(Data_set['y'])
    X = X[:, -input_dim:]
    y = y[:, :output_dim]
    # if norm == True:
    #     X = StandardScaler().fit_transform(X)
    #     y = StandardScaler().fit_transform(y)

    # Split dataset into train, valid and test set
    return split_data_set(X, y, test_size, valid_size, train_batch_size, valid_batch_size, seed, norm)

def load_ULA_channel(dataset_path, input_dim, output_dim, test_size, valid_size, train_batch_size, valid_batch_size,
                     gap=1, seed=42, norm=False):
    Data_set = np.load(dataset_path, allow_pickle=True).tolist()
    X = np.array(Data_set['x'])
    y = np.array(Data_set['y'])
    if len(X.shape) == 2:
        X = X[:, -input_dim * gap::gap]
        y = y[:, :output_dim:gap]
    elif len(X.shape) == 3:
        input_dim = int(input_dim / X.shape[2])
        X = X[:, -input_dim * gap::gap, :]
        output_dim = int(output_dim / y.shape[2])
        y = y[:, :output_dim:gap, :]
    else:
        raise Exception('Unsupported X shape')

    mean = np.array(Data_set['mean'])
    std = np.array(Data_set['std'])
    dl_train, dl_valid, dl_test = split_data_set(X, y, test_size, valid_size, train_batch_size, valid_batch_size, seed, norm)
    return dl_train, dl_valid, dl_test, mean, std

