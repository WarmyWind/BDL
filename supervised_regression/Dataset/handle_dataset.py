import numpy as np
import scipy.io as scio


def get_data_from_mat(filepath, index):
    mat = scio.loadmat(filepath)
    data = mat.get(index)  # 取出字典里的label
    return data


def prepare_sequential_dataset(file_path, obs_len, pred_len, norm=True, mat_index='H'):
    if file_path[-3:] == 'npy':
        data = np.load(file_path, allow_pickle=True)
    elif file_path[-3:] == 'mat':
        data = get_data_from_mat(file_path, mat_index)
    else:
        raise Exception('Invalid format of file!')
    assert len(data.shape) == 2 or len(data.shape) == 3

    if len(data.shape) == 2:
        mean = np.mean(data)
        std = np.std(data)
    else:
        mean = np.mean(np.reshape(data, (-1, data.shape[2])), axis=0)
        std = np.std(np.reshape(data, (-1, data.shape[2])), axis=0)

    if norm:
        data = (data - mean) / std

    X = []
    y = []
    for idx in range(data.shape[1]-obs_len-pred_len+1):
        _X = data[:, idx:idx+obs_len,...]
        _y = data[:, idx+obs_len:idx+obs_len+pred_len,...]
        X.append(_X)
        y.append(_y)

    return np.array(X), np.array(y), mean, std

def mat_to_npy(file_path, index):
    mat = scio.loadmat(file_path)
    data = mat.get(index)
    return np.array(data)

if __name__ == '__main__':
    file_path = 'speed1_3_5ms_channel.mat'
    data = mat_to_npy(file_path, 'H')
    data = np.swapaxes(data, 1, 2)
    data_real = np.real(data)
    data_imag = np.imag(data)
    data = np.concatenate((data_real, data_imag), axis=-1)
    data = np.reshape(data, (-1, data.shape[-2], data.shape[-1]))

    file_path = 'speed1_3_5ms_channel.npy'
    np.save(file_path, data, allow_pickle=True)
    obs_len = 5
    pred_len = 5
    X, y, mean, std = prepare_sequential_dataset(file_path, obs_len, pred_len, norm=True)
    X, y = np.squeeze(X), np.squeeze(y)

    # X = np.reshape(X, (-1, obs_len))
    # y = np.reshape(y, (-1, pred_len))
    np.save('normed_speed1_3_5ms_channel_dataset.npy', {'x':X, 'y':y, 'mean':mean, 'std':std}, allow_pickle=True)
