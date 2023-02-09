import numpy as np

def random_sample_data(dataset_path, save_path, sample_num=20000):
    Data_set = np.load(dataset_path, allow_pickle=True).tolist()
    X = Data_set['x']
    y = Data_set['y']
    idx = np.random.choice(X.shape[0], sample_num, replace=False)
    X_new = X[idx, :]
    y_new = y[idx, :]
    np.save(save_path, {'x': X_new, 'y': y_new}, allow_pickle=True)

dataset_path1 = 'v1_7BS_HO_supervised_20000data.npy'
dataset_path2 = 'v5_7BS_HO_supervised_20000data.npy'
dataset_path3 = 'v10_7BS_HO_supervised_20000data.npy'
dataset_list = [dataset_path1, dataset_path2, dataset_path3]
save_path = 'v1+v5+v10_7BS_HO_supervised_60000data.npy'

# save_path1 = 'v1_HO_supervised_20000data.npy'
# save_path2 = 'v5_HO_supervised_20000data.npy'
# save_path3 = 'v10_HO_supervised_20000data.npy'
# random_sample_data(dataset_path1, save_path1)
# random_sample_data(dataset_path2, save_path2)
# random_sample_data(dataset_path3, save_path3)

X = None
y = None
for dataset_path in dataset_list:
    Data_set = np.load(dataset_path, allow_pickle=True).tolist()
    X = Data_set['x'] if X is None else np.concatenate([X, Data_set['x']], axis=0)
    y = Data_set['y'] if y is None else np.concatenate([y, Data_set['y']], axis=0)

np.save(save_path, {'x':X, 'y':y}, allow_pickle=True)


