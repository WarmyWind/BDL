from torch.utils import data
from torchvision import datasets, transforms
import torch
from torch_geometric.data import Data
import numpy as np
import lib.function_wmmse_powercontrol as wf

def get_wGaussian_graph_data(K, num_H, var_noise=1, Pmin=0, seed=2017, WMMSE_eval=True):
    def generate_wGaussian(K, num_H, var_noise=1, Pmin=0, seed=2017):
        print('Generate Data ... (seed = %d)' % seed)
        np.random.seed(seed)
        # Pmax = 1
        # Pini = Pmax * np.ones((num_H, K, 1))
        # alpha = np.random.rand(num_H,K)
        alpha = np.random.rand(num_H, K)
        # alpha = np.ones((num_H,K))
        fake_a = np.ones((num_H, K))
        X = np.zeros((K ** 2, num_H))
        Y = np.zeros((K, num_H))
        # total_time = 0.0
        CH = 1 / np.sqrt(2) * (np.random.randn(num_H, K, K) + 1j * np.random.randn(num_H, K, K))
        H = abs(CH)
        # Y = wf.batch_WMMSE2(Pini, alpha, H, Pmax, var_noise)
        # Y2 = wf.batch_WMMSE2(Pini, fake_a, H, Pmax, var_noise)
        # return H, Y, alpha, Y2
        return H, alpha

    def get_cg(n):
        adj = []
        for i in range(0, n):
            for j in range(0, n):
                if (not (i == j)):
                    adj.append([i, j])
        return adj

    def build_graph(H, A, adj):
        K = H.shape[0]
        x1 = np.expand_dims(np.diag(H), axis=1)
        x2 = np.expand_dims(A, axis=1)
        x3 = np.ones((K, 1))
        x = np.concatenate((x1, x2, x3), axis=1)
        x = torch.tensor(x, dtype=torch.float)

        edge_attr = []
        for e in adj:
            try:
                edge_attr.append([H[e[0], e[1]], H[e[1], e[0]]])
            except:
                raise Exception('Wrong')
        edge_index = torch.tensor(adj, dtype=torch.long)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        y = torch.tensor(np.expand_dims(H, axis=0), dtype=torch.float)
        pos = torch.tensor(np.expand_dims(A, axis=0), dtype=torch.float)

        data = Data(x=x, edge_index=edge_index.t().contiguous(), edge_attr=edge_attr, y=y, pos=pos)
        return data

    HH, AA = generate_wGaussian(K, num_H, seed=seed, var_noise=var_noise)
    if WMMSE_eval:
        Pmax = 1
        Pini = Pmax * np.ones((num_H, K, 1))
        Y = wf.batch_WMMSE2(Pini, AA, HH, Pmax, var_noise)
        print('wmmse:', wf.np_sum_rate(HH.transpose(0, 2, 1), Y, AA, var_noise))

    data_list = []
    # cg = get_cg(K)
    cg = get_cg(K)
    for i in range(num_H):
        data = build_graph(HH[i],AA[i],cg)
        data_list.append(data)
    return data_list


def get_SVHN(root):
    input_size = 32
    num_classes = 10

    # NOTE: these are not correct mean and std for SVHN, but are commonly used
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    train_dataset = datasets.SVHN(
        root + "/SVHN", split="train", transform=transform, download=True
    )
    test_dataset = datasets.SVHN(
        root + "/SVHN", split="test", transform=transform, download=True
    )
    return input_size, num_classes, train_dataset, test_dataset


def get_CIFAR10(root):
    input_size = 32
    num_classes = 10

    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    # Alternative
    # normalize = transforms.Normalize(
    #     (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    # )

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )

    train_dataset = datasets.CIFAR10(
        root + "/CIFAR10", train=True, transform=train_transform, download=True
    )

    test_transform = transforms.Compose([transforms.ToTensor(), normalize])
    test_dataset = datasets.CIFAR10(
        root + "/CIFAR10", train=False, transform=test_transform, download=False
    )

    return input_size, num_classes, train_dataset, test_dataset


def get_CIFAR100(root):
    input_size = 32
    num_classes = 100
    normalize = transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2673, 0.2564, 0.2762))

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    train_dataset = datasets.CIFAR100(
        root + "/CIFAR100", train=True, transform=train_transform, download=True
    )

    test_transform = transforms.Compose([transforms.ToTensor(), normalize])
    test_dataset = datasets.CIFAR100(
        root + "/CIFAR100", train=False, transform=test_transform, download=False
    )

    return input_size, num_classes, train_dataset, test_dataset


all_datasets = {
    "SVHN": get_SVHN,
    "CIFAR10": get_CIFAR10,
    "CIFAR100": get_CIFAR100,
}


def get_dataset(dataset, root="./"):
    return all_datasets[dataset](root)


def get_dataloaders(dataset, train_batch_size=128, root="./"):
    ds = all_datasets[dataset](root)
    input_size, num_classes, train_dataset, test_dataset = ds

    kwargs = {"num_workers": 4, "pin_memory": True}

    train_loader = data.DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True, **kwargs
    )

    test_loader = data.DataLoader(
        test_dataset, batch_size=1000, shuffle=False, **kwargs
    )

    return train_loader, test_loader, input_size, num_classes
