from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import torch
import numpy as np

def create_datasets(data, label, data_size, valid_pct=0.15, seed=42):
    idx = np.arange(data_size)
    train_idx, test_valid_idx = train_test_split(idx,
                                                 test_size=valid_pct * 2,
                                                 random_state=seed,
                                                 shuffle=True)

    test_idx, val_idx = train_test_split(test_valid_idx,
                                          test_size=0.5,
                                          random_state=seed,
                                          shuffle=True)

    train_ds = TensorDataset(
        torch.tensor(data[train_idx]).float(),
        torch.tensor(label[train_idx]).long()
    )
    valid_ds = TensorDataset(
        torch.tensor(data[val_idx]).float(),
        torch.tensor(label[val_idx]).long()
    )
    test_ds = TensorDataset(
        torch.tensor(data[test_idx]).float(),
        torch.tensor(label[test_idx]).long()
    )

    return train_ds, valid_ds, test_ds

def create_loaders(data, bs=64, jobs=0):
    train_ds, valid_ds, test_ds = data

    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=jobs)
    valid_dl = DataLoader(valid_ds, batch_size=bs, shuffle=True, num_workers=jobs)
    test_dl = DataLoader(test_ds, batch_size=bs, shuffle=True, num_workers=jobs)

    return train_dl, valid_dl, test_dl

def create_datasets_tensor_data(data, label, data_size, valid_pct=0.15, seed=42):
    idx = np.arange(data_size)
    train_idx, test_valid_idx = train_test_split(idx,
                                                 test_size=valid_pct * 2,
                                                 random_state=seed,
                                                 shuffle=True)

    test_idx, val_idx = train_test_split(test_valid_idx,
                                          test_size=0.5,
                                          random_state=seed,
                                          shuffle=True)

    train_ds = TensorDataset(data[train_idx], torch.tensor(label[train_idx]).long())
    val_ds = TensorDataset(data[val_idx], torch.tensor(label[val_idx]).long())
    test_ds = TensorDataset(data[test_idx], torch.tensor(label[test_idx]).long())

    return train_ds, val_ds, test_ds, len(train_idx), len(test_idx), len(val_idx)

def create_loaders_tensor_data(data, bs=64, jobs=0):
    train_ds, valid_ds, test_ds, _, _, _ = data

    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=jobs)
    valid_dl = DataLoader(valid_ds, batch_size=bs, shuffle=True, num_workers=jobs)
    test_dl = DataLoader(test_ds, batch_size=bs, shuffle=True, num_workers=jobs)

    return train_dl, valid_dl, test_dl

def create_datasets_small_RAM(data, label, data_size, valid_pct=0.2, seed=42):
    idx = np.arange(data_size)
    train_idx, val_idx = train_test_split(idx,
                                                test_size=valid_pct,
                                                random_state=seed,
                                                shuffle=True)

    train_ds = TensorDataset(torch.tensor(data[train_idx]), torch.tensor(label[train_idx]).long())
    val_ds = TensorDataset(torch.tensor(data[val_idx]), torch.tensor(label[val_idx]).long())

    return train_ds, val_ds, len(train_idx), len(val_idx)

def create_loaders_small_RAM(data, bs=64, jobs=0):
    train_ds, valid_ds, _, _ = data

    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=jobs)
    valid_dl = DataLoader(valid_ds, batch_size=bs, shuffle=True, num_workers=jobs)

    return train_dl, valid_dl


