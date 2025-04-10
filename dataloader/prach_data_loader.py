import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

def create_training_datasets(data, label, valid_pct=0.2, seed=None):

  data_train, data_valid, label_train, label_valid = train_test_split(data, label,
                                                                      test_size=valid_pct,
                                                                      random_state=seed,
                                                                      shuffle=True)
  val_size = len(label_valid)
  train_dataset = TensorDataset(
      torch.tensor(data_train).type(torch.complex64),
      torch.tensor(label_train).long()
  )
  val_dataset = TensorDataset(
      torch.tensor(data_valid).type(torch.complex64),
      torch.tensor(label_valid).long()
  )

  return train_dataset, val_dataset, val_size

def create_training_loaders(data, batch_size=20, jobs=0):
  """Wraps the datasets returned by create_datasets function
  with data loaders."""

  train_dataset, val_dataset, val_size = data

  train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=jobs)
  val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=jobs)

  return  train_dataloader, val_dataloader, val_size

def create_testing_datasets(data_test, label_test, valid_pct=0.1, seed=None):

  test_dataset = TensorDataset(
      torch.tensor(data_test).type(torch.complex64),
      torch.tensor(label_test).long()
  )

  return test_dataset

def create_testing_loaders(data, batch_size=20, jobs=0):
  """Wraps the datasets returned by create_datasets function
  with data loaders."""

  test_dataset = data

  test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=jobs)

  return  test_dataloader