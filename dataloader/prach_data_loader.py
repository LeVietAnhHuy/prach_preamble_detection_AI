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
      torch.tensor(data_train),
      torch.tensor(label_train).long()
  )
  val_dataset = TensorDataset(
      torch.tensor(data_valid),
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