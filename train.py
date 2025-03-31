import torch
from torch import optim, nn
from torch.nn import functional as F
import numpy as np
import sys
import os

sys.path.append("dataloader")
from prach_data_loader import create_training_datasets, create_training_loaders

sys.path.append("models")
from models import NN_v1

data_name = 'rx_1_freqComb_12_numFrame_1_'
data_path = 'generated_dataset/pi_63_pci_158_rsi_39_prscs_30_puscs_30_zczc_8_fr_FR1_s_UnrestrictedSet_st_Unpaired_fs_0_snrRange_-40_21_/rx_1_freqComb_12_numFrame_1.npy'

weights_path = 'weights'

dataset = np.load(data_path)

train_size = dataset.shape[0]

label = dataset[:, -1].astype(int)
data = np.delete(dataset, -1, 1)

batch_size = 50

datasets = create_training_datasets(data, label)
train_dataloader, val_dataloader, val_size = create_training_loaders(datasets, batch_size=batch_size)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
learning_rate = 0.001  # important parameter
num_epochs = 50  # important parameter
interations_per_epoch = len(train_dataloader)
best_acc = 0
patience, trials = 500, 0
base = 1
step = 2
loss_history = []
acc_history = []

input_size = data.shape[1]
output_size = 64
model = NN_v1(input_size=input_size, output_size=output_size).to(device)
print('Model:')
print(model)

criterion = nn.CrossEntropyLoss(reduction='sum')  # important parameter
optimizer_Adam = optim.Adam(model.parameters(), lr=learning_rate)  # important parameter

total_params = sum(
    param.numel() for param in model.parameters()
)

print('Total parameters: ', total_params)

print('Start model training')

for epoch in range(1, num_epochs + 1):

    model.train()
    epoch_loss = 0


    for i, batch in enumerate(train_dataloader):
        x, y_batch = [t.to(device) for t in batch]
        optimizer_Adam.zero_grad()
        output = model(x)
        loss = criterion(output, y_batch)
        epoch_loss += loss.item()
        loss.backward()
        optimizer_Adam.step()

    epoch_loss /= train_size
    loss_history.append(epoch_loss)

    model.eval()
    correct, total = 0, 0

    for batch in val_dataloader:
        x, y_batch = [t.to(device) for t in batch]
        out = model(x)
        preds = F.log_softmax(out, dim=1).argmax(dim=1)
        total += y_batch.size(0)
        correct += (preds == y_batch).sum().item()

    acc = correct / total
    acc_history.append(acc)

    if epoch % base == 0:
        print(f'Epoch: {epoch:3d}. Loss: {epoch_loss:.4f}. Acc.: {acc:2.2%}')
        base *= step

    if acc > best_acc:
        trials = 0
        best_acc = acc
        torch.save(model.state_dict(), os.path.join(weights_path, data_name + 'best.pth'))
        print(f'Epoch {epoch} best model saved with accuracy: {best_acc:2.2%}')
    else:
        trials += 1
        if trials >= patience:
            print(f'Early stopping on epoch {epoch}')
            break
    if epoch % base == 0:
        print(f'Epoch: {epoch:3d}. Loss: {epoch_loss:.4f}. Acc.: {acc:2.2%}')
