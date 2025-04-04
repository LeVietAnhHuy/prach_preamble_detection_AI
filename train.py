import torch
from torch import optim, nn
from torch.nn import functional as F
import numpy as np
import sys
import os
import matplotlib.pyplot as plt

sys.path.append("dataloader")
from prach_data_loader import create_training_datasets, create_training_loaders

sys.path.append("models")
from models import ComplexNN_v1, ComplexNN_v2
from complextorch.nn.modules.loss import CVCauchyError
from complextorch.nn.modules.activation.split_type_A import CSigmoid

data_name = 'rx_1_freqComb_12_numFrame_1_'
data_path = 'generated_dataset/pi_63_pci_158_rsi_39_prscs_30_puscs_30_zczc_8_fr_FR1_s_UnrestrictedSet_st_Unpaired_fs_0_snrRange_-40_21_/rx_1_freqComb_12_numFrame_1.npy'

weights_path = 'weights'

dataset = np.load(data_path)

train_size = dataset.shape[0]


label = dataset[:, -1].astype(int)

# transform multivariate classification to binary classification
label = (label == 60).astype(int)

data = np.delete(dataset, -1, 1)

batch_size = 256

datasets = create_training_datasets(data, label)
train_dataloader, val_dataloader, val_size = create_training_loaders(datasets, batch_size=batch_size)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  #
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
# num_classes = np.unique(label).size
num_classes = 1

# model = ComplexNN_v1(input_size=input_size, output_size=num_classes, device=device).to(device)
model = ComplexNN_v2(input_size=input_size, output_size=num_classes).to(device)

model_name = '_ComplexNN_v2_'

print('Model:')
print(model)

# criterion = nn.CrossEntropyLoss(reduction='sum')  # important parameter
criterion = CVCauchyError()
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
        # x, y_batch = [t.to(device) for t in batch]
        x, y_batch = [t.to(device) for t in batch]
        optimizer_Adam.zero_grad()
        output = model(x)
        # output = output.float()
        loss = criterion(output, y_batch)
        epoch_loss += loss.item()
        loss.backward()
        optimizer_Adam.step()

    epoch_loss /= train_size
    loss_history.append(epoch_loss)

    model.eval()
    correct, total = 0, 0

    for batch in val_dataloader:
        # x, y_batch = [t.to(device) for t in batch]
        x, y_batch = [t.to("cuda") for t in batch]
        output = model(x)
        output = output.real
        # output = output.float()
        # preds = F.log_softmax(output, dim=1).argmax(dim=1)
        # preds = output.astype(int)
        preds = torch.sigmoid(output) > 0.6
        preds = torch.squeeze(preds).int()
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
        torch.save(model.state_dict(), os.path.join(weights_path, data_name + model_name + 'best.pth'))
        print(f'Epoch {epoch} best model saved with accuracy: {best_acc:2.2%}')
    else:
        trials += 1
        if trials >= patience:
            print(f'Early stopping on epoch {epoch}')
            break
    if epoch % base == 0:
        print(f'Epoch: {epoch:3d}. Loss: {epoch_loss:.4f}. Acc.: {acc:2.2%}')

plot_dir = 'plot'
os.makedirs(plot_dir, exist_ok=True)

# Plotting the loss & acc curves
plt.figure(figsize=(7, 5))
plt.plot(range(num_epochs), acc_history, color='g', label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

acc_plot = 'valAcc.png'
plt.savefig(os.path.join(plot_dir, acc_plot))


plt.figure(2, figsize=(7, 5))
plt.plot(range(num_epochs), loss_history, color='g', label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

loss_plot = 'valLoss.png'
plt.savefig(os.path.join(plot_dir, loss_plot))