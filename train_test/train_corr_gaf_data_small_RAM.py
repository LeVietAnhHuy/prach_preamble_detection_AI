import torch
import os
# from tqdm import tqdm
from tqdm.auto import tqdm
import numpy as np
import sys
from torch.nn import functional as F
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import glob
from torchvision import transforms

sys.path.append("/home/sktt1anhhuy/prach_preamble_detection_AI/dataloader")
from corr_gaf_data_loader import create_datasets_small_RAM ,create_loaders_small_RAM

gaf_corr_data_path = '/home/sktt1anhhuy/prach_preamble_detection_AI/gaf_corr_data'
gaf_corr_data_list = glob.glob(os.path.join(gaf_corr_data_path, '*train_info*.npy'))
gaf_corr_label_list = glob.glob(os.path.join(gaf_corr_data_path, '*train_label*.npy'))
plot_path = '/home/sktt1anhhuy/prach_preamble_detection_AI/plot'
train_test_log_path = '/home/sktt1anhhuy/prach_preamble_detection_AI/train_test'

# corr_vgg_input_data_tensor = torch.randn(10, 3, 32, 32)
# corr_vgg_input_label = array = np.random.rand(10)

save_model_path = '/home/sktt1anhhuy/prach_preamble_detection_AI/weights'
os.makedirs(save_model_path, exist_ok=True)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

bs = 6
seed = 42
np.random.seed(seed)

to_tensor_vgg_input = transforms.Compose([
        transforms.Resize(224),          # VGG default
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

lr = 0.001
n_epochs = 30
num_classes = 10
best_acc = 0
patience, trials = 500, 0
base = 1
step = 2

train_loss_history = []
val_loss_history = []
acc_history = []

model_name = ['vgg11_bn', 'vgg13_bn', 'vgg16_bn', 'vgg19_bn',
              'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
              'vit_b_16', 'vit_b_32', 'vit_l_16', 'vit_l_32', 'vit_h_14',
              ]
model_idx = 0

model = torch.hub.load('pytorch/vision:v0.10.0', model_name[model_idx], pretrained=True).to(device)

criterion = nn.CrossEntropyLoss(reduction='sum')
opt = optim.Adam(model.parameters(), lr=lr)
total_params = sum(param.numel() for param in model.parameters())
print('Total Parameter: ', total_params)
print(f'Start model training on:')

for epoch in tqdm(range(1, n_epochs + 1), dynamic_ncols=True, leave=False):
    total_train_size = 0
    total_val_size = 0

    train_loss = 0
    correct, total = 0, 0
    val_loss = 0

    for data_idx in tqdm(range(1, len(gaf_corr_data_list)), dynamic_ncols=True, leave=False):

        gaf_corr_data = np.load(os.path.join(gaf_corr_data_path, gaf_corr_data_list[data_idx]))
        dB_values = gaf_corr_data_list[data_idx].split('/')[-1].split('_')[2]

        for label in gaf_corr_label_list:
            if dB_values in label:
                gaf_corr_label = np.load(os.path.join(gaf_corr_data_path, label))

        data_size = gaf_corr_data.shape[0]

        datasets = create_datasets_small_RAM(gaf_corr_data, gaf_corr_label, data_size)

        train_dl, val_dl = create_loaders_small_RAM(datasets, bs=bs)

        total_train_size += datasets[2]
        total_val_size += datasets[3]

        model.train()

        for i, batch in enumerate(tqdm(train_dl)):
            x, y_batch = [t.to(device) for t in batch]
            x = to_tensor_vgg_input(x).float()
            opt.zero_grad()
            out = model(x)
            loss = criterion(out, y_batch)
            train_loss += loss.item()
            loss.backward()
            opt.step()

        model.eval()
        correct, total = 0, 0
        val_loss = 0
        for batch in tqdm(val_dl):
            x, y_batch = [t.to(device) for t in batch]
            x = to_tensor_vgg_input(x).float()
            out = model(x)
            loss = criterion(out, y_batch)
            val_loss += loss.item()
            preds = F.log_softmax(out, dim=1).argmax(dim=1)
            total += y_batch.size(0)
            correct += (preds == y_batch).sum().item()

            del x, y_batch, out, loss
            torch.cuda.empty_cache()

    train_loss /= total_train_size
    train_loss_history.append(train_loss)

    val_loss /= total_val_size
    val_loss_history.append(val_loss)

    acc = correct / total
    acc_history.append(acc)

    if epoch % base == 0:
        print(f'Epoch: {epoch:3d}. Loss: {train_loss:.4f}. Acc.: {acc:2.2%}')
        base *= step

    if acc > best_acc:
        trials = 0
        best_acc = acc

        torch.save(model.state_dict(), os.path.join(save_model_path, model_name[model_idx] + '_corr_gaf_data.pth'))
        print(f'Epoch {epoch} best model saved with accuracy: {best_acc:2.2%}')

    else:
        trials += 1
        if trials >= patience:
            print(f'Early stopping on epoch {epoch}')
            break

epochs = range(1, len(train_loss_history) + 1)
plt.figure()
plt.plot(epochs, train_loss_history, marker='o', label='Training Loss')
plt.plot(epochs, val_loss_history, marker='s', label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training vs. Validation Loss')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plot_path, model_name[model_idx] + '_train_val_loss.png'), dpi=300, bbox_inches='tight')
plt.close()