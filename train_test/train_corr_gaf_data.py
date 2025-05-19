import torch
import os
import tqdm
import numpy as np
import sys
from torch.nn import functional as F
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

sys.path.append("dataloader")
from corr_gaf_data_loader import create_datasets_tensor_data, create_loaders_tensor_data

gaf_corr_vgg_input_data_tensor_path = 'D:/5G_python/prach_ai/prach_preamble_detection_AI/corr_data_dot_mat/gaf_corr_vgg_input_data.pt'
label_path = 'D:/5G_python/prach_ai/prach_preamble_detection_AI/corr_data_dot_mat/gaf_corr_vgg_input_label.npy'
plot_path = 'D:/5G_python/prach_ai/prach_preamble_detection_AI/plot'

train_test_log_path = 'D:/5G_python/prach_ai/prach_preamble_detection_AI/train_test_log'

corr_vgg_input_data_tensor = torch.load(gaf_corr_vgg_input_data_tensor_path)
corr_vgg_input_label = np.load(label_path)

save_model_path = 'D:/5G_python/prach_ai/prach_preamble_detection_AI/weights'
os.makedirs(save_model_path, exist_ok=True)

bs = 64
seed = 1
np.random.seed(seed)
data_size = corr_vgg_input_data_tensor.shape[0]
datasets = create_datasets_tensor_data(corr_vgg_input_data_tensor, corr_vgg_input_label, data_size)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
train_dl, val_dl, test_dl, train_size, val_size, test_size = create_loaders_tensor_data(datasets, bs=bs)

lr = 0.001
n_epochs = 50
iterations_per_epoch = len(train_dl)
_, num_classes = np.unique(corr_vgg_input_label, return_counts=True)
best_acc = 0
patience, trials = 500, 0
base = 1
step = 2
train_loss_history = []
val_loss_history = []
acc_history = []

model_name = 'vgg11'

# model = AlexNet(num_classes=num_classes).to(device)
# model = ResNet18(residual_blocks, output_shape=nums_class).to(device)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False).to(device)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=False).to(device)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=False).to(device)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet101', pretrained=False).to(device)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=False).to(device)

model = torch.hub.load('pytorch/vision:v0.10.0', model_name, pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11_bn', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg13', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg13_bn', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16_bn', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg19', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg19_bn', pretrained=True)


criterion = nn.CrossEntropyLoss(reduction='sum')
opt = optim.Adam(model.parameters(), lr=lr)
total_params = sum(param.numel() for param in model.parameters())
print('Total Parameter: ', total_params)
print('Start model training')

for epoch in tqdm(range(1, n_epochs + 1)):
    model.train()
    train_loss = 0
    for i, batch in enumerate(train_dl):
        x, y_batch = [t.to(device) for t in batch]
        opt.zero_grad()
        out = model(x)
        loss = criterion(out, y_batch)
        train_loss += loss.item()
        loss.backward()
        opt.step()

    train_loss /= train_size
    train_loss_history.append(train_loss)

    model.eval()
    correct, total = 0, 0
    val_loss = 0
    for batch in val_dl:
        x, y_batch = [t.to(device) for t in batch]
        out = model(x)
        loss = criterion(out, y_batch)
        val_loss += loss.item()
        preds = F.log_softmax(out, dim=1).argmax(dim=1)
        total += y_batch.size(0)
        correct += (preds == y_batch).sum().item()

    acc = correct / total
    acc_history.append(acc)

    val_loss /= val_size
    val_loss_history.append(val_loss)

    if epoch % base == 0:
        print(f'Epoch: {epoch:3d}. Loss: {train_loss:.4f}. Acc.: {acc:2.2%}')
        base *= step

    if acc > best_acc:
        trials = 0
        best_acc = acc
        # torch.save(model.state_dict(), os.path.join(path_save_model, 'AlexNet.pth'))
        # torch.save(model.state_dict(), os.path.join(path_save_model, 'ResNet34.pth'))
        # torch.save(model.state_dict(), os.path.join(path_save_model, 'ResNet50.pth'))
        # torch.save(model.state_dict(), os.path.join(path_save_model, 'ResNet101.pth'))
        # torch.save(model.state_dict(), os.path.join(save_model_path, 'ResNet152.pth'))

        torch.save(model.state_dict(), os.path.join(save_model_path, model_name + '_corr_gaf_data.pth'))
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
plt.savefig(os.path.join(plot_path, model_name + '_train_val_loss.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"Start testing on {device}...\n")

model = torch.hub.load('pytorch/vision:v0.10.0', model_name, pretrained=False).to(device)
model.load_state_dict(torch.load(os.path.join(save_model_path, model_name + '_corr_gaf_data.pth')))

total_acc = []
num_test = 10
for i in tqdm(range(num_test)):
    correct, total = 0, 0
    for batch in test_dl:
        x, y_batch = [t.to(device) for t in batch]
        out = model(x)
        preds = F.log_softmax(out, dim=1).argmax(dim=1)
        total += y_batch.size(0)
        correct += (preds == y_batch).sum().item()

    total_acc.append(correct / total)
avg_acc = sum(total_acc) / num_test
print(' Average Accuracy: ', avg_acc)
