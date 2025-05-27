import torch
import os
from tqdm import tqdm
import numpy as np
from torch.nn import functional as F
import sys
from torchvision import transforms

sys.path.append("/home/sktt1anhhuy/prach_preamble_detection_AI/dataloader")
from corr_gaf_data_loader import create_single_datasets_tensor_data, create_single_loaders_small_RAM

save_model_path = '/home/sktt1anhhuy/prach_preamble_detection_AI/weights'
gaf_corr_data_test = '/home/sktt1anhhuy/prach_preamble_detection_AI/gaf_corr_data_test'


model_name = ['vgg11_bn', 'vgg13_bn', 'vgg16_bn', 'vgg19_bn']
model_idx = 0

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = torch.hub.load('pytorch/vision:v0.10.0', model_name[model_idx], pretrained=False).to(device)
model.load_state_dict(torch.load(os.path.join(save_model_path, model_name[model_idx] + '_corr_gaf_data.pth')))

snr_range = np.arange(-50, 31, 5)
bs = 16
num_test = 3

to_tensor_vgg_input = transforms.Compose([
        transforms.Resize(224),          # VGG default
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

for snr in tqdm(snr_range):
    print(f"Testing on snr = {snr}...\n")
    test_gaf_corr_data_name = 'gafcorr_data_' + str(snr) + 'dB_test_info.npy'
    test_gaf_corr_label_name = 'gafcorr_data_' + str(snr) + 'dB_test_label.npy'

    test_gaf_corr_data = np.load(os.path.join(gaf_corr_data_test, test_gaf_corr_data_name))
    test_gaf_corr_label = np.load(os.path.join(gaf_corr_data_test, test_gaf_corr_label_name))

    test_datasets = create_single_datasets_tensor_data(test_gaf_corr_data, test_gaf_corr_label)

    test_dl = create_single_loaders_small_RAM(test_datasets, bs=bs)

    total_acc = []

    for i in tqdm(range(num_test)):
        correct, total = 0, 0
        for batch in tqdm(test_dl):
            x, y_batch = [t.to(device) for t in batch]
            x = to_tensor_vgg_input(x).float()
            out = model(x)
            preds = F.log_softmax(out, dim=1).argmax(dim=1)
            total += y_batch.size(0)
            correct += (preds == y_batch).sum().item()

            del x, y_batch

        total_acc.append(correct / total)

    avg_acc = sum(total_acc) / num_test
    print(f'Average Accuracy over {num_test} of snr = {snr}: {avg_acc}\n')