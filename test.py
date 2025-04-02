import torch
import numpy as np
import sys
import os
from torch.nn import functional as F

sys.path.append("dataloader")
from prach_data_loader import create_testing_datasets, create_testing_loaders

sys.path.append("models")
from models import ComplexNN_v1

data_dir = 'generated_dataset'
config_dir = 'pi_63_pci_158_rsi_39_prscs_30_puscs_30_zczc_8_fr_FR1_s_UnrestrictedSet_st_Unpaired_fs_0_snrRange_-30_-29_'
testing_data_name = 'testing_rx_1_freqComb_12_numFrame_1.npy'

weights_path = 'weights'
weight_name = 'rx_1_freqComb_12_numFrame_1_best.pth'

testing_data_path = os.path.join(data_dir, config_dir, testing_data_name)

test_dataset = np.load(testing_data_path)

test_size = test_dataset.shape[0]

test_label = test_dataset[:, -1].astype(int)
test_label = (test_label == 60).astype(int) # transform multivariate classification to binary classification

test_data = np.delete(test_dataset, -1, 1)

batch_size = 64

test_datasets = create_testing_datasets(test_data, test_label)
test_dataloader = create_testing_loaders(test_datasets, batch_size=batch_size)

path_pt = os.path.join(weights_path, weight_name)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
input_size = test_data.shape[1]
num_classes = 64

model = ComplexNN_v1(input_size=input_size, output_size=num_classes, device=device).to(device)
model.load_state_dict(torch.load(path_pt))

model.eval()
correct, total = 0, 0
for batch in test_dataloader:
    x, y_batch = [t.to(device) for t in batch]
    output = model(x)

    output = output.float()
    # print(f"output {output}")
    preds = F.log_softmax(output, dim=1).argmax(dim=1)
    # print(preds)
    # print(y_batch)
    total += y_batch.size(0)
    correct += (preds == y_batch).sum().item()

acc = correct / total
print('Best Accuracy: ', acc)
