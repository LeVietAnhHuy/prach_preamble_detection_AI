import torch
import os
from tqdm import tqdm
import numpy as np
from torch.nn import functional as F
import sys
from torchvision import transforms
import mat73
from numpy.fft import ifft
import numpy.matlib
from pyts.image import GramianAngularField
import matplotlib.pyplot as plt

sys.path.append("/home/sktt1anhhuy/prach_preamble_detection_AI/dataloader")
from corr_gaf_data_loader import create_single_datasets_tensor_data, create_single_loaders_small_RAM

img_path = '/home/sktt1anhhuy/prach_preamble_detection_AI/image'
save_model_path = '/home/sktt1anhhuy/prach_preamble_detection_AI/weights'
fft_pair_data_path = '/home/sktt1anhhuy/prach_preamble_detection_AI/fft_pair_data'

model_name = ['vgg11_bn', 'vgg13_bn', 'vgg16_bn', 'vgg19_bn',
              'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
              'vit_b_16', 'vit_b_32', 'vit_l_16', 'vit_l_32', 'vit_h_14',
              ]
model_idx = 0


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = torch.hub.load('pytorch/vision:v0.10.0', model_name[model_idx], pretrained=False).to(device)
model.load_state_dict(torch.load(os.path.join(save_model_path, model_name[model_idx] + '_corr_gaf_data.pth')))

snr_range = np.arange(-15, 31, 5)
bs = 16
num_test = 3
num_rx = 8
save_plot = False
ifft_circ_shift = -10
prach_duration = 12
L_RA = 139
bs = 224

divisors = []
for i in range(1, int(12 / 2) + 1):
    if 12 % i == 0:
        divisors.append(i)
divisors.append(12)

to_tensor_vgg_input = transforms.Compose([
        transforms.Resize(224),          # VGG default
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

gaf = GramianAngularField(method='difference', image_size=32)

for snr in tqdm(snr_range):
    print(f"\nTesting on snr = {snr}dB...\n")

    data_dict_mat = []

    for rx_idx in range(num_rx):

        file_name = 'pair_fft_data_' + str(snr) + 'dB_rx' + str(rx_idx + 1) + '.mat'

        data_dict = mat73.loadmat(os.path.join(fft_pair_data_path, file_name))

        data_dict_mat.append(data_dict['data'])

    tot_num_samples = data_dict_mat[0].shape[0]
    tot_test_ite = int(tot_num_samples / prach_duration) # 127800
    tot_correct = 0

    for sample_idx in range(tot_test_ite):
        start_sample_idx = sample_idx * prach_duration
        end_sample_idx = start_sample_idx + prach_duration

        data_mat = []

        for rx in tqdm(range(num_rx)):
            data = data_dict_mat[rx][start_sample_idx:end_sample_idx, :L_RA]

            data_mat.append(data)

        x_u = data_dict_mat[0][start_sample_idx, L_RA:-1]
        label = int(data_dict_mat[0][start_sample_idx, -1])

        del data_dict

        data_mat = np.array(data_mat)  # 8x12x139
        data_mat_freq_comb_mat = []

        # frequency combining
        for div in divisors:
            data_mat_freq_comb = np.reshape(data_mat,
                                            (data_mat.shape[0], int(prach_duration / div), div, -1))
            data_mat_freq_comb = np.sum(data_mat_freq_comb, axis=2) / div
            data_mat_freq_comb_mat.append(data_mat_freq_comb)

            del data_mat_freq_comb

        del data_mat

        data_mat_freq_comb_mat = np.concatenate(data_mat_freq_comb_mat, axis=1)

        x_u_mat = np.matlib.repmat(x_u, data_mat_freq_comb_mat.shape[0], data_mat_freq_comb_mat.shape[1])
        x_u_mat = np.reshape(x_u_mat, (data_mat_freq_comb_mat.shape[0], data_mat_freq_comb_mat.shape[1], -1))

        corr_mat = np.multiply(np.conjugate(data_mat_freq_comb_mat), x_u_mat)  # 8x28x139

        del x_u_mat
        del data_mat_freq_comb_mat

        ifft_corr_mat = np.zeros((corr_mat.shape[0], corr_mat.shape[1], 1024))
        ifft_corr_mat[:, :, :L_RA] = corr_mat

        del corr_mat

        ifft_corr_mat = ifft(ifft_corr_mat, axis=-1)
        ifft_corr_mat = np.abs(ifft_corr_mat)
        ifft_corr_mat = np.roll(ifft_corr_mat, shift=ifft_circ_shift, axis=-1)  # 8x28x1024

        gaf_data = []
        for rx in range(num_rx):
            for idx in range(ifft_corr_mat.shape[1]):

                sample_2d = np.array([ifft_corr_mat[rx, idx, :]])

                # plt.figure()
                # plt.plot(sample_2d[0])
                # plt.grid(True)
                # plt.savefig(os.path.join(img_path, 'test_corr_gaf_MIMO.png'), dpi=300, bbox_inches='tight')
                # plt.close()

                gaf_img = gaf.fit_transform(sample_2d)
                gaf_img = gaf_img[0]
                img_3ch = np.stack([gaf_img, gaf_img, gaf_img], axis=0)

                gaf_data.append(img_3ch)

                del img_3ch

        gaf_label = np.zeros((1, num_rx * ifft_corr_mat.shape[1])) + label # 1x224
        gaf_label = gaf_label[0]
        gaf_data = np.array(gaf_data)

        test_datasets = create_single_datasets_tensor_data(gaf_data, gaf_label)

        test_dl = create_single_loaders_small_RAM(test_datasets, bs=bs)

        correct, total = 0, 0

        x, y_batch = [t.to(device) for t in test_dl]
        x = to_tensor_vgg_input(x).float()
        out = model(x)
        preds = F.log_softmax(out, dim=1).argmax(dim=1)
        total += y_batch.size(0)
        correct += (preds == y_batch).sum().item()

        avg_acc = correct / total

        if avg_acc > 1/10:
            tot_correct += 1

    tot_acc = tot_correct / tot_test_ite
    print(f'Average Accuracy  of snr = {snr}dB: {tot_acc}\n')