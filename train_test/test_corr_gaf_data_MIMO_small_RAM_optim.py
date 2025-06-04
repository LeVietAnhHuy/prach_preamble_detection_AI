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
import statistics
import warnings
from scipy import stats as st
import cupy as cp

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
model.load_state_dict(torch.load(os.path.join(save_model_path, model_name[model_idx] + '_corr_gaf_data_v0.pth')))

snr_range = np.arange(-50, 1, 5)
bs = 16
num_test = 3
num_rx = 8
save_plot = False
ifft_circ_shift = -10
prach_duration = 12
L_RA = 139
bs = 32

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

    tot_correct_without_rx_comb = 0
    tot_correct_rx_comb = 0

    for sample_idx in tqdm(range(tot_test_ite)):
        start_sample_idx = sample_idx * prach_duration
        end_sample_idx = start_sample_idx + prach_duration

        data_mat = []

        for rx in range(num_rx):
            data = data_dict_mat[rx][start_sample_idx:end_sample_idx, :L_RA]

            data_mat.append(data)

        x_u = data_dict_mat[0][start_sample_idx, L_RA:-1]
        label = np.abs(data_dict_mat[0][start_sample_idx, -1])

        data_mat = np.array(data_mat)  # 8x12x139
        data_mat_freq_comb_mat = []

        # frequency combining
        pre_idx_without_rx_comb_1ite = []
        pre_idx_rx_comb_1ite = []

        for div in divisors:
            # print(f'\n div = {div}\n')
            # div = 4

            # ## numpy cpu version
            # data_mat_freq_comb = np.reshape(data_mat,(data_mat.shape[0], int(prach_duration / div), div, -1))
            # data_mat_freq_comb = np.sum(data_mat_freq_comb, axis=2) / div
            #
            # x_u_mat = np.matlib.repmat(x_u, data_mat_freq_comb.shape[0], data_mat_freq_comb.shape[1])
            # x_u_mat = np.reshape(x_u_mat, (data_mat_freq_comb.shape[0], data_mat_freq_comb.shape[1], -1))
            #
            # corr_mat = np.multiply(np.conjugate(data_mat_freq_comb), x_u_mat)  # 8x28x139
            #
            # ifft_corr_mat = np.zeros((corr_mat.shape[0], corr_mat.shape[1], 1024), dtype=np.complex128)
            # ifft_corr_mat[:, :, :L_RA] = corr_mat
            #
            # ifft_corr_mat = ifft(ifft_corr_mat, axis=-1)
            #
            # # with warnings.catch_warnings():
            # #     warnings.simplefilter("ignore", np.ComplexWarning)
            #
            # ifft_corr_mat = np.absolute(ifft_corr_mat)
            # ifft_corr_mat = np.roll(ifft_corr_mat, shift=ifft_circ_shift, axis=-1)

            ## gpu cupy version

            data_mat_gpu = cp.asarray(data_mat)
            x_u_gpu = cp.asarray(x_u)

            data_mat_freq_comb = cp.reshape(data_mat_gpu, (data_mat_gpu.shape[0], int(prach_duration / div), div, -1))
            data_mat_freq_comb = cp.sum(data_mat_freq_comb, axis=2) / div

            x_u_mat = cp.tile(x_u_gpu, (data_mat_freq_comb.shape[0], data_mat_freq_comb.shape[1], 1))
            x_u_mat = cp.reshape(x_u_mat, (data_mat_freq_comb.shape[0], data_mat_freq_comb.shape[1], -1))

            corr_mat = cp.multiply(cp.conj(data_mat_freq_comb), x_u_mat)

            ifft_corr_mat = cp.zeros((corr_mat.shape[0], corr_mat.shape[1], 1024), dtype=cp.complex128)
            ifft_corr_mat[:, :, :L_RA] = corr_mat
            ifft_corr_mat = cp.fft.ifft(ifft_corr_mat, axis=-1)

            ifft_corr_mat = cp.abs(ifft_corr_mat)
            ifft_corr_mat = cp.roll(ifft_corr_mat, shift=ifft_circ_shift, axis=-1)

            ifft_corr_mat = cp.asnumpy(ifft_corr_mat)

            cp._default_memory_pool.free_all_blocks()


            # without non-coherent combining
            pre_idx_all_rx = []

            gaf_data = []
            for rx_idx in range(num_rx):
                for div_idx in range(int(prach_duration / div)):
                    sample_2d = np.array([ifft_corr_mat[rx_idx, div_idx, :]])

                    # plt.figure()
                    # plt.plot(sample_2d[0])
                    # plt.grid(True)
                    # plt.savefig(os.path.join(img_path, 'test_corr_gaf_MIMO.png'), dpi=300, bbox_inches='tight')
                    # plt.close()

                    gaf_img = gaf.fit_transform(sample_2d)
                    gaf_img = gaf_img[0]
                    img_3ch = np.stack([gaf_img, gaf_img, gaf_img], axis=0)

                    gaf_data.append(img_3ch)

            gaf_label = np.zeros((1, num_rx * int(prach_duration / div))) + label  # 1x224
            gaf_label = gaf_label[0]
            gaf_data = np.array(gaf_data)

            test_samples = create_single_datasets_tensor_data(gaf_data, gaf_label)

            test_dl = create_single_loaders_small_RAM(test_samples, bs=bs)

            for batch in test_dl:

                x, y_batch = [t.to(device) for t in batch]
                x = to_tensor_vgg_input(x).float()
                out = model(x)
                preds = F.log_softmax(out, dim=1).argmax(dim=1)
                preds = preds.tolist()
                pre_idx_all_rx += preds
                # pre_idx_all_rx.append(pre_idx_1rx)

            pre_idx_all_rx = np.array(pre_idx_all_rx)
            pre_idx_all_rx = np.reshape(pre_idx_all_rx, (-1, int(prach_duration / div)))
            pre_idx_final = st.mode(pre_idx_all_rx, axis=1)
            pre_idx_final = pre_idx_final.mode
            pre_idx_final = st.mode(pre_idx_final)

            ##############################
            pre_idx_without_rx_comb_1ite.append(pre_idx_final)
            ##############################

            # with non-coherent combining

            # ## numpy cpu version
            # ifft_corr_mat = np.sum(ifft_corr_mat, axis=0)

            ifft_corr_mat = cp.asarray(ifft_corr_mat)
            ifft_corr_mat = cp.sum(ifft_corr_mat, axis=0)
            ifft_corr_mat = cp.asnumpy(ifft_corr_mat)

            cp._default_memory_pool.free_all_blocks()

            pre_idx_1div = []

            gaf_data = []
            for div_idx in range(int(prach_duration / div)):
                sample_2d = np.array([ifft_corr_mat[div_idx, :]])

                # plt.figure()
                # plt.plot(sample_2d[0])
                # plt.grid(True)
                # plt.savefig(os.path.join(img_path, 'test_corr_gaf_MIMO.png'), dpi=300, bbox_inches='tight')
                # plt.close()

                gaf_img = gaf.fit_transform(sample_2d)
                gaf_img = gaf_img[0]
                img_3ch = np.stack([gaf_img, gaf_img, gaf_img], axis=0)

                gaf_data.append(img_3ch)

            gaf_label = np.zeros((1, int(prach_duration / div))) + label  # 1x224
            gaf_label = gaf_label[0]
            gaf_data = np.array(gaf_data)

            test_samples = create_single_datasets_tensor_data(gaf_data, gaf_label)

            test_dl = create_single_loaders_small_RAM(test_samples, bs=int(prach_duration / div))

            for batch in test_dl:
                x, y_batch = [t.to(device) for t in batch]
                x = to_tensor_vgg_input(x).float()
                out = model(x)
                preds = F.log_softmax(out, dim=1).argmax(dim=1)
                pre_idx_rx_comb = torch.mode(preds).values.item()
                pre_idx_1div.append(pre_idx_rx_comb)

            pre_idx_final = statistics.mode(pre_idx_1div)

            ##############################
            pre_idx_rx_comb_1ite.append(pre_idx_final)
            ##############################

        final_pre_idx_without_rx_comb_1ite = statistics.mode(pre_idx_without_rx_comb_1ite)
        final_pre_idx_rx_comb_1ite = statistics.mode(pre_idx_rx_comb_1ite)

        if final_pre_idx_without_rx_comb_1ite == label:
            tot_correct_without_rx_comb += 1

        if final_pre_idx_rx_comb_1ite == label:
            tot_correct_rx_comb += 1

    acc_without_rx_comb = tot_correct_without_rx_comb / tot_test_ite
    acc_rx_comb = tot_correct_rx_comb / tot_test_ite

    print(f'\nacc_without_rx_comb: {acc_without_rx_comb} \n'
          f'acc_rx_comb: {acc_rx_comb} \n')