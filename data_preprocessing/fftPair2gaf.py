import os
from tqdm import tqdm
import numpy as np
from pyts.image import GramianAngularField
import cupy as cp
import pdb
import mat73

train_fft_pair_data_path = '/home/hoang/prach_preamble_detection_AI/train_fft_pair_data'
test_fft_pair_data_path = '/home/hoang/prach_preamble_detection_AI/fft_pair_data'

gaf_corr_data_train_path = '/home/hoang/prach_preamble_detection_AI/gaf_corr_train_data'

num_rx = 8
snr_range = np.arange(-40, 1, 10)
div = 6
prach_duration = 12
L_RA = 139

for snr in tqdm(snr_range):
    print(f"\nGenerating training data {snr}dB\n")

    data_dict_mat = []
    
    # read 8 (might be 4, 2 which is num_rx) .mat file 
    for rx_idx in range(num_rx):

        file_name = 'pair_fft_data_' + str(snr) + 'dB_rx' + str(rx_idx + 1) + '.mat'

        #data_dict = mat73.loadmat(os.path.join(train_fft_pair_data_path, file_name))
        data_dict = mat73.loadmat(os.path.join(test_fft_pair_data_path, file_name))
        data_dict_mat.append(data_dict['data'])

    tot_num_samples = data_dict_mat[0].shape[0]
    tot_train_ite = int(tot_num_samples / prach_duration) # ??

    gaf_data = []
    gaf_label = []

    for sample_idx in tqdm(range(tot_train_ite)):
        start_sample_idx = sample_idx * prach_duration
        end_sample_idx = start_sample_idx + prach_duration

        data_mat = []
        
        # get 12 (prach_duration) samples of received 1x139 ZC sequences
        for rx in range(num_rx):
            data = data_dict_mat[rx][start_sample_idx:end_sample_idx, :L_RA]
            data_mat.append(data)
        # data_mat => 12x139
        
        pdb.set_trace()

        # get 1 samples of correct sequence index 1x139 ZC sequence, and reduced preamble index as label
        # other 11 samples are similar
        x_u = data_dict_mat[0][start_sample_idx, L_RA:-1]
        label = np.abs(data_dict_mat[0][start_sample_idx, -1])

        # 8x12x139
        data_mat = np.array(data_mat)
        # push data_mat to GPU
        data_mat = cp.asarray(data_mat)
        # 8x2x6x139
        data_mat = cp.reshape(data_mat, (data_mat.shape[0], int(prach_duration / div), div, -1))
        # coherent combining -> 8x2x139
        data_mat = cp.sum(data_mat, axis=2) / div

        # replicate correct sequence index ZC sequence to match the shape of data_mat
        x_u_mat = cp.tile(x_u_gpu, (data_mat.shape[0], data_mat.shape[1], 1))
        # 8x2x139
        x_u_mat = cp.reshape(x_u_mat, (data_mat.shape[0], data_mat.shape[1], -1))
        
        # fft-based cross-correlation
        corr_mat = cp.multiply(cp.conj(data_mat), x_u_mat)
        
        # transform to time-domain 1024-point signal
        ifft_corr_mat = cp.zeros((corr_mat.shape[0], corr_mat.shape[1], 1024), dtype=cp.complex128)
        ifft_corr_mat[:, :, :L_RA] = corr_mat
        ifft_corr_mat = cp.fft.ifft(ifft_corr_mat, axis=-1)

        ifft_corr_mat = cp.abs(ifft_corr_mat)

        # shift left
        ifft_corr_mat = cp.roll(ifft_corr_mat, shift=ifft_circ_shift, axis=-1)

        ifft_corr_mat = cp.asnumpy(ifft_corr_mat)

        cp._default_memory_pool.free_all_blocks()
        
        # non-coherent combining => 2x1024
        ifft_corr_mat = np.sum(ifft_corr_mat, axis=0) / num_rx
        

        # ifft_corr_mat = np.array([ifft_corr_mat])
        # convert each 1024-point signal in 2x1024 to 32x32 single GAF form
        #   and stack 3 replications to form 3x32x32 "image"
        for sub_sample_idx in range(int(prach_duration / div)):
            ifft_corr = np.array([ifft_corr_mat[sub_sample_idx, :]])
            gaf_img = gaf.fit_transform(ifft_corr)
            gaf_img = gaf_img[0]
            img_3ch = np.stack([gaf_img, gaf_img, gaf_img], axis=0)
            gaf_data.append(img_3ch)
            gaf_label.append(label)

    gaf_data = np.array(gaf_data)
    gaf_label = np.array(gaf_label)

    gaf_data_name = 'gaf' + 'corr_data_' + str(snr) + '_dB_' + type + '_info.npy'
    gaf_label_name = 'gaf' + 'corr_data_' + str(snr) + '_dB_' + type + '_label.npy'

    print(f'Saving {gaf_data_name}...')
    np.save(os.path.join(gaf_corr_train_data_path, gaf_data_name), gaf_data)
    print('done!')

    print(f'Saving {gaf_label_name}...')
    np.save(os.path.join(gaf_corr_train_data_path, gaf_label_name), gaf_label)
    print('done!')


