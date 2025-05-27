import numpy as np
import os
import sys
from tqdm import tqdm
from matplotlib import pyplot as plt

sys.path.append("/home/sktt1anhhuy/prach_preamble_detection_AI")
from configuration import PrachConfig, CarrierConfig
from get_random_access_configuration import RandomAccessConfig
from get_ncs_root_cv import get_NCS, get_u, get_C_v

img_path = '/home/sktt1anhhuy/prach_preamble_detection_AI/image'
corr_data_test = '/home/sktt1anhhuy/prach_preamble_detection_AI/split_corr_data_dot_npy'

prach_config = PrachConfig()
##########
prach_config.preambleIndex = 10
##########

prach_config.prachConfigurationIndex = 158
prach_config.rootSequenceIndex = 39
prach_config.subcarrierSpacing = 30
prach_config.zeroCorrelationZoneConfig = 8
prach_config.frequencyRange = 'FR1'
prach_config.set = 'UnrestrictedSet'
prach_config.spectrumType = 'Unpaired'
prach_config.frequencyStart = 0

carrier_config = CarrierConfig()

carrier_config.n_UL_RB = 273
carrier_config.subcarrierSpacing = 30
carrier_config.numFrame = 1

random_access_config = RandomAccessConfig()
random_access_config.get_full_random_access_config(prach_config, carrier_config)

# time_error_tolerance = 0.26 # micro second, for AWGNs
time_error_tolerance = 1.77 # micro second, for TDLC300-100

len_time_domain_signal = 1228800
sample_time_micro_second = (10 / (len_time_domain_signal / carrier_config.numFrame)) * 10e3

N_CS = get_NCS(prach_config, random_access_config)

_, u_arr_unique = get_u(prach_config, random_access_config, N_CS)

_, C_v_arr = get_C_v(prach_config, random_access_config, N_CS)

num_preamble_per_seq = np.zeros(len(u_arr_unique))

match random_access_config.preambleFormat:
    case '0' | '1' | '2' | '3':
        ifft_len = 2048
    case default:
        ifft_len = 1024

for u_index in range(len(u_arr_unique)):
    if u_index < len(u_arr_unique) - 1:
        num_preamble_per_seq[u_index] = C_v_arr.size
    else:
        num_preamble_per_seq[u_index] = 64 - u_index * C_v_arr.size

max_num_preamble_per_seq = int(np.max(num_preamble_per_seq))
over_sampling = ifft_len / random_access_config.L_RA

snr_range = np.arange(-50, 31, 5)
constant = 4
save_plot = False

for snr in tqdm(snr_range):
    test_corr_data_name = 'corr_data_' + str(snr) + 'dB_test.npy'
    corr_data_label = np.load(os.path.join(corr_data_test, test_corr_data_name))

    corr_data = corr_data_label[:, :-1]
    corr_label = corr_data_label[:, -1]

    del corr_data_label

    num_detected_preamble = 0

    print(f"Testing on {test_corr_data_name}...\n")
    for sample_idx in tqdm(range(corr_data.shape[0])):

        if save_plot:
            sample_idx = 24*8

        x_corr = corr_data[sample_idx, :]
        x_corr_label = corr_label[sample_idx]

        if save_plot:
            plt.figure()
            plt.plot(x_corr)
            plt.grid(True)
            plt.tight_layout()

        start_sample_window = int(np.ceil(over_sampling * N_CS * (max_num_preamble_per_seq - 1)))
        end_sample_window = int(ifft_len)

        peak_window_sum = 0
        num_peak_window = 0

        max_peak_threshold = np.mean(x_corr) * constant

        if save_plot:
            plt.axhline(y=max_peak_threshold, color='r', linestyle='-')
            plt.axvline(x=start_sample_window, color='m', linestyle='-')
            plt.axvline(x=end_sample_window, color='m', linestyle='-')

        for ncv in range(max_num_preamble_per_seq):

            max_peak_window = np.max(x_corr[start_sample_window:end_sample_window])
            if max_peak_window > max_peak_threshold:
                peak_window_sum += max_peak_window
                num_peak_window += 1

            if save_plot:
                plt.axvline(x=start_sample_window, color='m', linestyle='-')
                plt.axvline(x=end_sample_window, color='m', linestyle='-')

            start_sample_window = int(end_sample_window % ifft_len)
            end_sample_window = int(np.ceil(over_sampling * (ncv + 1) * N_CS))

        new_max_peak_threshold = ((np.sum(x_corr) - peak_window_sum) / (ifft_len - num_peak_window)) * constant

        if save_plot:
            plt.axhline(y=new_max_peak_threshold, color='g', linestyle='-')
            plt.savefig(os.path.join(img_path, 'corr.png'), dpi=300, bbox_inches='tight')
            plt.close()

        if x_corr_label == len(C_v_arr):
            num_detected_preamble += 1
            continue

        if new_max_peak_threshold <= max_peak_threshold:
            new_max_peak_threshold = max_peak_threshold

        start_sample_window = int(np.ceil(over_sampling * N_CS * (max_num_preamble_per_seq - 1)))
        end_sample_window = ifft_len

        for ncv in range(max_num_preamble_per_seq):

            window = x_corr[start_sample_window:end_sample_window]

            max_peak_window = np.max(window)
            max_peak_window_sample = int(np.where(window == max_peak_window)[0][0])

            if max_peak_window > new_max_peak_threshold:

                n_index = ncv
                end_win = int(end_sample_window)

                n_TA = np.floor((end_win - (max_peak_window_sample + start_sample_window - 1)) / 8)
                n_TA_time_micro_second = sample_time_micro_second * n_index

                # if n_TA_time_micro_second <= time_error_tolerance:
                #     if n_index == x_corr_label:
                #         num_detected_preamble += 1

                if n_index == x_corr_label:
                    num_detected_preamble += 1

            start_sample_window = int(end_sample_window % ifft_len)
            end_sample_window = int(np.ceil(over_sampling * (ncv + 1) * N_CS))

    print(f"Accuracy = {(num_detected_preamble / len(corr_label))}%\n")

    del corr_data
    del corr_label








