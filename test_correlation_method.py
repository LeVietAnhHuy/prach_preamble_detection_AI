import numpy as np
import os
from matplotlib import pyplot as plt
from tqdm import tqdm

from configuration import PrachConfig, CarrierConfig
from get_random_access_configuration import RandomAccessConfig
from get_ncs_root_cv import get_NCS, get_u, get_C_v

generated_data_dir = 'generated_dataset'
config_data_dir = 'corr_antenna_gain_combining_dataset'
config_data_path = os.path.join(generated_data_dir, config_data_dir)

dataset_names = ['rx_2_corr_rmsGainComb_freqComb.npy']

dataset_dir = os.path.join(config_data_path, dataset_names[-1])

data = np.load(dataset_dir)
num_samples = data.shape[0]

Y = data[:, -1].astype(int)
X = np.delete(data, -1, 1)

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

time_error_tolerance = 0.26 # micro second, for AWGNs
# time_error_tolerance = 1.77 # micro second, for TDLC300-100

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

constant = 27
num_detected_preamble = 0

print(f"Testing on {dataset_names[-1]}...")
for sample_index in tqdm(range(num_samples)):
    xcorr = X[sample_index, :]
    # fig, ax = plt.subplots()
    # ax.plot(xcorr)
    # plt.show()
    # plt.close(fig='all')
    start_sample_window = int(np.ceil(over_sampling * N_CS * (max_num_preamble_per_seq - 1)))
    end_sample_window = int(ifft_len)

    peak_window_sum = 0
    num_peak_window = 0

    max_peak_threshold = np.mean(xcorr) * constant


    for ncv in range(max_num_preamble_per_seq):

        max_peak_window = np.max(xcorr[start_sample_window:end_sample_window])
        if max_peak_window > max_peak_threshold:
            peak_window_sum += max_peak_window
            num_peak_window += 1

        start_sample_window = int(end_sample_window % ifft_len)
        end_sample_window = int(np.ceil(over_sampling * (ncv + 1) * N_CS))

    new_max_peak_threshold = ((np.sum(xcorr) - peak_window_sum) / (ifft_len - num_peak_window)) * constant

    if new_max_peak_threshold <= max_peak_threshold:
        new_max_peak_threshold = max_peak_threshold

    start_sample_window = int(np.ceil(over_sampling * N_CS * (max_num_preamble_per_seq - 1)))
    end_sample_window = ifft_len

    for ncv in range(max_num_preamble_per_seq):

        window = xcorr[start_sample_window:end_sample_window]

        max_peak_window = np.max(window)
        max_peak_window_sample = int(np.where(window == max_peak_window)[0][0])

        if max_peak_window > new_max_peak_threshold:

            n_index = ncv
            end_win = int(end_sample_window)

            n_TA = np.floor((end_win - (max_peak_window_sample + start_sample_window - 1)) / 8)
            n_TA_time_micro_second = sample_time_micro_second * n_index

            if n_TA_time_micro_second <= time_error_tolerance:
                if n_index == Y[sample_index]:
                    num_detected_preamble += 1

print(f"Accuracy = {(num_detected_preamble / num_samples) * 100}%")








