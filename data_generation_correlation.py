from configuration import PrachConfig, CarrierConfig
from get_random_access_configuration import RandomAccessConfig
from get_ncs_root_cv import get_NCS, get_u, get_C_v
from prach_modulation_demodulation import prach_modulation
import numpy as np
from numpy.fft import fft, fftshift, ifft
import torch
import matplotlib.pyplot as plt

from prach_ofdm_info import PachOFDMInfo
from pyphysim.reference_signals.zadoffchu import calcBaseZC
from pyphysim.channels.fading import TdlChannel, TdlChannelProfile
from pyphysim.channels.fading_generators import JakesSampleGenerator
from commpy.channels import awgn
import os
from tqdm import tqdm
import math
from antenna_gain_combining_methods import selection_combining, switch_combining, equal_gain_combining, \
    rms_gain_combining

# from get_snr import snr_db_from_received

generated_data_dir = 'generated_dataset'
config_data_dir = 'corr_dataset'

image_dir = 'image'
image_name_corr = 'corr.png'
full_image_path_corr = os.path.join(image_dir, image_name_corr)

tap_powers_dB = np.array([-6.9, 0, -7.7, -2.5, -2.4, -9.9, -8.0, -6.6, -7.1, -13.0, -14.2, -16.0])

tap_delays = np.array([0, 65, 70, 190, 195, 200, 240, 325, 520, 1045, 1510, 2595])

num_tx_antennas = 1
num_rx_antennas = 2

print('')
print('-----------------MIMO Configuration-----------------')
print(f"num_tx_antennas = {num_tx_antennas}")
print(f"num_rx_antennas = {num_rx_antennas}")

bandwidth = 100e6  # in Hetz
Fd = 100  # Doppler frequency (in Hz)

start_snr_dB = 15
end_snr_dB = 16
step_snr_dB = 1

print('')
print('-----------------Noise-----------------')
print("TDLC300-100")
print(f"AWGN: {start_snr_dB}dB -> {end_snr_dB}dB, step = {step_snr_dB}")
print('')

all_x_u_arr = []
all_preamble_arr = []
all_preamble_start_mapping_symbol_arr = []
all_preamble_end_mapping_symbol_arr = []

preamble_index_range = range(0, 64)

for preindex in preamble_index_range:
    prach_config = PrachConfig()

    ##########
    prach_config.preambleIndex = preindex
    ##########20

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

    N_CS = get_NCS(prach_config, random_access_config)

    u, u_arr_unique = get_u(prach_config, random_access_config, N_CS)

    C_v, C_v_arr = get_C_v(prach_config, random_access_config, N_CS)

    x_u = calcBaseZC(random_access_config.L_RA, u)
    all_x_u_arr.append(x_u)

    x_uv = np.roll(x_u, -C_v)

    y_uv = fft(x_uv)

    prach_ofdm_information = PachOFDMInfo()
    prach_ofdm_information.getPrachOFDMInfo(prach_config, random_access_config)

    (time_domain_signal,
     start_mapping_symbol_arr,
     end_mapping_symbol_arr,
     PrachStartingResourceElementIndex_freqDomain) = prach_modulation(prach_config,
                                                                      carrier_config,
                                                                      random_access_config)
    all_preamble_arr.append(time_domain_signal)
    all_preamble_start_mapping_symbol_arr.append(start_mapping_symbol_arr)
    all_preamble_end_mapping_symbol_arr.append(end_mapping_symbol_arr)

Ts = all_preamble_arr[0].size  # Sampling interval

jakesObj = JakesSampleGenerator(Fd, Ts, L=np.size(tap_powers_dB))
tdlChanlelProfile = TdlChannelProfile(tap_powers_dB=tap_powers_dB, tap_delays=tap_delays)
tdlChannel = TdlChannel(jakesObj, channel_profile=tdlChanlelProfile)

tdlChannel.set_num_antennas(num_rx_antennas=num_rx_antennas, num_tx_antennas=num_tx_antennas)

snr_dB_range = range(start_snr_dB, end_snr_dB, step_snr_dB)

default_numSample_perSlot = 30720
numSample_persubframe = int(default_numSample_perSlot * (carrier_config.subcarrierSpacing / 15) * 2)
numSample_perFrame = numSample_persubframe * 10

divisors = []
for i in range(1, int(random_access_config.prachDuration / 2) + 1):
    if random_access_config.prachDuration % i == 0:
        divisors.append(i)
divisors.append(random_access_config.prachDuration)

iff_circ_shift = -9

# divisor = divisors[-1]


num_sample_per_snr = 150000
num_sample_target_preamble_index = 1000
target_preamble_index = 60

num_Cv = 0

print('Generating data:')
for snr_dB in tqdm(snr_dB_range):
    num_sample = 0

    dataset = []
    while num_sample <= num_sample_per_snr:
        for preamble_index in range(60, 64):
            #preamble_index = np.random.randint(64)
            # if num_sample <= num_sample_target_preamble_index:
            #     preamble_index = target_preamble_index

            # received_test_signal = awgn(all_preamble_arr[preamble_index], snr_dB=snr_dB)
            # received_test_signal = tdlChannel.corrupt_data(received_test_signal)

            received_test_signal = tdlChannel.corrupt_data(all_preamble_arr[preamble_index])
            for antenna_index in range(num_rx_antennas):
                received_test_signal[antenna_index, :] = awgn(received_test_signal[antenna_index, :], snr_dB=snr_dB) # (num_rx, 1228800)

            frame_index = 0
            slot_index = 0
            start_mapping_symbol = all_preamble_start_mapping_symbol_arr[preamble_index][frame_index, slot_index]
            end_mapping_symbol = all_preamble_end_mapping_symbol_arr[preamble_index][frame_index, slot_index]
            # Extract preamble
            received_test_signal = received_test_signal[:, start_mapping_symbol:(end_mapping_symbol + 1)] # (num_rx, 51088)
            # antenna gain combining methods
            # received_test_signal = selection_combining(received_test_signal)
            # antenna_gain_combining_signal = rms_gain_combining(received_test_signal)

            received_test_signal = received_test_signal[:, prach_ofdm_information.cyclicPrefixLen:] # (num_rx, 49152)
            received_test_signal = np.reshape(received_test_signal, (num_rx_antennas, random_access_config.prachDuration, -1)) # (num_rx, 12, 4096)

            # received_test_signal = fft(received_test_signal, axis=-1)
            #received_test_signal = fftshift(received_test_signal, axes=-1)

            received_test_signal = torch.from_numpy(received_test_signal).to('cuda')
            received_test_signal = torch.fft.fft(received_test_signal, dim=-1)
            received_test_signal = torch.fft.fftshift(received_test_signal, dim=-1)
            received_test_signal = received_test_signal.cpu().numpy()

            received_test_signal = received_test_signal[:, :, PrachStartingResourceElementIndex_freqDomain:(PrachStartingResourceElementIndex_freqDomain + random_access_config.L_RA)] # (num_rx, 12, 139)
            received_test_signal = received_test_signal * math.sqrt(received_test_signal.shape[-1])

            # num_sum_prach_rep = np.random.choice(divisors)
            num_sum_prach_rep = 1
            num_block_prach = int(random_access_config.prachDuration / num_sum_prach_rep)

            received_test_signal = np.reshape(received_test_signal, (num_block_prach, num_rx_antennas, num_sum_prach_rep, -1))

            freq_comb_signal = np.sum(received_test_signal, axis=2)
            received_freq_comb_x_uv_fft_mat = freq_comb_signal / num_sum_prach_rep
            num_Cv += 1
            if(num_Cv == C_v_arr.size):
                preamble_index = abs(preamble_index - 20)

            x_u_mat = np.matlib.repmat(all_x_u_arr[preamble_index], num_block_prach, num_rx_antennas)
            x_u_mat = np.reshape(x_u_mat, (num_block_prach, num_rx_antennas, -1))

            # x_u_fft_mat = fft(x_u_mat, axis=-1)

            x_u_mat = torch.from_numpy(x_u_mat).to('cuda')
            x_u_mat = torch.fft.fft(x_u_mat, dim=-1)
            x_u_mat = x_u_mat.cpu().numpy()

            xcorr = np.conj(received_freq_comb_x_uv_fft_mat) * x_u_mat

            match random_access_config.preambleFormat:
                case '0' | '1' | '2' | '3':
                    ifft_len = 2048
                case default:
                    ifft_len = 1024

            x_corr_ifft = np.zeros((num_block_prach, num_rx_antennas, ifft_len))
            x_corr_ifft[:, :, :random_access_config.L_RA] = xcorr
            #x_corr_ifft = ifft(x_corr_ifft, axis=-1)

            x_corr_ifft = torch.from_numpy(x_corr_ifft).to('cuda')
            x_corr_ifft = torch.fft.ifft(x_corr_ifft, dim=-1)
            x_corr_ifft = x_corr_ifft.cpu().numpy()

            x_corr_ifft = np.abs(x_corr_ifft)
            plt.plot(x_corr_ifft[0, 0, :])
            full_image_path_corr = os.path.join(image_dir, 'abs_corr.png')
            plt.savefig(full_image_path_corr, dpi=300)
            print(f"Plot saved as '{full_image_path_corr}'")
            plt.close()

            x_corr_ifft = np.roll(x_corr_ifft, shift=iff_circ_shift, axis=-1)
            plt.plot(x_corr_ifft[0, 0, :])
            full_image_path_corr = os.path.join(image_dir, 'abs_roll_corr.png')
            plt.savefig(full_image_path_corr, dpi=300)
            print(f"Plot saved as '{full_image_path_corr}'")
            plt.close()



            window_idx = preamble_index % C_v_arr.size
            if (num_Cv == C_v_arr.size):
                window_idx = C_v_arr.size
                num_Cv = 0



            for i in range(x_corr_ifft.shape[0]):
                for j in range(x_corr_ifft.shape[1]):

                    plt.plot(x_corr_ifft[i, j, :])
                    plt.savefig(full_image_path_corr, dpi=300)
                    print(f"Plot saved as '{full_image_path_corr}'")
                    plt.close()

                    ########################################
                    break
                    ########################################
                    x_corr = np.append(x_corr_ifft[i, j, :], window_idx)  # label

                    dataset.append(x_corr)
                    print(f"\n{num_sample}/{num_sample_per_snr}, PreIdx = {preamble_index}, windowIdx = {window_idx}, num_sum_prach_rep = {num_sum_prach_rep}, {snr_dB}dB")
                    num_sample += 1

            # for block_idx in range(num_block_prach):
            #     # x_corr_ifft_rx_gain = selection_=  ombining(x_corr_ifft[block_idx, :, :])
            #     x_corr_ifft_rx_gain = rms_gain_combining(x_corr_ifft[block_idx, :, :])
            #
            #     window_idx = preamble_index % C_v_arr.size
            #
            #     x_corr_ifft_rx_gain = np.append(x_corr_ifft_rx_gain, window_idx)  # label
            #     dataset.append(x_corr_ifft_rx_gain)
            #
            #     print(f"\n{num_sample}/{num_sample_per_snr}, PreIdx = {preamble_index}, windowIdx = {window_idx}, num_sum_prach_rep = {num_sum_prach_rep}, {snr_dB}dB")
            #
            #     num_sample += 1

    dataset_np = np.array(dataset)
    config_data_path = os.path.join(generated_data_dir, config_data_dir)
    os.makedirs(config_data_path, exist_ok=True)

    dataset_name = 'corr_' + str(snr_dB) + 'dB.npy'

    dataset_dir = os.path.join(config_data_path, dataset_name)

    print('')
    print(f"Saving data to {dataset_dir}...")
    np.save(dataset_dir, dataset_np)
    print('Done!')

    del dataset_np
    del dataset




