from configuration import PrachConfig, CarrierConfig
from get_random_access_configuration import RandomAccessConfig
from get_ncs_root_cv import get_NCS, get_u, get_C_v
from prach_modulation_demodulation import prach_modulation
import numpy as np
from numpy.fft import fft, fftshift
from prach_ofdm_info import PachOFDMInfo
from pyphysim.reference_signals.zadoffchu import calcBaseZC
from pyphysim.channels.fading import TdlChannel, TdlChannelProfile
from pyphysim.channels.fading_generators import JakesSampleGenerator
from commpy.channels import awgn
import os
from tqdm import tqdm

tap_powers_dB = np.array([-6.9, 0, -7.7, -2.5, -2.4, -9.9, -8.0, -6.6, -7.1, -13.0, -14.2, -16.0])

tap_delays = np.array([0, 65, 70, 190, 195, 200, 240, 325, 520, 1045, 1510, 2595])

num_tx_antennas = 1
num_rx_antennas = 1

print('')
print('-----------------MIMO Configuration-----------------')
print(f"num_tx_antennas = {num_tx_antennas}")
print(f"num_rx_antennas = {num_rx_antennas}")

bandwidth = 100e6  # in Hetz
Fd = 100  # Doppler frequency (in Hz)

start_snr_dB = -40
end_snr_dB = 21
step_snr_dB = 5

print('')
print('-----------------Noise-----------------')
print("TDLC300-100")
print(f"AWGN: {start_snr_dB}dB -> {end_snr_dB}dB, step = {step_snr_dB}")
print('')

all_preamble_arr = []
all_preamble_start_mapping_symbol_arr = []
all_preamble_end_mapping_symbol_arr = []

preamble_index_range = range(0, 64)

for preindex in preamble_index_range:
    prach_config = PrachConfig()

    ##########
    prach_config.preambleIndex = preindex
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

    N_CS = get_NCS(prach_config, random_access_config)

    u, u_arr_unique = get_u(prach_config, random_access_config, N_CS)

    C_v, C_v_arr = get_C_v(prach_config, random_access_config, N_CS)

    x_u = calcBaseZC(random_access_config.L_RA, u)

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

sum_num = 12

dataset = []
num_sample_per_snr = 5000
num_sample_target_preamble_index = 1000
target_preamble_index = 60

print('Generating data:')
for snr_dB in tqdm(snr_dB_range):
    num_sample = 1

    while num_sample <= num_sample_per_snr:

        preamble_index = np.random.randint(64)
        if num_sample <= num_sample_target_preamble_index:
            preamble_index = target_preamble_index

        # received_test_signal = awgn(all_preamble_arr[preamble_index], snr_dB=snr_dB)
        # received_test_signal = tdlChannel.corrupt_data(received_test_signal)


        received_test_signal = tdlChannel.corrupt_data(all_preamble_arr[preamble_index])
        received_test_signal = awgn(received_test_signal, snr_dB=snr_dB)

        for antenna_index in range(num_rx_antennas):
            single_antenna_signal = received_test_signal[antenna_index, :]

            for frame_index in range(carrier_config.numFrame):

                preamble_slot = []
                freq_comb_fft_pre_arr = []
                for slot_index in range(all_preamble_start_mapping_symbol_arr[preamble_index].shape[1]):
                    start_mapping_symbol = all_preamble_start_mapping_symbol_arr[preamble_index][frame_index, slot_index]
                    end_mapping_symbol = all_preamble_end_mapping_symbol_arr[preamble_index][frame_index, slot_index]

                    single_slot_signal = single_antenna_signal[start_mapping_symbol:(end_mapping_symbol + 1)]
                    single_slot_signal_without_cp = single_slot_signal[prach_ofdm_information.cyclicPrefixLen:]
                    single_slot_signal_arr = np.reshape(single_slot_signal_without_cp, (random_access_config.prachDuration, -1))
                    single_slot_signal_fft_arr = fft(single_slot_signal_arr)

                    multiplier = int(random_access_config.prachDuration / sum_num)
                    num_col = single_slot_signal_fft_arr.shape[1]
                    signal_fft_multiplier = np.reshape(single_slot_signal_fft_arr, (multiplier, sum_num, num_col))

                    for multiple_index in range(multiplier):
                        avg_freq_signal = np.sum(signal_fft_multiplier[multiple_index, :, :], axis=0) / sum_num
                        avg_freq_signal_fftshift = fftshift(avg_freq_signal)

                        freq_sequence = avg_freq_signal_fftshift[PrachStartingResourceElementIndex_freqDomain:(PrachStartingResourceElementIndex_freqDomain + random_access_config.L_RA)]
                        freq_sequence = np.append(freq_sequence, preamble_index)   # label
                        dataset.append(freq_sequence)

                        print(f"\n{num_sample}/{num_sample_per_snr}, {snr_dB}dB")

                        num_sample += 1

dataset_np = np.array(dataset)
print('')
print('-----------------Preamble Data shape-----------------')
print(f"preamble_arr_shape = {dataset_np.shape}")

generated_data_dir = 'generated_dataset'
config_data_dir = 'pi_' + str(prach_config.preambleIndex) + \
                       '_pci_' + str(prach_config.prachConfigurationIndex) + \
                       '_rsi_' + str(prach_config.rootSequenceIndex) + \
                       '_prscs_' + str(prach_config.subcarrierSpacing) + \
                       '_puscs_' + str(carrier_config.subcarrierSpacing) + \
                       '_zczc_' + str(prach_config.zeroCorrelationZoneConfig) + \
                       '_fr_' + prach_config.frequencyRange + \
                       '_s_' + prach_config.set + \
                       '_st_' + prach_config.spectrumType + \
                       '_fs_' + str(prach_config.frequencyStart) + \
                       '_snrRange_' + str(start_snr_dB) + '_' + str(end_snr_dB) + '_' + str()

config_data_path = os.path.join(generated_data_dir, config_data_dir)
os.makedirs(config_data_path, exist_ok=True)
dataset_name = 'rx_' + str(num_rx_antennas) + '_freqComb_' + str(sum_num)  + '_numFrame_' + str(carrier_config.numFrame) + '.npy'
dataset_dir = os.path.join(config_data_path, dataset_name)

print('')
print(f"Saving data to {dataset_dir}...")
np.save(dataset_dir, dataset_np)
print('Done!')