from configuration import PrachConfig, CarrierConfig
from get_random_access_configuration import RandomAccessConfig
from get_ncs_root_cv import get_NCS, get_u, get_C_v
from prach_modulation_demodulation import prach_modulation
import numpy as np
from numpy.fft import fft
from prach_ofdm_info import PachOFDMInfo
from pyphysim.reference_signals.zadoffchu import calcBaseZC
from pyphysim.channels.fading import TdlChannel, TdlChannelProfile
from pyphysim.channels.fading_generators import JakesSampleGenerator
from commpy.channels import awgn

prach_config = PrachConfig()

prach_config.preambleIndex = 60
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

prach_config.display_config()
print('')
carrier_config.display_config()
print('')
random_access_config.display_random_access_config()
print('')

N_CS = get_NCS(prach_config, random_access_config)
print('-----------------Sequence Configuration-----------------')
print(f"N_CS = {N_CS}")

u, u_arr_unique = get_u(prach_config, random_access_config, N_CS)
print(f"u = {u}")
# print(u_arr_unique)

C_v, C_v_arr = get_C_v(prach_config, random_access_config, N_CS)
print(f"C_v = {C_v}")
# print(C_v_arr)

x_u = calcBaseZC(random_access_config.L_RA, u)
# print('----------------x_u-----------------')
# print(x_u)

x_uv = np.roll(x_u, -C_v)
# print('----------------x_uv-----------------')
# print(x_uv)

y_uv = fft(x_uv)
# print('----------------y_uv-----------------')
# print(y_uv)

prach_ofdm_information = PachOFDMInfo()
prach_ofdm_information.getPrachOFDMInfo(prach_config, random_access_config)
print('')
prach_ofdm_information.display_prach_ofdm_info()

time_domain_signal, start_mapping_symbol_arr, end_mapping_symbol_arr = prach_modulation(prach_config, carrier_config,
                                                                                    random_access_config)
# print(start_mapping_symbol_arr)
# print(end_mapping_symbol_arr)
print('')
print('-----------------Channel Configuration-----------------')
print("Channel: TDLC300-100")

bandwidth = 100e6  # in Hetz
Fd = 100  # Doppler frequency (in Hz)
Ts = time_domain_signal.size  # Sampling interval

tap_powers_dB = np.array([-6.9, 0, -7.7, -2.5, -2.4, -9.9, -8.0, -6.6, -7.1, -13.0, -14.2, -16.0])

tap_delays = np.array([0, 65, 70, 190, 195, 200, 240, 325, 520, 1045, 1510, 2595])

num_tx_antennas = 1
num_rx_antennas = 2
print('')
print('-----------------MIMO Configuration-----------------')
print(f"num_tx_antennas = {num_tx_antennas}")
print(f"num_rx_antennas = {num_rx_antennas}")

jakesObj = JakesSampleGenerator(Fd, Ts, L=np.size(tap_powers_dB))
tdlChanlelProfile = TdlChannelProfile(tap_powers_dB=tap_powers_dB, tap_delays=tap_delays)
tdlChannel = TdlChannel(jakesObj, channel_profile=tdlChanlelProfile)

tdlChannel.set_num_antennas(num_rx_antennas=num_rx_antennas, num_tx_antennas=num_tx_antennas)

# print(f"tdlChanlelProfile.rms_delay_spread = {tdlChanlelProfile.rms_delay_spread}\n")

snr_dB_range = range(-40, 40, 5)
snr_dB = 0
received_test_signal = awgn(time_domain_signal, snr_dB=snr_dB)
received_test_signal = tdlChannel.corrupt_data(received_test_signal)

default_numSample_perSlot = 30720
numSample_persubframe = int(default_numSample_perSlot * (carrier_config.subcarrierSpacing / 15) * 2)
numSample_perFrame = numSample_persubframe * 10

# preamble_arr = np.array([])
preamble_arr = []

# for snr_dB in snr_dB_range:
for antenna_index in range(num_rx_antennas):
    single_antenna_signal = received_test_signal[antenna_index, :]
    # preamble_frame = np.array([])
    preamble_frame = []
    for frame_index in range(carrier_config.numFrame):
        # single_frame_signal = single_antenna_signal[(frame_index*numSample_perFrame):(frame_index*numSample_perFrame + numSample_perFrame)]

        # preamble_slot = np.array([])
        preamble_slot = []
        for slot_index in range(start_mapping_symbol_arr.shape[1]):
            start_mapping_symbol = start_mapping_symbol_arr[frame_index, slot_index]
            end_mapping_symbol = end_mapping_symbol_arr[frame_index, slot_index]

            # single_slot_signal = single_frame_signal[start_mapping_symbol:(end_mapping_symbol + 1)]
            single_slot_signal = single_antenna_signal[start_mapping_symbol:(end_mapping_symbol + 1)]
            # preamble_slot = np.concatenate((preamble_slot, single_slot_signal), axis=1)
            preamble_slot.append(single_slot_signal)
        # preamble_frame = np.concatenate((preamble_frame, preamble_slot))
        preamble_frame.append(preamble_slot)
    # preamble_arr = np.concatenate((preamble_arr, preamble_frame))
    preamble_arr.append(preamble_frame)


preamble_arr_np = np.array(preamble_arr)
print('')
print('-----------------Preamble Data shape-----------------')
print(f"preamble_arr_shape = {preamble_arr_np.shape}")














